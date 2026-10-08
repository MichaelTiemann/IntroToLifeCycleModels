function [V,Policy]=ValueFnIter_FHorz_ExpAsset_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1_raw=prod(n_d1);
N_d2_raw=prod(n_d2);
if N_d1_raw == 0 && N_d2_raw == 0
    N_d_raw = 0;
elseif N_d1_raw == 0
    N_d_raw = N_d2_raw;
elseif N_d2_raw == 0
    N_d_raw = N_d1_raw;
else
    N_d_raw = N_d1_raw * N_d2_raw;
end
has_d=(N_d_raw > 0);
has_d1=(N_d1_raw > 0);
N_d1=max(N_d1_raw, 1);
N_d2=max(N_d2_raw, 1);
N_d=N_d1 * N_d2;

if ~has_d1
    d_gridvals = d2_gridvals;
    n_d1 = 0; % ensures CreateReturnFnMatrix handles it correctly as having no d1
end

N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1 * N_a2;

N_z_raw=prod(n_z);
has_z=(N_z_raw > 0);
N_z=max(N_z_raw, 1);

if ~has_z
    z_gridvals_J=zeros(1, 1, N_j);
    n_z=0; % Tell CreateReturnFnMatrix to omit z
    pi_z_J=ones(1, 1, N_j);
elseif size(z_gridvals_J, 3) < N_j
    z_gridvals_J=repmat(z_gridvals_J, 1, 1, N_j);
    pi_z_J=repmat(pi_z_J, 1, 1, N_j);
end

N_e=prod(n_e);

if size(e_gridvals_J, 2) < N_j + 1
    e_gridvals_J=repmat(e_gridvals_J, 1, 1, N_j);
    pi_e_J=repmat(pi_e_J, 1, N_j + 1);
end

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
Policy(4,:,:,:,:)=2; % 1=all weight to lower coarse a1, 2=usual linear weights, 3=all weight to upper coarse a1

%%
% n_a1prime=n_a1;
% a1prime_gridvals=a1_gridvals;
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e),'gpuArray');
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
end

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
zind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1
zindB=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
zeindB=zindB+N_z*shiftdim((0:1:N_e-1),-2); % already includes -1

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

d2_for_l2 = rem(gpuArray(1:1:N_d)' - 1, N_d2) + 1; % For indexing into N_d2 later

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        ReturnMatrix=reshape(ReturnMatrix, [N_d, N_a1, N_a1, N_a2, N_z, N_e]);

        % Calc the max and it's index
        [~,maxindex]=max(ReturnMatrix,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_e]; Level=2, Refine=0
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy(1,:,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset     =ceil(maxindexL2/N_d);
        linidx_lower =d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zeindB;
        linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zeindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,:,N_j)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            ReturnMatrix_e=reshape(ReturnMatrix_e, [N_d, N_a1, N_a1, N_a2, N_z, 1]);

            % Calc the max and it's index
            [~,maxindex]=max(ReturnMatrix_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policy(1,:,:,e_c,N_j)=d_ind; % d2
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset     =ceil(maxindexL2/N_d);
            linidx_lower =d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zindB;
            linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,:,e_c,N_j)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
                ReturnMatrix_ze=reshape(ReturnMatrix_ze, [N_d, N_a1, N_a1, N_a2, 1, 1]);

                % Calc the max and it's index
                [~,maxindex]=max(ReturnMatrix_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
                Policy(1,:,z_c,e_c,N_j)=d_ind; % d2
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
                L2offset     =ceil(maxindexL2/N_d);
                linidx_lower =d_ind                  + N_d*n2long*aind;
                linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind;
                isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(4,:,z_c,e_c,N_j)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    if vfoptions.experienceassete | vfoptions.experienceassetze
        [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2,n_z,z_gridvals_J(:,:,N_j),n_e,e_gridvals_J(:,:,N_j));
    else
        [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    end
    % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    EVpre=sum(shiftdim(pi_e_J(:,N_j+1),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % First, switch V_Jplus1 into Kron form
    EV=InterpolateExpAssetEV(EVpre, n_a2, N_d2, N_a1, N_a2, N_z, a2primeIndex, a2primeProbs);
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]; Level=1, Refine=0
        ReturnMatrix=reshape(ReturnMatrix, [N_d1, N_d2, N_a1, N_a1, N_a2, N_z, N_e]);
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]);
        % Interpolate EV over aprime_grid
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z,N_e]
        entireRHS = ReturnMatrix + shiftdim(DiscountedEV, -1);
        entireRHS = reshape(entireRHS, [N_d, N_a1, N_a1, N_a2, N_z, N_e]);

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]; Level=2, Refine=0
        da1primea2z = d2_for_l2 + N_d2*(a1primeindexesfine-1) + N_d2*N_a1prime*a2ind + N_d2*N_a1prime*N_a2*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z(:)),[N_d*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy(1,:,:,:,N_j)=d_ind; % d2
        Policy(2,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset     =ceil(maxindexL2/N_d);
        linidx_lower =d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zeindB;
        linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zeindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,:,N_j)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2,N_z]; Level=1, Refine=0
            ReturnMatrix_e=reshape(ReturnMatrix_e, [N_d1, N_d2, N_a1, N_a1, N_a2, N_z, 1]);
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z,N_e]
            entireRHS_e = ReturnMatrix_e + shiftdim(DiscountedEV, -1);
            entireRHS_e = reshape(entireRHS_e, [N_d, N_a1, N_a1, N_a2, N_z, 1]);

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_z]; Level=2, Refine=0
            da1primea2z = d2_for_l2 + N_d2*(a1primeindexesfine-1) + N_d2*N_a1prime*a2ind + N_d2*N_a1prime*N_a2*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z(:)),[N_d*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policy(1,:,:,e_c,N_j)=d_ind; % d2
            Policy(2,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset     =ceil(maxindexL2/N_d);
            linidx_lower =d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zindB;
            linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,:,e_c,N_j)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountFactorParamsVec*reshape(EV(:,:,:,:,z_c),[N_d2,N_a1,1,N_a2,1]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp_z=permute(interp1(a1_gridvals,permute(DiscountedEV_z,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z,N_e]
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=1, Refine=0
                ReturnMatrix_ze=reshape(ReturnMatrix_ze, [N_d1, N_d2, N_a1, N_a1, N_a2, 1, 1]);
                entireRHS_ze = ReturnMatrix_ze + shiftdim(DiscountedEV_z, -1);
                entireRHS_ze = reshape(entireRHS_ze, [N_d, N_a1, N_a1, N_a2, 1, 1]);

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
                da1primea2 = d2_for_l2 + N_d2*(a1primeindexesfine-1) + N_d2*N_a1prime*a2ind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(da1primea2(:)),[N_d*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
                Policy(1,:,z_c,e_c,N_j)=d_ind; % d2
                Policy(2,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
                L2offset     =ceil(maxindexL2/N_d);
                linidx_lower =d_ind                  + N_d*n2long*aind;
                linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind;
                isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(4,:,z_c,e_c,N_j)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
            end
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
        if vfoptions.experienceassete | vfoptions.experienceassetze
        [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2,n_z,z_gridvals_J(:,:,jj),n_e,e_gridvals_J(:,:,jj));
    else
        [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    end
    % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    EVpre=sum(shiftdim(pi_e_J(:,jj+1),-2).*V(:,:,:,jj+1),3); % First, switch V_Jplus1 into Kron form
    EV=InterpolateExpAssetEV(EVpre, n_a2, N_d2, N_a1, N_a2, N_z, a2primeIndex, a2primeProbs);
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    if vfoptions.lowmemory==0

        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]);
        % Interpolate EV over aprime_grid
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z]

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]; Level=1, Refine=0
        ReturnMatrix=reshape(ReturnMatrix, [N_d1, N_d2, N_a1, N_a1, N_a2, N_z, N_e]);

        entireRHS=ReturnMatrix+shiftdim(DiscountedEV,-1); % autofill 3rd dim to N_a1
        entireRHS = reshape(entireRHS, [N_d, N_a1, N_a1, N_a2, N_z, N_e]);

        % Calc the max and it's index
        [~,maxindex]=max(entireRHS,[],2);

        % Turn this into the 'midpoint'
        midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_z,N_e]; Level=2, Refine=0
        da1primea2z = d2_for_l2 + N_d2*(a1primeindexesfine-1) + N_d2*N_a1prime*a2ind + N_d2*N_a1prime*N_a2*zind;
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z(:)),[N_d*n2long,N_a1*N_a2,N_z,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        Policy(1,:,:,:,jj)=d_ind; % d2
        Policy(2,:,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(3,:,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
        % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
        L2offset     =ceil(maxindexL2/N_d);
        linidx_lower =d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zeindB;
        linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zeindB;
        isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(4,:,:,:,jj)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z]

            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2,N_z]; Level=1, Refine=0
            ReturnMatrix_e=reshape(ReturnMatrix_e, [N_d1, N_d2, N_a1, N_a1, N_a2, N_z, 1]);

            entireRHS_e=ReturnMatrix_e+shiftdim(DiscountedEV,-1); % autofill 3rd dim to N_a1
            entireRHS_e = reshape(entireRHS_e, [N_d, N_a1, N_a1, N_a2, N_z, 1]);

            % Calc the max and it's index
            [~,maxindex]=max(entireRHS_e,[],2);

            % Turn this into the 'midpoint'
            midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2,N_z]; Level=2, Refine=0
            da1primea2z = d2_for_l2 + N_d2*(a1primeindexesfine-1) + N_d2*N_a1prime*a2ind + N_d2*N_a1prime*N_a2*zind;
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z(:)),[N_d*n2long,N_a1*N_a2,N_z]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*zindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z
            Policy(1,:,:,e_c,jj)=d_ind; % d2
            Policy(2,:,:,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(3,:,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
            % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
            L2offset     =ceil(maxindexL2/N_d);
            linidx_lower =d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*zindB;
            linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*zindB;
            isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(4,:,:,e_c,jj)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);

            DiscountedEV_z=DiscountFactorParamsVec*reshape(EV(:,:,:,:,z_c),[N_d2,N_a1,1,N_a2,1]);
            % Interpolate EV over aprime_grid
            DiscountedEVinterp_z=permute(interp1(a1_gridvals,permute(DiscountedEV_z,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]); % [N_d2,N_a1prime,1,N_a2,N_z,N_e]

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=1, Refine=0
                ReturnMatrix_ze=reshape(ReturnMatrix_ze, [N_d1, N_d2, N_a1, N_a1, N_a2, 1, 1]);
                entireRHS_ze = ReturnMatrix_ze + shiftdim(DiscountedEV_z, -1);
                entireRHS_ze = reshape(entireRHS_ze, [N_d, N_a1, N_a1, N_a2, 1, 1]);

                % Calc the max and it's index
                [~,maxindex]=max(entireRHS_ze,[],2);

                % Turn this into the 'midpoint'
                midpoint=max(min(maxindex,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,N_a1prime,N_a1,N_a2]; Level=2, Refine=0
                da1primea2=d2_for_l2+N_d*(a1primeindexesfine-1)+N_d*N_a1prime*a2ind;
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp_z(da1primea2(:)),[N_d*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d)+1;
                allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
                Policy(1,:,z_c,e_c,jj)=d_ind; % d2
                Policy(2,:,z_c,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(3,:,z_c,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % a1primeL2ind
                % L2 flag: detect -Inf on the coarse a1 neighbour we'd put weight on (at chosen d)
                L2offset     =ceil(maxindexL2/N_d);
                linidx_lower =d_ind                  + N_d*n2long*aind;
                linidx_upper =d_ind + N_d*(n2long-1) + N_d*n2long*aind;
                isInfLower   =(ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper   =(ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict=(L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict=(L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(4,:,z_c,e_c,jj)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), -1);
            end
        end
    end
end



%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:,:)=Policy(2,:,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% %% For experience asset, just output Policy as single index and then use Case2 to UnKron
% Policy=shiftdim(Policy3(1,:,:,:,:)+N_d*(Policy3(2,:,:,:,:)-1)+N_d*N_a1*(Policy3(3,:,:,:,:)-1)+N_d*N_a1*(n2short+2)*(Policy(4,:,:,:,:)-1),1);

if ~has_d
    Policy = Policy(2:end, :, :, :, :);
end


end
