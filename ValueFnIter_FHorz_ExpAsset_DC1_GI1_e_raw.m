function [V,Policy]=ValueFnIter_FHorz_ExpAsset_DC1_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1 = prod(n_d1);
N_d2 = prod(n_d2);
has_d1 = (N_d1 > 0); Nd1_eff = max(N_d1, 1);
has_d2 = (N_d2 > 0); Nd2_eff = max(N_d2, 1);
has_d = has_d1 | has_d2;
Nd_eff = Nd1_eff * Nd2_eff;
d_offset = double(has_d); % 1 if true, 0 if false
% ensures CreateReturnFnMatrix handles it correctly as having no d1
if ~has_d1; d_gridvals = d2_gridvals; n_d1 = 0; end

N_a1 = prod(n_a1); has_a1 = (N_a1 > 0); Na1_eff = max(N_a1, 1);
N_a2 = prod(n_a2);
N_a = Na1_eff * N_a2;

% Check explicit vfoptions flags to see which exogenous states are passed to aprimeFn
has_exp_z = vfoptions.experienceassetz == 1 || vfoptions.experienceassetze == 1;
has_exp_e = vfoptions.experienceassete == 1 || vfoptions.experienceassetze == 1;
has_exp_u = vfoptions.experienceassetu == 1;

N_z = prod(n_z); Nz_eff = max(N_z, 1);
if N_z == 0
    pi_z_J = ones(1, 1, N_j);
    z_gridvals_J = zeros(1, 1, N_j);
    pass_n_z = 0; pass_z_grid = [];
elseif has_exp_z
    pass_n_z = n_z;
end

N_e = prod(vfoptions.n_e);
if N_e == 0
    pi_e_J = ones(1, 1, N_j);
    e_gridvals_J = zeros(1, 1, N_j);
    pass_n_e = 0; pass_e_grid = [];
elseif has_exp_e
    e_gridvals_J = vfoptions.e_gridvals_J;
    pass_n_e = vfoptions.n_e;
end

V=zeros(N_a,Nz_eff,N_e,N_j,'gpuArray');
Policy=zeros(3 + d_offset,N_a,Nz_eff,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
Policy(3 + d_offset,:,:,:,:)=2; % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==0
    midpoint=zeros(Nd_eff,1,Na1_eff,N_a2,Nz_eff,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(Nd_eff,1,Na1_eff,N_a2,Nz_eff,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint=zeros(Nd_eff,1,Na1_eff,N_a2,'gpuArray');
end

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
Na1_effprime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1); % already includes -1
zind=shiftdim(gpuArray(0:1:Nz_eff-1),-3); % already includes -1
zindB=shiftdim(gpuArray(0:1:Nz_eff-1),-1); % already includes -1
eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
zeindB=zindB+Nz_eff*eind; % already includes -1

a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2); % already includes -1
d2ind=repelem(gpuArray(1:1:Nd2_eff)',Nd1_eff,1); % [N_d,1]; maps full d-index to d2-component

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
 if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoint(:,1,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n2long, n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(a1primeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2,Nz_eff,N_e]; Level=2, Refine=0
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,Nd_eff)+1;
        allind=d_ind+Nd_eff*aind+Nd_eff*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        if has_d
            Policy(1,:,:,:,N_j) = d_ind; % Combined d index (really d2)
        end
        Policy(1+d_offset,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(2+d_offset,:,:,:,N_j)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/Nd_eff);
        linidx_lower = d_ind                     + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zeindB;
        linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zeindB;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(3+d_offset,:,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    [~,maxindex]=max(ReturnMatrix_ii_e,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n2long, n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexes), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2,Nz_eff]; Level=2, Refine=0
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_e,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,Nd_eff)+1;
            allind=d_ind+Nd_eff*aind+Nd_eff*N_a*zindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_z
            if has_d
                Policy(1,:,:,e_c,N_j) = d_ind; % Combined d index (really d2)
            end
            Policy(1+d_offset,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(2+d_offset,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/Nd_eff);
            linidx_lower = d_ind                     + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zindB;
            linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zindB;
            isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(3+d_offset,:,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        end
    elseif vfoptions.lowmemory==2
        for z_c=1:Nz_eff
            if N_z>0; z_val = z_gridvals_J(z_c, :, N_j); else; z_val = []; end
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoint(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                        [~,maxindex]=max(ReturnMatrix_ii_ze,[],2);
                        midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d2-1-by-n_a1-by-n_a2
                a1primeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
                % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexes), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2]; Level=2, Refine=0
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_ze,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,Nd_eff)+1;
                allind=d_ind+Nd_eff*aind; % midpoint is n_d2-by-1-by-n_a1-by-n_a2
                if has_d
                    Policy(1,:,z_c,e_c,N_j) = d_ind; % Combined d index (really d2)
                end
                Policy(1+d_offset,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(2+d_offset,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/Nd_eff);
                linidx_lower = d_ind                     + Nd_eff*n2long*aind;
                linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind;
                isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(3+d_offset,:,z_c,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=sum(shiftdim(pi_e_J(:,N_j+1),-2).*reshape(vfoptions.V_Jplus1,[N_a,Nz_eff,N_e]),3); % First, switch V_Jplus1 into Kron form
    if N_z > 0; EVpre = EVpre * pi_z_J(:,:,N_j)'; end

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    if has_exp_z
        pass_z_grid = z_gridvals_J(:,:,N_j);
    end
    if has_exp_e
        pass_e_grid = e_gridvals_J(:,:,N_j);
    end
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2, pass_n_z, pass_z_grid, pass_n_e, pass_e_grid); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [Nd2_eff,N_a2], whereas aprimeProbs is [Nd2_eff,N_a2]

    if length(n_a2)==1
        a1_offsets=repelem(gpuArray(1:1:Na1_eff)',Nd2_eff,N_a2);
        a2_idx_exp=repmat(a2primeIndex,Na1_eff,1,1); % Expands correctly across 3D
        aprimeProbs=repmat(a2primeProbs,Na1_eff,1,1);

        aprimeIndex_full=a1_offsets+Na1_eff*(a2_idx_exp-1);
        aprimeplus1Index_full=a1_offsets+Na1_eff*a2_idx_exp;

        % aprimeIndex=repelem(gpuArray(1:1:Na1_eff)',Nd2_eff,N_a2)+Na1_eff*repmat(a2primeIndex-1,Na1_eff,1,1); % [Nd2_eff*Na1_eff,N_a2]
        % aprimeplus1Index=repelem(gpuArray(1:1:Na1_eff)',Nd2_eff,N_a2)+Na1_eff*repmat(a2primeIndex,Na1_eff,1,1); % [Nd2_eff*Na1_eff,N_a2]
        % aprimeProbs=repmat(a2primeProbs,Na1_eff,1,Nz_eff); % [Nd2_eff*Na1_eff,N_a2,Nz_eff]

        % Vlower=reshape(EV(aprimeIndex(:),:),[Nd2_eff*Na1_eff,N_a2,Nz_eff]);
        % Vupper=reshape(EV(aprimeplus1Index(:),:),[Nd2_eff*Na1_eff,N_a2,Nz_eff]);

        z_offset = shiftdim((0:Nz_eff-1) * N_a, -1);

        Vlower = EVpre(aprimeIndex_full + z_offset);
        Vupper = EVpre(aprimeplus1Index_full + z_offset);

        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,Nd2_eff,N_a2], per-dim factored (lower-grid
        % index and prob of lower, one row per a2 dim). Fold to the four corners keeping the
        % a1prime offset, then nested 2-corner interp with skipinterp at each level and
        % per-contribution NaN cleanup for 0*(-Inf).
        n_a2_1=n_a2(1);
        loIdx_1=squeeze(a2primeIndex(1,:,:,:));
        loIdx_2=squeeze(a2primeIndex(2,:,:,:));
        prob_1_exp=repmat(squeeze(a2primeProbs(1,:,:,:)),Na1_eff,1,1);
        prob_2_exp=repmat(squeeze(a2primeProbs(2,:,:,:)),Na1_eff,1,1);

        a1_offsets=repelem(gpuArray(1:1:Na1_eff)',Nd2_eff,N_a2);
        aprime_ll=a1_offsets+Na1_eff*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,Na1_eff,1,1);
        aprime_hl=a1_offsets+Na1_eff*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,Na1_eff,1,1);
        aprime_lh=a1_offsets+Na1_eff*repmat(loIdx_1+n_a2_1*loIdx_2-1,Na1_eff,1,1);
        aprime_hh=a1_offsets+Na1_eff*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,Na1_eff,1,1);

        z_offset = shiftdim((0:Nz_eff-1) * N_a, -1);

        V_ll = EVpre(aprime_ll + z_offset);
        V_hl = EVpre(aprime_hl + z_offset);
        V_lh = EVpre(aprime_lh + z_offset);
        V_hh = EVpre(aprime_hh + z_offset);
        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    % Already applied the probabilities from interpolating onto grid

    EV(isnan(EV))=0; % NOT SURE THIS IS NEEDED??? EV is over (d2,a1prime,a2)

    if vfoptions.lowmemory==0

        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[Nd2_eff,Na1_eff,1,N_a2,Nz_eff,N_e]);
        % Interpolate EV over aprime_grid
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);   % [Nd2_eff,Na1_effprime,1,N_a2,Nz_eff,N_e]
        % d1-dim is implicit singleton in DiscountedEV/DiscountedEVinterp, broadcasts at use sites

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV,Nd1_eff,1,1,1,1); % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoint(:,1,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                d2aprimez=d2ind+Nd2_eff*(a1primeindexes-1)+Nd2_eff*Na1_eff*a2ind+Nd2_eff*Na1_eff*N_a2*zind; % [N_d,maxgap+1,1,N_a2,Nz_eff,N_e]; linear index into DiscountedEV [Nd2_eff,Na1_eff,1,N_a2,Nz_eff]
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d2aprimez);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n2long, n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2,Nz_eff,N_e]; Level=2, Refine=0
        da1primea2z=d2ind+Nd2_eff*(a1primeindexesfine-1)+Nd2_eff*Na1_effprime*a2ind+Nd2_eff*Na1_effprime*N_a2*zind; % [N_d,n2long,Na1_eff,N_a2,Nz_eff,N_e]; linear index into DiscountedEVinterp [Nd2_eff,Na1_effprime,1,N_a2,Nz_eff]
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z),[Nd_eff*n2long,Na1_eff*N_a2,Nz_eff,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,Nd_eff)+1;
        allind=d_ind+Nd_eff*aind+Nd_eff*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        if has_d
            Policy(1,:,:,:,N_j) = d_ind; % Combined d index (really d2)
        end
        Policy(1+d_offset,:,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(2+d_offset,:,:,:,N_j)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/Nd_eff);
        linidx_lower = d_ind                     + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zeindB;
        linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zeindB;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(3+d_offset,:,:,:,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            DiscountedEV_e = DiscountFactorParamsVec * reshape(EV(:,:,:,e_c), [Nd2_eff, Na1_eff, 1, N_a2, Nz_eff]);
            DiscountedEVinterp_e = permute(interp1(a1_gridvals, permute(DiscountedEV_e, [2,1,3,4,5]), a1prime_grid), [2,1,3,4,5]);

            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV_e,Nd1_eff,1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    d2aprimez=d2ind+Nd2_eff*(a1primeindexes-1)+Nd2_eff*Na1_eff*a2ind+Nd2_eff*Na1_eff*N_a2*zind; % [N_d,maxgap+1,1,N_a2,Nz_eff]; linear index into DiscountedEV [Nd2_eff,Na1_eff,1,N_a2,Nz_eff]
                    entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV_e(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2,Nz_eff]; Level=2, Refine=0
            da1primea2z=d2ind+Nd2_eff*(a1primeindexesfine-1)+Nd2_eff*Na1_effprime*a2ind+Nd2_eff*Na1_effprime*N_a2*zind; % [N_d,n2long,Na1_eff,N_a2,Nz_eff]; linear index into DiscountedEVinterp [Nd2_eff,Na1_effprime,1,N_a2,Nz_eff]
            entireRHS_ii_e=ReturnMatrix_ii_e+reshape(DiscountedEVinterp_e(da1primea2z),[Nd_eff*n2long,Na1_eff*N_a2,Nz_eff]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            V(:,:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,Nd_eff)+1;
            allind=d_ind+Nd_eff*aind+Nd_eff*N_a*zindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_z
            if has_d
                Policy(1,:,:,e_c,N_j) = d_ind; % Combined d index (really d2)
            end
            Policy(1+d_offset,:,:,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(2+d_offset,:,:,e_c,N_j)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/Nd_eff);
            linidx_lower = d_ind                     + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zindB;
            linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zindB;
            isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(3+d_offset,:,:,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        end
    elseif vfoptions.lowmemory==2

        for z_c=1:Nz_eff
            if N_z>0; z_val = z_gridvals_J(z_c, :, N_j); else; z_val = []; end
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_ze = DiscountFactorParamsVec * reshape(EV(:,:,z_c,e_c), [Nd2_eff, Na1_eff, 1, N_a2]);
                DiscountedEVinterp_ze = permute(interp1(a1_gridvals, permute(DiscountedEV_ze, [2,1,3,4]), a1prime_grid), [2,1,3,4]);

                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_ze,Nd1_eff,1,1,1);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoint(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                        d2aprime=d2ind+Nd2_eff*(a1primeindexes-1)+Nd2_eff*Na1_eff*a2ind; % [N_d,maxgap+1,1,N_a2]; linear index into DiscountedEV_z [Nd2_eff,Na1_eff,1,N_a2]
                        entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_ze(d2aprime);
                        [~,maxindex]=max(entireRHS_ii_ze,[],2);
                        midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2]; Level=2, Refine=0
                da1primea2=d2ind+Nd2_eff*(a1primeindexesfine-1)+Nd2_eff*Na1_effprime*a2ind; % [N_d,n2long,Na1_eff,N_a2]; linear index into DiscountedEVinterp_z [Nd2_eff,Na1_effprime,1,N_a2]
                entireRHS_ii_ze=ReturnMatrix_ii_ze+reshape(DiscountedEVinterp_ze(da1primea2),[Nd_eff*n2long,Na1_eff*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                V(:,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,Nd_eff)+1;
                allind=d_ind+Nd_eff*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
                if has_d
                    Policy(1,:,z_c,e_c,N_j) = d_ind; % Combined d index (really d2)
                end
                Policy(1+d_offset,:,z_c,e_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(2+d_offset,:,z_c,e_c,N_j)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/Nd_eff);
                linidx_lower = d_ind                     + Nd_eff*n2long*aind;
                linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind;
                isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(3+d_offset,:,z_c,e_c,N_j) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
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
    if has_exp_z
        pass_z_grid = z_gridvals_J(:,:,jj);
    end
    if has_exp_e
        pass_e_grid = e_gridvals_J(:,:,jj);
    end
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2, pass_n_z, pass_z_grid, pass_n_e, pass_e_grid); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [Nd2_eff,N_a2], whereas aprimeProbs is [Nd2_eff,N_a2]

    EVpre=sum(shiftdim(pi_e_J(:,jj+1),-2).*V(:,:,:,jj+1),3); % First, switch V_Jplus1 into Kron form
    if N_z > 0; EVpre = EVpre * pi_z_J(:,:,jj)'; end

    if length(n_a2)==1
        a1_offsets=repelem(gpuArray(1:1:Na1_eff)',Nd2_eff,N_a2);
        a2_idx_exp=repmat(a2primeIndex,Na1_eff,1,1); % Expands correctly across 3D
        aprimeProbs=repmat(a2primeProbs,Na1_eff,1,1);

        aprimeIndex_full=a1_offsets+Na1_eff*(a2_idx_exp-1);
        aprimeplus1Index_full=a1_offsets+Na1_eff*a2_idx_exp;

        z_offset = shiftdim((0:Nz_eff-1) * N_a, -1);

        Vlower = EVpre(aprimeIndex_full + z_offset);
        Vupper = EVpre(aprimeplus1Index_full + z_offset);

        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,Nd2_eff,N_a2], per-dim factored (lower-grid
        % index and prob of lower, one row per a2 dim). Fold to the four corners keeping the
        % a1prime offset, then nested 2-corner interp with skipinterp at each level and
        % per-contribution NaN cleanup for 0*(-Inf).
        n_a2_1=n_a2(1);
        loIdx_1=squeeze(a2primeIndex(1,:,:,:));
        loIdx_2=squeeze(a2primeIndex(2,:,:,:));
        prob_1_exp=repmat(squeeze(a2primeProbs(1,:,:,:)),Na1_eff,1,1);
        prob_2_exp=repmat(squeeze(a2primeProbs(2,:,:,:)),Na1_eff,1,1);

        a1_offsets=repelem(gpuArray(1:1:Na1_eff)',Nd2_eff,N_a2);
        aprime_ll=a1_offsets+Na1_eff*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,Na1_eff,1,1);
        aprime_hl=a1_offsets+Na1_eff*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,Na1_eff,1,1);
        aprime_lh=a1_offsets+Na1_eff*repmat(loIdx_1+n_a2_1*loIdx_2-1,Na1_eff,1,1);
        aprime_hh=a1_offsets+Na1_eff*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,Na1_eff,1,1);

        z_offset = shiftdim((0:Nz_eff-1) * N_a, -1);

        V_ll = EVpre(aprime_ll + z_offset);
        V_hl = EVpre(aprime_hl + z_offset);
        V_lh = EVpre(aprime_lh + z_offset);
        V_hh = EVpre(aprime_hh + z_offset);
        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    % Already applied the probabilities from interpolating onto grid

    EV(isnan(EV))=0; % NOT SURE THIS IS NEEDED? EV is over (d2,a1prime,a2)

    if vfoptions.lowmemory==0

        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[Nd2_eff,Na1_eff,1,N_a2,Nz_eff,N_e]);
        % Interpolate EV over aprime_grid
        DiscountedEVinterp=permute(interp1(a1_gridvals,permute(DiscountedEV,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);   % [Nd2_eff,Na1_effprime,1,N_a2,Nz_eff,N_e]
        % d1-dim is implicit singleton in DiscountedEV/DiscountedEVinterp, broadcasts at use sites

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV,Nd1_eff,1,1,1,1); % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoint(:,1,level1ii,:,:,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                d2aprimez=d2ind+Nd2_eff*(a1primeindexes-1)+Nd2_eff*Na1_eff*a2ind+Nd2_eff*Na1_eff*N_a2*zind; % [N_d,maxgap+1,1,N_a2,Nz_eff,N_e]; linear index into DiscountedEV [Nd2_eff,Na1_eff,1,N_a2,Nz_eff]
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d2aprimez);
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
        % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2-by-n_z-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n2long, n_a1,n_a2,n_z,n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2,Nz_eff,N_e]; Level=2, Refine=0
        da1primea2z=d2ind+Nd2_eff*(a1primeindexesfine-1)+Nd2_eff*Na1_effprime*a2ind+Nd2_eff*Na1_effprime*N_a2*zind; % [N_d,n2long,Na1_eff,N_a2,Nz_eff,N_e]; linear index into DiscountedEVinterp [Nd2_eff,Na1_effprime,1,N_a2,Nz_eff]
        entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEVinterp(da1primea2z),[Nd_eff*n2long,Na1_eff*N_a2,Nz_eff,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,Nd_eff)+1;
        allind=d_ind+Nd_eff*aind+Nd_eff*N_a*zeindB; % midpoint is n_d-by-1-by-n_a1-by-n_a2-by-n_z-by-n_e
        if has_d
            Policy(1,:,:,:,jj) = d_ind; % Combined d index (really d2)
        end
        Policy(1+d_offset,:,:,:,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
        Policy(2+d_offset,:,:,:,jj)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/Nd_eff);
        linidx_lower = d_ind                     + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zeindB;
        linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zeindB;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(3+d_offset,:,:,:,jj) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            DiscountedEV_e = DiscountFactorParamsVec * reshape(EV(:,:,:,e_c), [Nd2_eff, Na1_eff, 1, N_a2, Nz_eff]);
            DiscountedEVinterp_e = permute(interp1(a1_gridvals, permute(DiscountedEV_e, [2,1,3,4,5]), a1prime_grid), [2,1,3,4,5]);

            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV_e,Nd1_eff,1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoint(:,1,level1ii,:,:)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                    d2aprimez=d2ind+Nd2_eff*(a1primeindexes-1)+Nd2_eff*Na1_eff*a2ind+Nd2_eff*Na1_eff*N_a2*zind; % [N_d,maxgap+1,1,N_a2,Nz_eff]; linear index into DiscountedEV [Nd2_eff,Na1_eff,1,N_a2,Nz_eff]
                    entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV_e(d2aprimez);
                    [~,maxindex]=max(entireRHS_ii_e,[],2);
                    midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d2-1-by-n_a1-by-n_a2-by-n_z
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
            % aprime possibilities are n_d2-by-n2long-by-n_a1-by-n_a2-by-n_z
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n2long,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2,Nz_eff]; Level=2, Refine=0
            da1primea2z=d2ind+Nd2_eff*(a1primeindexesfine-1)+Nd2_eff*Na1_effprime*a2ind+Nd2_eff*Na1_effprime*N_a2*zind; % [N_d,n2long,Na1_eff,N_a2,Nz_eff]; linear index into DiscountedEVinterp [Nd2_eff,Na1_effprime,1,N_a2,Nz_eff]
            entireRHS_ii_e=ReturnMatrix_ii_e+reshape(DiscountedEVinterp_e(da1primea2z),[Nd_eff*n2long,Na1_eff*N_a2,Nz_eff]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            V(:,:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,Nd_eff)+1;
            allind=d_ind+Nd_eff*aind+Nd_eff*N_a*zindB; % midpoint is n_d2-by-1-by-n_a1-by-n_a2-by-n_z
            if has_d
                Policy(1,:,:,e_c,jj) = d_ind; % Combined d index (really d2)
            end
            Policy(1+d_offset,:,:,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
            Policy(2+d_offset,:,:,e_c,jj)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/Nd_eff);
            linidx_lower = d_ind                     + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zindB;
            linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind + Nd_eff*n2long*N_a*zindB;
            isInfLower = (ReturnMatrix_ii_e(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii_e(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(3+d_offset,:,:,e_c,jj) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

        end
    elseif vfoptions.lowmemory==2

        for z_c=1:Nz_eff
            if N_z>0; z_val = z_gridvals_J(z_c, :, jj); else; z_val = []; end
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_ze = DiscountFactorParamsVec * reshape(EV(:,:,z_c,e_c), [Nd2_eff, Na1_eff, 1, N_a2]);
                DiscountedEVinterp_ze = permute(interp1(a1_gridvals, permute(DiscountedEV_ze, [2,1,3,4]), a1prime_grid), [2,1,3,4]);

                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_ze,Nd1_eff,1,1,1);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoint(:,1,level1ii,:)=maxindex1;

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)'; % just a1
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),Na1_eff-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0); % Level 3 as DC1+GI; Level=3, Refine=0
                        d2aprime=d2ind+Nd2_eff*(a1primeindexes-1)+Nd2_eff*Na1_eff*a2ind; % [N_d,maxgap+1,1,N_a2]; linear index into DiscountedEV_z [Nd2_eff,Na1_eff,1,N_a2]
                        entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_ze(d2aprime);
                        [~,maxindex]=max(entireRHS_ii_ze,[],2);
                        midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                % Turn this into the 'midpoint'
                midpoint=max(min(midpoint,n_a1(1)-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d-1-by-n_a1-by-n_a2
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint, fine index
                % aprime possibilities are n_d-by-n2long-by-n_a1-by-n_a2
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n2long, n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % [N_d,Na1_effprime,Na1_eff,N_a2]; Level=2, Refine=0
                da1primea2=d2ind+Nd2_eff*(a1primeindexesfine-1)+Nd2_eff*Na1_effprime*a2ind; % [N_d,n2long,Na1_eff,N_a2]; linear index into DiscountedEVinterp_z [Nd2_eff,Na1_effprime,1,N_a2]
                entireRHS_ii_ze=ReturnMatrix_ii_ze+reshape(DiscountedEVinterp_ze(da1primea2),[Nd_eff*n2long,Na1_eff*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_ze,[],1);
                V(:,z_c,e_c,jj)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,Nd_eff)+1;
                allind=d_ind+Nd_eff*aind; % midpoint is n_d-by-1-by-n_a1-by-n_a2
                if has_d
                    Policy(1,:,z_c,e_c,jj) = d_ind; % Combined d index (really d2)
                end
                Policy(1+d_offset,:,z_c,e_c,jj)=shiftdim(squeeze(midpoint(allind)),-1); % a1prime midpoint
                Policy(2+d_offset,:,z_c,e_c,jj)=shiftdim(ceil(maxindexL2/Nd_eff),-1); % a1primeL2ind

                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/Nd_eff);
                linidx_lower = d_ind                     + Nd_eff*n2long*aind;
                linidx_upper = d_ind + Nd_eff*(n2long-1) + Nd_eff*n2long*aind;
                isInfLower = (ReturnMatrix_ii_ze(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii_ze(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                Policy(3+d_offset,:,z_c,e_c,jj) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    end
end





%% With grid interpolation, which from midpoint to lower grid index
% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2+d_offset,:,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1+d_offset,:,:,:,:)=Policy(1+d_offset,:,:,:,:)-adjust; % lower grid point
Policy(2+d_offset,:,:,:,:)=Policy(2+d_offset,:,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% %% For experience asset, just output Policy as single index and then use Case2 to UnKron
% Policy=shiftdim(Policy(1,:,:,:,:)+N_d*(Policy(1+d_offset,:,:,:,:)-1)+N_d*Na1_eff*(Policy(2+d_offset,:,:,:,:)-1)+N_d*Na1_eff*(n2short+2)*(Policy(3+d_offset,:,:,:,:)-1),1);


end
