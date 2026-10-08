function [V,Policy]=ValueFnIter_FHorz_SemiExo_DC1_GI1_raw(n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_d=[n_d1,n_d2];
N_d1_raw=prod(n_d1);
N_d2_raw=prod(n_d2);
has_d1 = (N_d1_raw > 0); N_d1 = max(N_d1_raw, 1);
has_d2 = (N_d2_raw > 0); N_d2 = max(N_d2_raw, 1);
if ~has_d1
    n_d = n_d2;
end
N_d = N_d1 * N_d2; % Needed for N_j when converting to form of Policy
d_total = has_d1 + has_d2; % Dynamically sets Policy size

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z_raw=prod(n_z);
has_z = (N_z_raw > 0); N_z = max(N_z_raw, 1);
N_bothz = N_semiz * N_z;
if ~has_z
    n_bothz = n_semiz;
else
    n_bothz = [n_semiz, n_z];
end

V=zeros(N_a,N_bothz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy=zeros(3+d_total,N_a,N_bothz,N_j,'gpuArray'); % first dim indexes the optimal choice for d1,d2,aprime and aprime2 (in GI layer)
Policy(3+d_total,:,:,:)=2; % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
if has_d1 && has_d2
    special_n_d = [n_d1, ones(1, max(length(n_d2), 1))];
    d_gridvals = [repmat(d1_gridvals, N_d2, 1), repelem(d2_gridvals, N_d1, 1)];
    % version to use when looping over d2
    d12_gridvals = permute(reshape(d_gridvals, [N_d1, N_d2, max(length(n_d1)+length(n_d2), 1)]), [1, 3, 2]);
elseif has_d1
    special_n_d = n_d1;
    d_gridvals = d1_gridvals;
    d12_gridvals = d1_gridvals;
elseif has_d2
    special_n_d = ones(1, max(length(n_d2), 1));
    d_gridvals = d2_gridvals;
    % version to use when looping over d2
    d12_gridvals = permute(d2_gridvals, [3, 2, 1]);
else
    special_n_d = [];
    d_gridvals = [];
    d12_gridvals = [];
end

aind=gpuArray(0:1:N_a-1); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
bothzind2=shiftdim(gpuArray(0:1:N_bothz-1),-2);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
    semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1 (loop z, vectorize semiz)
    semizind2=shiftdim(gpuArray(0:1:N_semiz-1),-2);
elseif vfoptions.lowmemory==2
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

if has_z
    bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];
else
    bothz_gridvals_J=semiz_gridvals_J;
end

% Preallocate
V_ford2_jj=zeros(N_a,N_bothz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_bothz,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_bothz,N_d2,'gpuArray');
PolicyL2flag_ford2_jj=2*ones(N_a,N_bothz,N_d2,'gpuArray');
% Preallocate
if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d1,1,N_a,N_bothz,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(N_d1,1,N_a,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoints_jj=zeros(N_d1,1,N_a,'gpuArray');
end

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);


if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        midpoints_Nj=zeros(N_d,1,N_a,N_bothz,'gpuArray');

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_bothz, d_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        % Treat standard problem as just being the first layer
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Just keep the 'midpoint' version of maxindex1 [as GI]
        midpoints_Nj(:,1,level1ii,:)=maxindex1;

        % Second level based on monotonicity
        maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
        maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
        if has_d1
            maxgap = max(maxgap, [], 1); % Max over d1
        end
        maxgap = squeeze(maxgap);

        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_bothz, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
            end
        end

        % Turn this into the 'midpoint'
        midpoints_Nj=max(min(midpoints_Nj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-1-by-n_a-by-n_z
        aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_bothz,d_gridvals,aprime_grid(aprimeindexes),a_grid,bothz_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*bothzind; % midpoint is n_d-by-1-by-n_a-by-n_z
        curr_offset = 1;
        if has_d1
            Policy(curr_offset,:,:,N_j)=shiftdim(rem(d_ind - 1, N_d1) + 1, -1); %d1
            curr_offset = curr_offset + 1;
        end
        if has_d2
            Policy(curr_offset,:,:,N_j)=shiftdim(ceil(d_ind / N_d1), -1); %d2
        end

        Policy(d_total+1,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)), -1); % midpoint
        Policy(d_total+2,:,:,N_j)=shiftdim(ceil(maxindexL2 / N_d), -1); % aprimeL2ind

        % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
        L2offset = ceil(maxindexL2/N_d);
        linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
        linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*bothzind;
        isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        Policy(d_total+3,:,:,N_j)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)), -1);

    elseif vfoptions.lowmemory==1 % parallel over semiz, loop over z
        midpoints_Nj=zeros(N_d,1,N_a,N_semiz,'gpuArray');

        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, [n_semiz,special_n_z], d_gridvals, a_grid, a_grid(level1ii), z_valblock, ReturnFnParamsVec,1);
            % Treat standard problem as just being the first layer
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_Nj(:,1,level1ii,:)=maxindex1;

            % Second level based on monotonicity
            maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
            maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
            if has_d1
                maxgap = max(maxgap, [], 1); % Max over d1
            end
            maxgap = squeeze(maxgap);

            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, [n_semiz,special_n_z], d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, ReturnFnParamsVec,3);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_Nj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_Nj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                end
            end

            % Turn this into the 'midpoint'
            midpoints_Nj=max(min(midpoints_Nj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a-by-n_semiz
            aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_semiz
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,[n_semiz,special_n_z],d_gridvals,aprime_grid(aprimeindexes),a_grid,z_valblock,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,semizblock,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind+N_d*N_a*semizind; % midpoint is n_d-by-1-by-n_a-by-n_semiz
            curr_offset = 1;
            if has_d1
                Policy(curr_offset,:,:,N_j)=shiftdim(rem(d_ind - 1, N_d1) + 1, -1); %d1
                curr_offset = curr_offset + 1;
            end
            if has_d2
                Policy(curr_offset,:,:,N_j)=shiftdim(ceil(d_ind / N_d1), -1); %d2
            end

            Policy(d_total+1,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)), -1); % midpoint
            Policy(d_total+2,:,:,N_j)=shiftdim(ceil(maxindexL2 / N_d), -1); % aprimeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*semizind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*semizind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            Policy(d_total+3,:,:,N_j)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)), -1);
        end

    elseif vfoptions.lowmemory==2 % joint loop over bothz
        midpoints_Nj=zeros(N_d,1,N_a,'gpuArray');

        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_bothz, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);
            % Treat standard problem as just being the first layer
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_Nj(:,1,level1ii)=maxindex1;

            % Second level based on monotonicity
            maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
            maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
            if has_d1
                maxgap = max(maxgap, [], 1); % Max over d1
            end
            maxgap = squeeze(maxgap);

            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_bothz, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,3);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_Nj(:,1,curraindex)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    midpoints_Nj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                end
            end

            % Turn this into the 'midpoint'
            midpoints_Nj=max(min(midpoints_Nj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-1-by-n_a
            aprimeindexes=(midpoints_Nj+(midpoints_Nj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,special_n_bothz,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            curr_offset = 1;
            if has_d1
                Policy(curr_offset,:,:,N_j)=shiftdim(rem(d_ind - 1, N_d1) + 1, -1); %d1
                curr_offset = curr_offset + 1;
            end
            if has_d2
                Policy(curr_offset,:,:,N_j)=shiftdim(ceil(d_ind / N_d1), -1); %d2
            end

            Policy(d_total+1,:,:,N_j)=shiftdim(squeeze(midpoints_Nj(allind)), -1); % midpoint
            Policy(d_total+2,:,:,N_j)=shiftdim(ceil(maxindexL2 / N_d), -1); % aprimeL2ind

            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d);
            linidx_lower = d_ind                  + N_d*n2long*aind;
            linidx_upper = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);

            Policy(d_total+3,:,:,N_j)=shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)), -1);
        end
    end

else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);    % First, switch V_Jplus1 into Kron form ([N_a,N_bothz]; was erroneously [N_a,N_semiz,N_z])

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            if has_z
                pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order
            else
                pi_bothz = pi_semiz_J(:,:,d2_c,N_j);
            end

            EV_d2inf=(EV==-Inf);
            EV_d2=EV;
            EV_d2(EV_d2inf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV_d2=EV_d2*pi_bothz';
            EV_d2(EV_d2inf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV_d2=reshape(EV_d2,[N_a,1,N_bothz]);

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_d2,-1);
            % Treat standard problem as just being the first layer
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,level1ii,:)=maxindex1;

            % Second level based on monotonicity
            maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
            maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
            if has_d1
                maxgap = max(maxgap, [], 1); % Max over d1
            end
            maxgap = squeeze(maxgap);

            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_z
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                    aprimez=aprimeindexes+N_a*bothzind2;
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[N_d1,(maxgap(ii)+1),1,N_bothz]); % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                end
            end

            % Now for the interpolation layer

            % Interpolate the expectations
            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            % Turn maxindex into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-n_bothz
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*bothzind2; % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_bothz]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % loweredge is n_d-by-1-by-n_a-by-n_bothz
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoints_jj(allind));

            % L2 flag for this d2
            L2offset_d2 = ceil(maxindex/N_d1);
            linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
            linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
            isInfLower = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
            inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
            PolicyL2flag_ford2_jj(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

        end

    elseif vfoptions.lowmemory==1 % parallel over semiz, loop over z
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            if has_z
                pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order
            else
                pi_bothz = pi_semiz_J(:,:,d2_c,N_j);
            end

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                % Calc the condl expectation term (except beta): loop z, vectorize over semiz
                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, [n_semiz,special_n_z], d12c_gridvals, a_grid, a_grid(level1ii), z_valblock, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);
                % Treat standard problem as just being the first layer
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,level1ii,:)=maxindex1;

                % Second level based on monotonicity
                maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
                maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
                if has_d1
                    maxgap = max(maxgap, [], 1); % Max over d1
                end
                maxgap = squeeze(maxgap);

                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d1-by-1-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d1-by-maxgap(ii)+1-by-1-by-n_semiz
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, [n_semiz,special_n_z], d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, ReturnFnParamsVec,3);
                        aprimez=aprimeindexes+N_a*semizind2;
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2z(aprimez),[N_d1,(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 3rd-dim
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                    end
                end

                % Now for the interpolation layer

                % Interpolate the expectations
                EVinterp_d2z=interp1(a_grid,EV_d2z,aprime_grid);

                % Turn maxindex into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d1-by-1-by-n_a-by-n_semiz
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d1-by-n2long-by-n_a-by-n_semiz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, [n_semiz,special_n_z], d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, ReturnFnParamsVec,2);
                aprimez=aprimeindexes+n2aprime*semizind2; % the current aprime
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2z(aprimez),[N_d1*n2long,N_a,N_semiz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,semizblock,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,semizblock,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % loweredge is n_d1-by-1-by-n_a-by-n_semiz
                midpoint_ford2_jj(:,semizblock,d2_c)=squeeze(midpoints_jj(allind));

                % L2 flag for this d2
                L2offset_d2 = ceil(maxindex/N_d1);
                linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                isInfLower = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                PolicyL2flag_ford2_jj(:,semizblock,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end

    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            if has_z
                pi_bothz=kron(pi_z_J(:,:,N_j), pi_semiz_J(:,:,d2_c,N_j)); % reverse order
            else
                pi_bothz = pi_semiz_J(:,:,d2_c,N_j);
            end

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                % Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2); % [N_a, 1]

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_z,-1);
                % Treat standard problem as just being the first layer
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,level1ii)=maxindex1;

                % Second level based on monotonicity
                maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
                maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
                if has_d1
                    maxgap = max(maxgap, [], 1); % Max over d1
                end
                maxgap = squeeze(maxgap);

                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d1-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d1-by-maxgap(ii)+1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,3);
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_z(aprimeindexes),[N_d1,(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 3rd-dim
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                    end
                end

                % Now for the interpolation layer

                % Interpolate the expectations
                EVinterp_z=interp1(a_grid,EV_z,aprime_grid);

                % Turn maxindex into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d1-by-1-by-n_a
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d1-by-n2long-by-n_a
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_z(aprimeindexes),[N_d1*n2long,N_a]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind; % loweredge is n_d1-by-1-by-n_a
                midpoint_ford2_jj(:,z_c,d2_c)=squeeze(midpoints_jj(allind));

                % L2 flag for this d2
                L2offset_d2 = ceil(maxindex/N_d1);
                linidx_lower = d1_ind                  + N_d1*n2long*aind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                isInfLower = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                PolicyL2flag_ford2_jj(:,z_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end
    end

    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj, maxindex] = max(V_ford2_jj, [], 3); % max over d2
    V(:,:,jj) = V_jj;

    maxindex = reshape(maxindex, [N_a * N_semiz * N_z, 1]);
    d1aprimeL2_ind = reshape(Policy_ford2_jj((1:1:N_a*N_bothz)' + (N_a*N_bothz)*(maxindex-1)), [1, N_a, N_bothz]);

    curr_offset = 1;
    if has_d1
        Policy(curr_offset, :, :, jj) = shiftdim(rem(d1aprimeL2_ind - 1, N_d1) + 1, -1); % d1
        curr_offset = curr_offset + 1;
    end
    if has_d2
        Policy(curr_offset, :, :, jj) = reshape(maxindex, [1, N_a, N_semiz * N_z]); %d2
    end

    Policy(d_total+1,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_bothz)' + (N_a*N_bothz)*(maxindex-1)), [1, N_a, N_bothz]); % midpoint
    Policy(d_total+2,:,:,jj)=shiftdim(ceil(d1aprimeL2_ind / N_d1), -1); % aprimeL2ind
    Policy(d_total+3,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_bothz)' + (N_a*N_bothz)*(maxindex-1)), [1, N_a, N_bothz]);
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

    EV=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            if has_z
                pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order
            else
                pi_bothz = pi_semiz_J(:,:,d2_c,jj);
            end

            EV_d2inf=(EV==-Inf);
            EV_d2=EV;
            EV_d2(EV_d2inf)=-1e250; % stop -Inf*0 -> NaN inside the product
            EV_d2=EV_d2*pi_bothz';
            EV_d2(EV_d2inf*(pi_bothz'>0)>0)=-Inf; % exact -Inf restoration
            EV_d2=reshape(EV_d2,[N_a,1,N_bothz]);

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_d2,-1);
            % Treat standard problem as just being the first layer
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' version of maxindex1 [as GI]
            midpoints_jj(:,1,level1ii,:)=maxindex1;

            % Second level based on monotonicity
            maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
            maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
            if has_d1
                maxgap = max(maxgap, [], 1); % Max over d1
            end
            maxgap = squeeze(maxgap);

            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_z
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_z
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                    aprimez=aprimeindexes+N_a*bothzind2;
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2(aprimez),[N_d1,(maxgap(ii)+1),1,N_bothz]); % autoexpand level1iidiff(ii) in 3rd-dim
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                end
            end

            % Now for the interpolation layer

            % Interpolate the expectations
            EVinterp_d2=interp1(a_grid,EV_d2,aprime_grid);

            % Turn maxindex into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a-by-n_bothz
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a-by-n_bothz
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*bothzind2; % the current aprime
            entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2(aprimez),[N_d1*n2long,N_a,N_bothz]);
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            V_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*bothzind; % loweredge is n_d1-by-1-by-n_a-by-n_bothz
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoints_jj(allind));

            % L2 flag for this d2
            L2offset_d2 = ceil(maxindex/N_d1);
            linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
            linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
            isInfLower = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
            inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
            PolicyL2flag_ford2_jj(:,:,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));

        end

    elseif vfoptions.lowmemory==1 % parallel over semiz, loop over z
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            if has_z
                pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order
            else
                pi_bothz = pi_semiz_J(:,:,d2_c,jj);
            end

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                % Calc the condl expectation term (except beta): loop z, vectorize over semiz
                EV_d2z=EV.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz, N_semiz]
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % [N_a, 1, N_semiz]

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, [n_semiz,special_n_z], d12c_gridvals, a_grid, a_grid(level1ii), z_valblock, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_d2z,-1);
                % Treat standard problem as just being the first layer
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,level1ii,:)=maxindex1;

                % Second level based on monotonicity
                maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
                maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
                if has_d1
                    maxgap = max(maxgap, [], 1); % Max over d1
                end
                maxgap = squeeze(maxgap);

                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d1-by-1-by-n_semiz
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d1-by-maxgap(ii)+1-by-1-by-n_semiz
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, [n_semiz,special_n_z], d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_valblock, ReturnFnParamsVec,3);
                        aprimez=aprimeindexes+N_a*semizind2;
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_d2z(aprimez),[N_d1,(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 3rd-dim
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                    end
                end

                % Now for the interpolation layer

                % Interpolate the expectations
                EVinterp_d2z=interp1(a_grid,EV_d2z,aprime_grid);

                % Turn maxindex into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d1-by-1-by-n_a-by-n_semiz
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d1-by-n2long-by-n_a-by-n_semiz
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, [n_semiz,special_n_z], d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_valblock, ReturnFnParamsVec,2);
                aprimez=aprimeindexes+n2aprime*semizind2; % the current aprime
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_d2z(aprimez),[N_d1*n2long,N_a,N_semiz]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,semizblock,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,semizblock,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind+N_d1*N_a*semizind; % loweredge is n_d1-by-1-by-n_a-by-n_semiz
                midpoint_ford2_jj(:,semizblock,d2_c)=squeeze(midpoints_jj(allind));

                % L2 flag for this d2
                L2offset_d2 = ceil(maxindex/N_d1);
                linidx_lower = d1_ind                  + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                isInfLower = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                PolicyL2flag_ford2_jj(:,semizblock,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end

    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            if has_z
                pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d2_c,jj)); % reverse order
            else
                pi_bothz = pi_semiz_J(:,:,d2_c,jj);
            end

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                % Calc the condl expectation term (except beta), which depends on z but not on control variables
                EV_z=EV.*shiftdim(pi_bothz(z_c,:)',-1);
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2); % [N_a, 1]

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*shiftdim(EV_z,-1);
                % Treat standard problem as just being the first layer
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Just keep the 'midpoint' version of maxindex1 [as GI]
                midpoints_jj(:,1,level1ii)=maxindex1;

                % Second level based on monotonicity
                maxgap=maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :);
                maxgap=max(maxgap, [], 4); % Max over z/bothz (add dim 5 for _e_raw)
                if has_d1
                    maxgap = max(maxgap, [], 1); % Max over d1
                end
                maxgap = squeeze(maxgap);

                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d1-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d1-by-maxgap(ii)+1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,3);
                        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EV_z(aprimeindexes),[N_d1,(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 3rd-dim
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii);
                        midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1,1); % unfortunately doesn't autofill
                    end
                end

                % Now for the interpolation layer

                % Interpolate the expectations
                EVinterp_z=interp1(a_grid,EV_z,aprime_grid);

                % Turn maxindex into the 'midpoint'
                midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
                % midpoint is n_d1-by-1-by-n_a
                aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
                % aprime possibilities are n_d1-by-n2long-by-n_a
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_bothz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_d2ii+DiscountFactorParamsVec*reshape(EVinterp_z(aprimeindexes),[N_d1*n2long,N_a]);
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                V_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind; % loweredge is n_d1-by-1-by-n_a
                midpoint_ford2_jj(:,z_c,d2_c)=squeeze(midpoints_jj(allind));

                % L2 flag for this d2
                L2offset_d2 = ceil(maxindex/N_d1);
                linidx_lower = d1_ind                  + N_d1*n2long*aind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                isInfLower = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset_d2 >= 2)         & (L2offset_d2 <= n2short+1);
                inUpperStrict = (L2offset_d2 >= n2short+3) & (L2offset_d2 <= n2long-1);
                PolicyL2flag_ford2_jj(:,z_c,d2_c) = squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper));
            end
        end
    end

    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj, maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,jj)=V_jj;

    maxindex=reshape(maxindex, [N_a * N_semiz * N_z, 1]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_bothz)' + (N_a*N_bothz)*(maxindex-1)), [1, N_a, N_bothz]);

    curr_offset=1;
    if has_d1
        Policy(curr_offset,:,:,jj)=shiftdim(rem(d1aprimeL2_ind - 1, N_d1) + 1, -1); % d1
        curr_offset=curr_offset+1;
    end
    if has_d2
        Policy(curr_offset,:,:,jj)=reshape(maxindex, [1, N_a, N_semiz * N_z]); %d2
    end

    Policy(d_total+1,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_bothz)' + (N_a*N_bothz)*(maxindex-1)), [1, N_a, N_bothz]); % midpoint
    Policy(d_total+2,:,:,jj)=shiftdim(ceil(d1aprimeL2_ind / N_d1), -1); % aprimeL2ind
    Policy(d_total+3,:,:,jj)=reshape(PolicyL2flag_ford2_jj((1:1:N_a*N_bothz)' + (N_a*N_bothz)*(maxindex-1)), [1, N_a, N_bothz]);
end



%% Currently Policy(3,:) is the midpoint, and Policy(4,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(3,:) to 'lower grid point' and then have Policy(4,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(d_total + 2, :, :, :) < 1 + n2short + 1); % if second layer is choosing below midpoint
Policy(d_total+1,:,:,:)=Policy(d_total+1,:,:,:)-adjust; % lower grid point
Policy(d_total+2,:,:,:)=Policy(d_total+2,:,:,:)-(n2short+1)*(~adjust); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

% Policy=squeeze(Policy(1,:,:,:)+N_d1*(Policy(2,:,:,:)-1)+N_d*(Policy(3,:,:,:)-1)+N_d*N_a*(Policy(4,:,:,:)-1)+N_d*N_a*(n2short+2)*(Policy(5,:,:,:)-1));


end
