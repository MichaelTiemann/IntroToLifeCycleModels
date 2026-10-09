function [V,Policy2]=ValueFnIter_FHorz_DC2_e_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer in both endo states
% lowmemory: =0 vectorize over z, =1 loop over z

N_d_raw=prod(n_d);
has_d = ~isempty(n_d) && prod(n_d) > 0;
N_d=max(N_d_raw,1);

N_a=prod(n_a);

N_z_raw=prod(n_z);
has_z = ~isempty(n_z) && prod(n_z) > 0;
N_z=max(N_z_raw, 1);

N_e=prod(vfoptions.n_e);

N_a1=n_a(1);
N_a2=n_a(2);

if ~has_z
    z_gridvals_J = zeros(1, 1, N_j); % Pad to safely index (:,:,jj)
    pi_z_J = ones(1, 1, N_j); % Pad with 1s so EV * 1 = EV (no-op)
    n_z = 0; % Tell CreateReturnFnMatrix to omit z
elseif size(z_gridvals_J, 3) < N_j
    z_gridvals_J = repmat(z_gridvals_J, 1, 1, N_j); % Time-invariant fallback
    pi_z_J = repmat(pi_z_J, 1, 1, N_j); % Time-invariant fallback
end

V=zeros(N_a1,N_a2,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a1,N_a2,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if has_d
    d_gridvals=CreateGridvals(n_d,d_grid,1);
else
    n_d=0;
    d_gridvals=[];
end

a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
level11ii=round(linspace(1,n_a(1),vfoptions.level1n(1)));
level12kk=round(linspace(1,n_a(2),vfoptions.level1n(2)));

% precompute
if vfoptions.lowmemory==0
    zind=shiftdim(0:1:N_z-1,-2); % already includes -1
    eind=shiftdim(gpuArray(0:1:N_e-1),-3); % already includes -1
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e),'gpuArray');
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z),'gpuArray');
    special_n_e=ones(1,length(n_e),'gpuArray');
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % (d,a1a2prime,a1,a2,z)

        % First, we want a1a2prime conditional on (d,1,a,z)
        % We would just do
        % [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
        % So instead for now we instead do following two lines
        [~,maxindex1]=max(permute(ReturnMatrix_ii,[2,1,3,4,5]),[],1);
        maxindex1=permute(maxindex1,[2,1,3,4,5]);

        %% Level 2
        % Split maxindex1 into a1prime and a2prime
        maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z,N_e]);
        maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z,N_e]);

        % Attempt for improved version
        maxgap1=squeeze(max(max(max(maxindex11(:,1,2:end,2:end,:)-maxindex11(:,1,1:end-1,1:end-1,:),[],6),[],5),[],1));
        maxgap2=squeeze(max(max(max(maxindex12(:,1,2:end,2:end,:)-maxindex12(:,1,1:end-1,1:end-1,:),[],6),[],5),[],1));
        for ii=1:(vfoptions.level1n(1)-1)
            % Perfectly partition a1: No redundant boundary evaluations!
            curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

            for kk=1:(vfoptions.level1n(2)-1)
                % Perfectly partition a2: No redundant boundary evaluations!
                curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                % Cap the loweredges (Safe regardless of gaps)
                loweredge1 = min(maxindex11(:,1,ii,kk,:,:), N_a1-maxgap1(ii,kk));
                loweredge2 = min(maxindex12(:,1,ii,kk,:,:), N_a2-maxgap2(ii,kk));

                % 1. Package the handle
                ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);

                % 2. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, []);

                % 3. Assign results
                V(curra1index,curra2index,:,:,N_j) = shiftdim(Vtempii,1);
                allind = dind + N_d*zind + N_d*N_z*eind;
                Policy(curra1index,curra2index,:,:,N_j) = shiftdim(maxindexfix + N_d*(loweredge1(allind)-1) + N_d*N_a1*(loweredge2(allind)-1), 1);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            % (d,a1a2prime,a1,a2)

            % First, we want a1a2prime conditional on (d,1,a1,a2)
            % We would just do
            % [~,maxindex1]=max(ReturnMatrix_ii,[],2);
            % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
            % So instead for now we instead do following two lines
            [~,maxindex1]=max(permute(ReturnMatrix_ii,[2,1,3,4]),[],1);
            maxindex1=permute(maxindex1,[2,1,3,4]);

            %% Level 2
            % Split maxindex1 into a1prime and a2prime
            maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);
            maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);

            % Attempt for improved version
            maxgap1 = squeeze(max(max(maxindex11(:,1,2:end,2:end,:)-maxindex11(:,1,1:end-1,1:end-1,:), [], 5), [], 1));
            maxgap2 = squeeze(max(max(maxindex12(:,1,2:end,2:end,:)-maxindex12(:,1,1:end-1,1:end-1,:), [], 5), [], 1));
            for ii=1:(vfoptions.level1n(1)-1)
                % Perfectly partition a1: No redundant boundary evaluations!
                curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

                for kk=1:(vfoptions.level1n(2)-1)
                    % Perfectly partition a2: No redundant boundary evaluations!
                    curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                    % Cap the loweredges (Safe regardless of gaps)
                    loweredge1 = min(maxindex11(:,1,ii,kk,:), N_a1-maxgap1(ii,kk));
                    loweredge2 = min(maxindex12(:,1,ii,kk,:), N_a2-maxgap2(ii,kk));

                    % 1. Package the handle
                    ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);

                    % 2. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                    [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, []);

                    % 3. Assign results
                    V(curra1index,curra2index,:,e_c,N_j) = shiftdim(Vtempii,1);
                    allind = dind + N_d*zind;
                    Policy(curra1index,curra2index,:,e_c,N_j) = shiftdim(maxindexfix + N_d*(loweredge1(allind)-1) + N_d*N_a1*(loweredge2(allind)-1), 1);
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_val, e_val, ReturnFnParamsVec, 1);
                % (d,a1a2prime,a1,a2)

                % First, we want a1a2prime conditional on (d,1,a1,a2)
                % We would just do
                % [~,maxindex1]=max(ReturnMatrix_ii,[],2);
                % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
                % So instead for now we instead do following two lines
                [~,maxindex1]=max(permute(ReturnMatrix_ii,[2,1,3,4]),[],1);
                maxindex1=permute(maxindex1,[2,1,3,4]);

                %% Level 2
                % Split maxindex1 into a1prime and a2prime
                maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);
                maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);

                % Attempt for improved version
                maxgap1=squeeze(max(maxindex11(:,1,2:end,2:end)-maxindex11(:,1,1:end-1,1:end-1),[],1));
                maxgap2=squeeze(max(maxindex12(:,1,2:end,2:end)-maxindex12(:,1,1:end-1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n(1)-1)
                    % Perfectly partition a1: No redundant boundary evaluations!
                    curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

                    for kk=1:(vfoptions.level1n(2)-1)
                        % Perfectly partition a2: No redundant boundary evaluations!
                        curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                        % Cap the loweredges (Safe regardless of gaps)
                        loweredge1 = min(maxindex11(:,1,ii,kk), N_a1-maxgap1(ii,kk));
                        loweredge2 = min(maxindex12(:,1,ii,kk), N_a2-maxgap2(ii,kk));

                        % 1. Package the handle
                        ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_val, e_val, ReturnFnParamsVec, 2);

                        % 2. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                        [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, []);

                        % 3. Assign results
                        V(curra1index,curra2index,z_c,e_c,N_j) = shiftdim(Vtempii,1);
                        Policy(curra1index,curra2index,z_c,e_c,N_j) = shiftdim(maxindexfix + N_d*(loweredge1(dind)-1) + N_d*N_a1*(loweredge2(dind)-1), 1);
                    end
                end
            end
        end
    end

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a1,N_a2,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,3); % sum over z', leaving a singular second dimension
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[1,N_a1*N_a2,1,1,N_z,N_e]); % [1,aprime,1,1,z]; d-dim is singleton, broadcasts at use sites

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        % (d,a1a2prime,a1,a2,z)

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        % First, we want a1a2prime conditional on (d,1,a,z)
        % We would just do
        % [~,maxindex1]=max(entireRHS_ii,[],2);
        % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
        % So instead for now we instead do following two lines
        [~,maxindex1]=max(permute(entireRHS_ii,[2,1,3,4,5]),[],1);
        maxindex1=permute(maxindex1,[2,1,3,4,5]);

        %% Level 2
        % Split maxindex1 into a1prime and a2prime
        maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z,N_e]);
        maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z,N_e]);

        % Attempt for improved version
        maxgap1=squeeze(max(max(max(maxindex11(:,1,2:end,2:end,:,:)-maxindex11(:,1,1:end-1,1:end-1,:,:),[],6),[],5),[],1));
        maxgap2=squeeze(max(max(max(maxindex12(:,1,2:end,2:end,:,:)-maxindex12(:,1,1:end-1,1:end-1,:,:),[],6),[],5),[],1));
        for ii=1:(vfoptions.level1n(1)-1)
            % Perfectly partition a1: No redundant boundary evaluations!
            curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

            for kk=1:(vfoptions.level1n(2)-1)
                % Perfectly partition a2: No redundant boundary evaluations!
                curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                % Cap the loweredges (Safe regardless of gaps)
                loweredge1 = min(maxindex11(:,1,ii,kk,:,:), N_a1-maxgap1(ii,kk));
                loweredge2 = min(maxindex12(:,1,ii,kk,:,:), N_a2-maxgap2(ii,kk));

                % 1. Package the handle
                ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);

                % 2. Extract EV Slice
                a1primeindexes = loweredge1 + repmat((0:1:maxgap1(ii,kk)), 1, maxgap2(ii,kk)+1);
                a2primeindexes = loweredge2 + repelem((0:1:maxgap2(ii,kk)), 1, maxgap1(ii,kk)+1);
                aprimeze = a1primeindexes + N_a1*(a2primeindexes-1) + N_a*shiftdim((0:1:N_z-1),-3) + N_a*N_z*shiftdim((0:1:N_e-1),-4);

                reshape_size = [N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1), 1, 1, N_z, N_e];
                EV_RHS_slice = reshape(DiscountedEV(aprimeze), reshape_size);

                % 3. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, EV_RHS_slice);

                % 4. Assign results
                V(curra1index,curra2index,:,:,N_j) = shiftdim(Vtempii,1);
                allind = dind + N_d*zind + N_d*N_z*eind;
                Policy(curra1index,curra2index,:,:,N_j) = shiftdim(maxindexfix + N_d*(loweredge1(allind)-1) + N_d*N_a1*(loweredge2(allind)-1), 1);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            DiscountedEV_e = DiscountedEV(:,:,1,1,:,e_c); % Isolate the current e slice

            e_val = e_gridvals_J(:,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
            % (d,a1a2prime,a1,a2)

            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_e;

            % First, we want a1a2prime conditional on (d,1,a)
            % We would just do
            % [~,maxindex1]=max(entireRHS_ii,[],2);
            % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
            % So instead for now we instead do following two lines
            [~,maxindex1]=max(permute(entireRHS_ii,[2,1,3,4,5]),[],1);
            maxindex1=permute(maxindex1,[2,1,3,4,5]);

            %% Level 2
            % Split maxindex1 into a1prime and a2prime
            maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);
            maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);

            % Attempt for improved version
            maxgap1 = squeeze(max(max(maxindex11(:,1,2:end,2:end,:)-maxindex11(:,1,1:end-1,1:end-1,:), [], 5), [], 1));
            maxgap2 = squeeze(max(max(maxindex12(:,1,2:end,2:end,:)-maxindex12(:,1,1:end-1,1:end-1,:), [], 5), [], 1));
            for ii=1:(vfoptions.level1n(1)-1)
                % Perfectly partition a1: No redundant boundary evaluations!
                curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

                for kk=1:(vfoptions.level1n(2)-1)
                    % Perfectly partition a2: No redundant boundary evaluations!
                    curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                    % Cap the loweredges (Safe regardless of gaps)
                    loweredge1 = min(maxindex11(:,1,ii,kk), N_a1-maxgap1(ii,kk));
                    loweredge2 = min(maxindex12(:,1,ii,kk), N_a2-maxgap2(ii,kk));

                    % 1. Package the handle
                    ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);

                    % 2. Extract EV Slice
                    a1primeindexes = loweredge1 + repmat((0:1:maxgap1(ii,kk)), 1, maxgap2(ii,kk)+1);
                    a2primeindexes = loweredge2 + repelem((0:1:maxgap2(ii,kk)), 1, maxgap1(ii,kk)+1);
                    aprimez = a1primeindexes + N_a1*(a2primeindexes-1) + N_a*shiftdim((0:1:N_z-1),-3);

                    reshape_size = [N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1), 1, 1, N_z];
                    EV_RHS_slice = reshape(DiscountedEV_e(aprimez), reshape_size);

                    % 3. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                    [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, EV_RHS_slice);

                    % 4. Assign results
                    V(curra1index,curra2index,:,e_c,N_j) = shiftdim(Vtempii,1);
                    allind = dind + N_d*zind;
                    Policy(curra1index,curra2index,:,e_c,N_j) = shiftdim(maxindexfix + N_d*(loweredge1(allind)-1) + N_d*N_a1*(loweredge2(allind)-1), 1);
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val = z_gridvals_J(:,:,N_j);
            for e_c=1:N_e
                DiscountedEV_ze = DiscountedEV(:,:,1,1,z_c,e_c);

                e_val = e_gridvals_J(:,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_val, e_val, ReturnFnParamsVec, 1);
                % (d,a1a2prime,a1,a2)

                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_ze;

                % First, we want a1a2prime conditional on (d,1,a)
                % We would just do
                % [~,maxindex1]=max(entireRHS_ii,[],2);
                % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
                % So instead for now we instead do following two lines
                [~,maxindex1]=max(permute(entireRHS_ii,[2,1,3,4]),[],1);
                maxindex1=permute(maxindex1,[2,1,3,4]);

                %% Level 2
                % Split maxindex1 into a1prime and a2prime
                maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);
                maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);

                % Attempt for improved version
                maxgap1=squeeze(max(maxindex11(:,1,2:end,2:end)-maxindex11(:,1,1:end-1,1:end-1),[],1));
                maxgap2=squeeze(max(maxindex12(:,1,2:end,2:end)-maxindex12(:,1,1:end-1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n(1)-1)
                    % Perfectly partition a1: No redundant boundary evaluations!
                    curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

                    for kk=1:(vfoptions.level1n(2)-1)
                        % Perfectly partition a2: No redundant boundary evaluations!
                        curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                        % Cap the loweredges (Safe regardless of gaps)
                        loweredge1 = min(maxindex11(:,1,ii,kk), N_a1-maxgap1(ii,kk));
                        loweredge2 = min(maxindex12(:,1,ii,kk), N_a2-maxgap2(ii,kk));

                        % 1. Package the handle
                        ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_val, e_val, ReturnFnParamsVec, 2);

                        % 2. Extract EV Slice
                        a1primeindexes = loweredge1 + repmat((0:1:maxgap1(ii,kk)), 1, maxgap2(ii,kk)+1);
                        a2primeindexes = loweredge2 + repelem((0:1:maxgap2(ii,kk)), 1, maxgap1(ii,kk)+1);
                        aprime = a1primeindexes + N_a1*(a2primeindexes-1);

                        reshape_size = [N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1), 1, 1];
                        EV_RHS_slice = reshape(DiscountedEV_ze(aprime), reshape_size);

                        % 3. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                        [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, EV_RHS_slice);

                        % 4. Assign results
                        V(curra1index,curra2index,z_c,e_c,N_j) = shiftdim(Vtempii,1);
                        Policy(curra1index,curra2index,z_c,e_c,N_j) = shiftdim(maxindexfix + N_d*(loweredge1(dind)-1) + N_d*N_a1*(loweredge2(dind)-1), 1);
                    end
                end
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

    EV=V(:,:,:,jj+1).*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,3); % sum over z', leaving a singular second dimension
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[1,N_a1*N_a2,1,1,N_z,N_e]); % [1,aprime,1,1,z]; d-dim is singleton, broadcasts at use sites

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        % (d,a1a2prime,a1,a2,z)

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        % First, we want a1a2prime conditional on (d,1,a,z)
        % We would just do
        % [~,maxindex1]=max(entireRHS_ii,[],2);
        % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
        % So instead for now we instead do following two lines
        [~,maxindex1]=max(permute(entireRHS_ii,[2,1,3,4,5]),[],1);
        maxindex1=permute(maxindex1,[2,1,3,4,5]);

        %% Level 2
        % Split maxindex1 into a1prime and a2prime
        maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z,N_e]);
        maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z,N_e]);

        % Attempt for improved version
        maxgap1=squeeze(max(max(max(maxindex11(:,1,2:end,2:end,:,:)-maxindex11(:,1,1:end-1,1:end-1,:,:),[],6),[],5),[],1));
        maxgap2=squeeze(max(max(max(maxindex12(:,1,2:end,2:end,:,:)-maxindex12(:,1,1:end-1,1:end-1,:,:),[],6),[],5),[],1));
        for ii=1:(vfoptions.level1n(1)-1)
            % Perfectly partition a1: No redundant boundary evaluations!
            curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

            for kk=1:(vfoptions.level1n(2)-1)
                % Perfectly partition a2: No redundant boundary evaluations!
                curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                % Cap the loweredges (Safe regardless of gaps)
                loweredge1 = min(maxindex11(:,1,ii,kk,:,:), N_a1-maxgap1(ii,kk));
                loweredge2 = min(maxindex12(:,1,ii,kk,:,:), N_a2-maxgap2(ii,kk));

                % 1. Package the handle
                ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);

                % 2. Extract EV Slice
                a1primeindexes = loweredge1 + repmat((0:1:maxgap1(ii,kk)), 1, maxgap2(ii,kk)+1);
                a2primeindexes = loweredge2 + repelem((0:1:maxgap2(ii,kk)), 1, maxgap1(ii,kk)+1);
                aprimeze = a1primeindexes + N_a1*(a2primeindexes-1) + N_a*shiftdim((0:1:N_z-1),-3) + N_a*N_z*shiftdim((0:1:N_e-1),-4);

                reshape_size = [N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1), 1, 1, N_z, N_e];
                EV_RHS_slice = reshape(DiscountedEV(aprimeze), reshape_size);

                % 3. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, EV_RHS_slice);

                % 4. Assign results
                V(curra1index,curra2index,:,:,jj) = shiftdim(Vtempii,1);
                allind = dind + N_d*zind + N_d*N_z*eind;
                Policy(curra1index,curra2index,:,:,jj) = shiftdim(maxindexfix + N_d*(loweredge1(allind)-1) + N_d*N_a1*(loweredge2(allind)-1), 1);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            DiscountedEV_e = DiscountedEV(:,:,1,1,:,e_c); % Isolate the current e slice
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_gridvals_J(z_c,:,jj), e_val, ReturnFnParamsVec, 1);
            % (d,a1a2prime,a1,a2)

            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_e;

            % First, we want a1a2prime conditional on (d,1,a)
            % We would just do
            % [~,maxindex1]=max(entireRHS_ii,[],2);
            % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
            % So instead for now we instead do following two lines
            [~,maxindex1]=max(permute(entireRHS_ii,[2,1,3,4,5]),[],1);
            maxindex1=permute(maxindex1,[2,1,3,4,5]);

            %% Level 2
            % Split maxindex1 into a1prime and a2prime
            maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);
            maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2),N_z]);

            % Attempt for improved version
            maxgap1=squeeze(max(maxindex11(:,1,2:end,2:end)-maxindex11(:,1,1:end-1,1:end-1),[],1));
            maxgap2=squeeze(max(maxindex12(:,1,2:end,2:end)-maxindex12(:,1,1:end-1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n(1)-1)
                % Perfectly partition a1: No redundant boundary evaluations!
                curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

                for kk=1:(vfoptions.level1n(2)-1)
                    % Perfectly partition a2: No redundant boundary evaluations!
                    curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                    % Cap the loweredges (Safe regardless of gaps)
                    loweredge1 = min(maxindex11(:,1,ii,kk,:), N_a1-maxgap1(ii,kk));
                    loweredge2 = min(maxindex12(:,1,ii,kk,:), N_a2-maxgap2(ii,kk));

                    % 1. Package the handle
                    ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_gridvals_J(z_c,:,jj), e_val, ReturnFnParamsVec, 2);

                    % 2. Extract EV Slice
                    a1primeindexes = loweredge1 + repmat((0:1:maxgap1(ii,kk)), 1, maxgap2(ii,kk)+1);
                    a2primeindexes = loweredge2 + repelem((0:1:maxgap2(ii,kk)), 1, maxgap1(ii,kk)+1);
                    aprimez = a1primeindexes + N_a1*(a2primeindexes-1) + N_a*shiftdim((0:1:N_z-1),-3);

                    reshape_size = [N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1), 1, 1, N_z];
                    EV_RHS_slice = reshape(DiscountedEV_e(aprimez), reshape_size);

                    % 3. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                    [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, EV_RHS_slice);

                    % 4. Assign results
                    V(curra1index,curra2index,:,e_c,jj) = shiftdim(Vtempii,1);
                    allind = dind + N_d*zind;
                    Policy(curra1index,curra2index,:,e_c,jj) = shiftdim(maxindexfix + N_d*(loweredge1(allind)-1) + N_d*N_a1*(loweredge2(allind)-1), 1);
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,1,1,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                % n-Monotonicity
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level11ii), a2_grid(level12kk), z_val, e_val, ReturnFnParamsVec, 1);
                % (d,a1a2prime,a1,a2)

                entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

                % First, we want a1a2prime conditional on (d,1,a)
                % We would just do
                % [~,maxindex1]=max(entireRHS_ii,[],2);
                % But there is an error in Matlab for max in second dimension on GPU: https://au.mathworks.com/matlabcentral/answers/2152160-error-in-index-returned-by-max-in-the-second-dimension-in-obscure-case
                % So instead for now we instead do following two lines
                [~,maxindex1]=max(permute(entireRHS_ii,[2,1,3,4]),[],1);
                maxindex1=permute(maxindex1,[2,1,3,4]);

                %% Level 2
                % Split maxindex1 into a1prime and a2prime
                maxindex11=reshape(rem(maxindex1-1,N_a1)+1,[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);
                maxindex12=reshape(ceil(maxindex1/N_a1),[N_d,1,vfoptions.level1n(1),vfoptions.level1n(2)]);

                % Attempt for improved version
                maxgap1=squeeze(max(maxindex11(:,1,2:end,2:end)-maxindex11(:,1,1:end-1,1:end-1),[],1));
                maxgap2=squeeze(max(maxindex12(:,1,2:end,2:end)-maxindex12(:,1,1:end-1,1:end-1),[],1));
                for ii=1:(vfoptions.level1n(1)-1)
                    % Perfectly partition a1: No redundant boundary evaluations!
                    curra1index = (level11ii(ii) + (ii > 1)) : level11ii(ii+1);

                    for kk=1:(vfoptions.level1n(2)-1)
                        % Perfectly partition a2: No redundant boundary evaluations!
                        curra2index = (level12kk(kk) + (kk > 1)) : level12kk(kk+1);

                        % Cap the loweredges (Safe regardless of gaps)
                        loweredge1 = min(maxindex11(:,1,ii,kk), N_a1-maxgap1(ii,kk));
                        loweredge2 = min(maxindex12(:,1,ii,kk), N_a2-maxgap2(ii,kk));

                        % 1. Package the handle
                        ReturnFnHandle = @(a1p, a2p) CreateReturnFnMatrix_Disc_DC2_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid(a2p), a1_grid(curra1index), a2_grid(curra2index), z_val, e_val, ReturnFnParamsVec, 2);

                        % 2. Extract EV Slice
                        DiscountedEV_ze = DiscountedEV(:,:,1,1,z_c,e_c);
                        a1primeindexes = loweredge1 + repmat((0:1:maxgap1(ii,kk)), 1, maxgap2(ii,kk)+1);
                        a2primeindexes = loweredge2 + repelem((0:1:maxgap2(ii,kk)), 1, maxgap1(ii,kk)+1);
                        aprime = a1primeindexes + N_a1*(a2primeindexes-1);

                        reshape_size = [N_d*(maxgap1(ii,kk)+1)*(maxgap2(ii,kk)+1), 1, 1];
                        EV_RHS_slice = reshape(DiscountedEV(aprime), reshape_size);

                        % 3. Call the helper! (Naturally collapses to 1D or 0D if gaps are 0)
                        [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1(ii,kk), maxgap2(ii,kk), N_d, N_a1, EV_RHS_slice);

                        % 4. Assign results
                        V(curra1index,curra2index,z_c,e_c,jj) = shiftdim(Vtempii,1);
                        Policy(curra1index,curra2index,z_c,e_c,jj) = shiftdim(maxindexfix + N_d*(loweredge1(dind)-1) + N_d*N_a1*(loweredge2(dind)-1), 1);
                    end
                end
            end
        end

    end

    % Can skip V reshape as code works without this, but needs to be done for Policy (or more precisely for Policy2)
    Policy=reshape(Policy,[N_a,N_z,N_j]);

    %%
    Policy2=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
    Policy2(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
    Policy2(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);


end
