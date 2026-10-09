function [V,Policy]=ValueFnIter_FHorz_ExpAsset_DC1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1_raw=prod(n_d1);
N_d2_raw=prod(n_d2);
has_d1=(N_d1_raw > 0);
N_d1=max(N_d1_raw, 1);
N_d2=max(N_d2_raw, 1);
N_d=N_d1 * N_d2;

if ~has_d1
    d_gridvals = d2_gridvals;
    n_d1 = 0; % ensures CreateReturnFnMatrix handles it correctly as having no d1
end

N_a1_raw=prod(n_a1);
N_a2_raw=prod(n_a2);
N_a1=max(N_a1_raw,1);
N_a2=max(N_a2_raw,1);
N_a=N_a1*N_a2;

N_z_raw=prod(n_z);
N_z=max(N_z_raw,1);

N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==0
    % precompute
    eBind=shiftdim((0:1:N_e-1),-2); % already includes -1
    eind = shiftdim((0:1:N_e-1), -4);
    % precompute
    zind=shiftdim((0:1:N_z-1),-3); % already includes -1
    zBind=shiftdim((0:1:N_z-1),-1); % already includes -1
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    % precompute
    zind=shiftdim((0:1:N_z-1),-3); % already includes -1
    zBind=shiftdim((0:1:N_z-1),-1); % already includes -1
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2ind=shiftdim((0:1:N_a2-1),-2);
a2Bind=gpuArray(0:1:N_a2-1);
d2ind=repelem((1:1:N_d2)',N_d1,1); % d2 component of each d=(d1,d2), [N_d,1]

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        ReturnMatrix_ii = reshape(ReturnMatrix_ii, [N_d, N_a1, vfoptions.level1n, N_a2, N_z, N_e]);

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            % Naturally cap the loweredge
            loweredge = min(maxindex1(:,1,ii,:,:,:), N_a1-maxgap(ii));

            % 1. Package the handle (Notice Level = 2)
            ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);

            % 2. Call the helper!
            [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, [], []);

            % 3. Assign results
            V(curraindex,:,:,N_j) = shiftdim(Vtempii,1);
            allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii)) + N_d*N_a2*zBind + N_d*N_a2*N_z*eBind;
            Policy(curraindex,:,:,N_j) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            ReturnMatrix_ii_e = reshape(ReturnMatrix_ii_e, [N_d, N_a1, vfoptions.level1n, N_a2, N_z, 1]);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z,1]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                % Naturally cap the loweredge
                loweredge = min(maxindex1(:,1,ii,:,:,:), N_a1-maxgap(ii));

                % 1. Package the handle (Notice Level = 2)
                ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);

                % 2. Call the helper!
                [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, [], []);

                % 3. Assign results
                V(curraindex,:,e_c,N_j) = shiftdim(Vtempii,1);
                allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii)) + N_d*N_a2*zBind;
                Policy(curraindex,:,e_c,N_j) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
                ReturnMatrix_ii_ze = reshape(ReturnMatrix_ii_ze, [N_d, N_a1, vfoptions.level1n, N_a2, 1, 1]);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    % Naturally cap the loweredge
                    loweredge = min(maxindex1(:,1,ii,:,:,:), N_a1-maxgap(ii));

                    % 1. Package the handle (Notice Level = 2)
                    ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec, 2, 0);

                    % 2. Call the helper!
                    [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, [], []);

                    % 3. Assign results
                    V(curraindex,z_c,e_c,N_j) = shiftdim(Vtempii,1);
                    allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii));
                    Policy(curraindex,z_c,e_c,N_j) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
                end
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

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        ReturnMatrix_ii=reshape(ReturnMatrix_ii, [N_d1, N_d2, N_a1, vfoptions.level1n, N_a2, N_z, N_e]);
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z,N_e]);

        entireRHS_ii = ReturnMatrix_ii + shiftdim(DiscountedEV, -1);
        entireRHS_ii = reshape(entireRHS_ii, [N_d, N_a1, vfoptions.level1n, N_a2, N_z, N_e]);

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            % Naturally cap the loweredge
            loweredge = min(maxindex1(:,1,ii,:,:,:), N_a1-maxgap(ii));
            a1primeindexes = loweredge + (0:1:maxgap(ii)); % Needed to slice EV below

            % 1. Package the handle (Notice Level = 3)
            ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3, 0);

            % 2. Extract EV and define reshape
            d2aprimez = d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*a2ind + N_d2*N_a*zind + N_d2*N_a*N_z*eind;
            EV_RHS_slice = DiscountedEV(d2aprimez);
            reshape_size = [N_d*(maxgap(ii)+1), level1iidiff(ii)*N_a2, N_z, N_e];

            % 3. Call the helper!
            [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);

            % 4. Assign results
            V(curraindex,:,:,jj) = shiftdim(Vtempii,1);
            allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii)) + N_d*N_a2*zBind + N_d*N_a2*N_z*eBind;
            Policy(curraindex,:,:,jj) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
        end

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            ReturnMatrix_ii=reshape(ReturnMatrix_ii, [N_d1, N_d2, N_a1, vfoptions.level1n, N_a2, N_z, 1]);
            DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z,1]);

            entireRHS_ii = ReturnMatrix_ii + shiftdim(DiscountedEV, -1);
            entireRHS_ii = reshape(entireRHS_ii, [N_d, N_a1, vfoptions.level1n, N_a2, N_z, 1]);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                % Naturally cap the loweredge
                loweredge = min(maxindex1(:,1,ii,:,:), N_a1-maxgap(ii));
                a1primeindexes = loweredge + (0:1:maxgap(ii)); % Needed to slice EV below

                % 1. Package the handle (Notice Level = 3)
                ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3, 0);

                % 2. Extract EV and define reshape
                d2aprimez = d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*a2ind + N_d2*N_a*zind;
                EV_RHS_slice = DiscountedEV(d2aprimez);
                reshape_size = [N_d*(maxgap(ii)+1), level1iidiff(ii)*N_a2, N_z, 1];

                % 3. Call the helper!
                [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);

                % 4. Assign results
                V(curraindex,:,e_c,jj) = shiftdim(Vtempii,1);
                allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii)) + N_d*N_a2*zBind;
                Policy(curraindex,:,e_c,jj) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountFactorParamsVec*reshape(EV(:,:,z_c),[N_d2,N_a1,1,N_a2,1,1]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
                ReturnMatrix_ii_ze=reshape(ReturnMatrix_ii_ze, [N_d1, N_d2, N_a1, vfoptions.level1n, N_a2, 1, 1]);
                entireRHS_ii_ze = ReturnMatrix_ii_ze + shiftdim(DiscountedEV_z, -1);
                entireRHS_ii_ze = reshape(entireRHS_ii_ze, [N_d, N_a1, vfoptions.level1n, N_a2, 1, 1]);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    % Naturally cap the loweredge
                    loweredge = min(maxindex1(:,1,ii,:), N_a1-maxgap(ii));
                    a1primeindexes = loweredge + (0:1:maxgap(ii)); % Needed to slice EV below

                    % 1. Package the handle (Notice Level = 3)
                    ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec, 3, 0);

                    % 2. Extract EV and define reshape
                    d2aprimez = d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*a2ind + N_d2*N_a*zind;
                    EV_RHS_slice = DiscountedEV(d2aprimez);
                    reshape_size = [N_d*(maxgap(ii)+1), level1iidiff(ii)*N_a2, 1, 1];

                    % 3. Call the helper!
                    [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);

                    % 4. Assign results
                    V(curraindex,z_c,e_c,jj) = shiftdim(Vtempii,1);
                    allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii));
                    Policy(curraindex,z_c,e_c,jj) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
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

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0
        ReturnMatrix_ii=reshape(ReturnMatrix_ii, [N_d1, N_d2, N_a1, vfoptions.level1n, N_a2, N_z, N_e]);

        entireRHS_ii=ReturnMatrix_ii+shiftdim(DiscountedEV,-1); % autofill 3rd dim to N_a1
        entireRHS_ii = reshape(entireRHS_ii, [N_d, N_a1, vfoptions.level1n, N_a2, N_z, N_e]);

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,:,:,jj)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            % Naturally cap the loweredge
            loweredge = min(maxindex1(:,1,ii,:,:,:), N_a1-maxgap(ii));
            a1primeindexes = loweredge + (0:1:maxgap(ii)); % Needed to slice EV below

            % 1. Package the handle (Notice Level = 3)
            ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3, 0);

            % 2. Extract EV and define reshape
            d2aprimez = d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*a2ind + N_d2*N_a*zind + N_d2*N_a*N_z*eind;
            EV_RHS_slice = DiscountedEV(d2aprimez);
            reshape_size = [N_d*(maxgap(ii)+1), level1iidiff(ii)*N_a2, N_z, N_e];

            % 3. Call the helper!
            [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);

            % 4. Assign results
            V(curraindex,:,:,jj) = shiftdim(Vtempii,1);
            allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii)) + N_d*N_a2*zBind + N_d*N_a2*N_z*eBind;
            Policy(curraindex,:,:,jj) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
        end

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
            ReturnMatrix_ii_e=reshape(ReturnMatrix_ii_e, [N_d1, N_d2, N_a1, vfoptions.level1n, N_a2, N_z, 1]);

            entireRHS_ii_e=ReturnMatrix_ii_e+shiftdim(DiscountedEV,-1); % autofill 3rd dim to N_a1
            entireRHS_ii_e = reshape(entireRHS_ii_e, [N_d, N_a1, vfoptions.level1n, N_a2, N_z, 1]);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,:,e_c,jj)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                % Naturally cap the loweredge
                loweredge = min(maxindex1(:,1,ii,:,:), N_a1-maxgap(ii));
                a1primeindexes = loweredge + (0:1:maxgap(ii)); % Needed to slice EV below

                % 1. Package the handle (Notice Level = 3)
                ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3, 0);

                % 2. Extract EV and define reshape
                d2aprimez = d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*a2ind + N_d2*N_a*zind;
                EV_RHS_slice = DiscountedEV(d2aprimez);
                reshape_size = [N_d*(maxgap(ii)+1), level1iidiff(ii)*N_a2, N_z, 1];

                % 3. Call the helper!
                [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);

                % 4. Assign results
                V(curraindex,:,e_c,jj) = shiftdim(Vtempii,1);
                allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii)) + N_d*N_a2*zBind;
                Policy(curraindex,:,e_c,jj) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountFactorParamsVec*reshape(EV(:,:,z_c),[N_d2,N_a1,1,N_a2,1,1]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0
                ReturnMatrix_ii_ze=reshape(ReturnMatrix_ii_ze, [N_d1, N_d2, N_a1, vfoptions.level1n, N_a2, 1, 1]);
                entireRHS_ii_ze = ReturnMatrix_ii_ze + shiftdim(DiscountedEV_z, -1);
                entireRHS_ii_ze = reshape(entireRHS_ii_ze, [N_d, N_a1, vfoptions.level1n, N_a2, 1, 1]);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    % Naturally cap the loweredge
                    loweredge = min(maxindex1(:,1,ii,:), N_a1-maxgap(ii));
                    a1primeindexes = loweredge + (0:1:maxgap(ii)); % Needed to slice EV below

                    % 1. Package the handle (Notice Level = 3)
                    ReturnFnHandle = @(a1p) CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, reshape(a1_gridvals(a1p), size(a1p)), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec, 3, 0);

                    % 2. Extract EV and define reshape
                    d2aprimez = d2ind + N_d2*(a1primeindexes-1) + N_d2*N_a1*a2ind;
                    EV_RHS_slice = DiscountedEV(d2aprimez);
                    reshape_size = [N_d*(maxgap(ii)+1), level1iidiff(ii)*N_a2, 1, 1];

                    % 3. Call the helper!
                    [Vtempii, maxindex, dind] = RefineSearch_ExpAsset_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);

                    % 4. Assign results
                    V(curraindex,z_c,e_c,jj) = shiftdim(Vtempii,1);
                    allind = dind + N_d*repelem(a2Bind,1,level1iidiff(ii));
                    Policy(curraindex,z_c,e_c,jj) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
                end
            end
        end
    end
end

%%
Policy=shiftdim(Policy,-1);


end
