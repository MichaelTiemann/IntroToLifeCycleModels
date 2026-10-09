function [V,Policy]=ValueFnIter_FHorz_DC2A_e_raw(n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% divide-and-conquer in the first endo state
% lowmemory: =0 vectorize, =1 loop over e, =2 loop over e and z

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
level1ii=round(linspace(1,n_a(1),vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

pi_e_J=shiftdim(pi_e_J,-2); % Move to third dimension

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
a2Bind=shiftdim(gpuArray(0:1:N_a2-1),-1); % already includes -1
if vfoptions.lowmemory==0
    zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
    eind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
    zBind=shiftdim(gpuArray(0:1:N_z-1),-4); % already includes -1
elseif vfoptions.lowmemory==1
    zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
    zBind=shiftdim(gpuArray(0:1:N_z-1),-4); % already includes -1
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
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
            loweredge = min(maxindex1(:,1,:,ii,:,:,:), N_a1-maxgap(ii)); 
            
            % 1. Package the handle and shape
            ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
            reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_z, N_e];
            
            % 2. Call your new helper!
            [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size);
            
            % 3. Assign results
            V(curraindex,:,:,N_j) = shiftdim(Vtempii,1);
            allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d*N_a2*N_a2*zind + N_d*N_a2*N_a2*N_z*eind; 
            Policy(curraindex,:,:,N_j) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_vals=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_vals, ReturnFnParamsVec,1,0);

            % First, we want a1prime conditional on (d,1,a2prime,a,z)
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
                loweredge = min(maxindex1(:,1,:,ii,:,:), N_a1-maxgap(ii));

                % 1. Package the handle and shape
                ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_vals, ReturnFnParamsVec, 2, 0);
                reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_z, 1];

                % 2. Call your new helper!
                [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size);

                % 3. Assign results
                V(curraindex,:,:,N_j) = shiftdim(Vtempii,1);
                allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d*N_a2*N_a2*zind;
                Policy(curraindex,:,:,N_j) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_vals=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_vals=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, e_vals, ReturnFnParamsVec,1,0);

                % First, we want a1prime conditional on (d,1,a2prime,a)
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
                    loweredge = min(maxindex1(:,1,:,ii,:), N_a1-maxgap(ii));

                    % 1. Package the handle and shape
                    ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, e_vals, ReturnFnParamsVec, 2, 0);
                    reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, 1, 1];

                    % 2. Call your new helper!
                    [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size);

                    % 3. Assign results
                    V(curraindex,:,:,N_j) = shiftdim(Vtempii,1);
                    allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii));
                    Policy(curraindex,:,:,N_j) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
                end
            end
        end
    end

else

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*pi_e_J(1,1,:,N_j+1),3); % Using V_Jplus1

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension


    if vfoptions.lowmemory==0
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[1,N_a1,N_a2,1,1,N_z]); % autoexpand d into 1st-dim
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autofill e

        % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
            loweredge = min(maxindex1(:,1,:,ii,:,:,:), N_a1-maxgap(ii)); 
            
            % 1. Package the handle and shape
            ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2, 0);
            reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_z, N_e];
            
            % 2. Extract the relevant Expected Value subset
            aprimez = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind + N_a*zBind;
            EV_RHS_slice = DiscountedEV(reshape(aprimez, reshape_size));
            
            % 3. Call your new helper!
            [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size, EV_RHS_slice);
            
            % 4. Assign results
            V(curraindex,:,:,N_j) = shiftdim(Vtempii,1);
            allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d*N_a2*N_a2*zind + N_d*N_a2*N_a2*N_z*eind; 
            Policy(curraindex,:,:,N_j) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
        end

    elseif vfoptions.lowmemory==1
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[1,N_a1,N_a2,1,1,N_z]); % autoexpand d into 1st-dim
        for e_c=1:N_e
            e_vals=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), e_vals, ReturnFnParamsVec,1,0);

            entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV; % autofill e

            % First, we want a1prime conditional on (d,1,a2prime,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
                loweredge = min(maxindex1(:,1,:,ii,:,:), N_a1-maxgap(ii));

                % 1. Package the handle and shape
                ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), e_vals, ReturnFnParamsVec, 2, 0);
                reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_z, 1];

                % 2. Extract the relevant Expected Value subset
                aprimez = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind + N_a*zBind;
                EV_RHS_slice = DiscountedEV(reshape(aprimez, reshape_size));

                % 3. Call your new helper!
                [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size, EV_RHS_slice);

                % 4. Assign results
                V(curraindex,:,:,N_j) = shiftdim(Vtempii,1);
                allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d*N_a2*N_a2*zind;
                Policy(curraindex,:,:,N_j) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_vals=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z = DiscountFactorParamsVec * reshape(EV(:,:,z_c), [1, N_a1, N_a2, 1, 1]);
            for e_c=1:N_e
                e_vals=e_gridvals_J(e_c,:,N_j);
                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, e_vals, ReturnFnParamsVec,1,0);

                entireRHS_ii=ReturnMatrix_ii_ze+DiscountedEV_z; % autofill e

                % First, we want a1prime conditional on (d,1,a2prime,a)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
                    loweredge = min(maxindex1(:,1,:,ii,:), N_a1-maxgap(ii));

                    % 1. Package the handle and shape
                    ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, e_vals, ReturnFnParamsVec, 2, 0);
                    reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, 1, 1];

                    % 2. Extract the relevant Expected Value subset
                    aprime = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind;
                    EV_RHS_slice = DiscountedEV_z(reshape(aprime, reshape_size));

                    % 3. Call your new helper!
                    [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size, EV_RHS_slice);

                    % 4. Assign results
                    V(curraindex,:,:,N_j) = shiftdim(Vtempii,1);
                    allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii));
                    Policy(curraindex,:,:,N_j) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
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

    EV=sum(V(:,:,:,jj+1).*pi_e_J(1,1,:,jj+1),3);

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    if vfoptions.lowmemory==0
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[1,N_a1,N_a2,1,1,N_z]); % autoexpand d into 1st-dim
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % autofill e

        % First, we want a1prime conditional on (d,1,a2prime,a,z,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,:,:,jj)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
            loweredge = min(maxindex1(:,1,:,ii,:,:,:), N_a1-maxgap(ii)); 
            
            % 1. Package the handle and shape
            ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 2, 0);
            reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_z, N_e];
            
            % 2. Extract the relevant Expected Value subset
            aprimez = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind + N_a*zBind;
            EV_RHS_slice = DiscountedEV(reshape(aprimez, reshape_size));
            
            % 3. Call your new helper!
            [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size, EV_RHS_slice);
            
            % 4. Assign results
            V(curraindex,:,:,jj) = shiftdim(Vtempii,1);
            allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d*N_a2*N_a2*zind + N_d*N_a2*N_a2*N_z*eind; 
            Policy(curraindex,:,:,jj) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
        end

    elseif vfoptions.lowmemory==1
        DiscountedEV=DiscountFactorParamsVec*reshape(EV,[1,N_a1,N_a2,1,1,N_z]); % autoexpand d into 1st-dim
        for e_c=1:N_e
            e_vals=e_gridvals_J(e_c,:,jj);
            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), e_vals, ReturnFnParamsVec,1,0);

            entireRHS_ii=ReturnMatrix_ii_e+DiscountedEV; % autofill e

            % First, we want a1prime conditional on (d,1,a2prime,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,:,e_c,jj)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
                loweredge = min(maxindex1(:,1,:,ii,:,:), N_a1-maxgap(ii));

                % 1. Package the handle and shape
                ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), e_vals, ReturnFnParamsVec, 2, 0);
                reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_z, 1];

                % 2. Extract the relevant Expected Value subset
                aprimez = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind + N_a*zBind;
                EV_RHS_slice = DiscountedEV(reshape(aprimez, reshape_size));

                % 3. Call your new helper!
                [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size, EV_RHS_slice);

                % 4. Assign results
                V(curraindex,:,:,jj) = shiftdim(Vtempii,1);
                allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d*N_a2*N_a2*zind;
                Policy(curraindex,:,:,jj) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_vals=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z = DiscountFactorParamsVec * reshape(EV(:,:,z_c), [1, N_a1, N_a2, 1, 1]);
            for e_c=1:N_e
                e_vals=e_gridvals_J(e_c,:,jj);
                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, e_vals, ReturnFnParamsVec,1,0);

                entireRHS_ii=ReturnMatrix_ii_ze+DiscountedEV_z; % autofill e

                % First, we want a1prime conditional on (d,1,a2prime,a)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1) + N_a1*repelem(a2ind',level1iidiff(ii),1);
                    loweredge = min(maxindex1(:,1,:,ii,:), N_a1-maxgap(ii));

                    % 1. Package the handle and shape
                    ReturnFnHandle = @(a1p) CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, special_n_z, special_n_e, d_gridvals, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, e_vals, ReturnFnParamsVec, 2, 0);
                    reshape_size = [N_d*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, 1, 1];

                    % 2. Extract the relevant Expected Value subset
                    aprime = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind;
                    EV_RHS_slice = DiscountedEV_z(reshape(aprime, reshape_size));

                    % 3. Call your new helper!
                    [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d, N_a1, reshape_size, EV_RHS_slice);

                    % 4. Assign results
                    V(curraindex,:,:,jj) = shiftdim(Vtempii,1);
                    allind = dind + N_d*a2primeind + N_d*N_a2*repelem(a2ind,1,level1iidiff(ii));
                    Policy(curraindex,:,:,jj) = shiftdim(maxindexfix + N_d*(loweredge(allind)-1), 1);
                end
            end
        end
    end

end

%%
Policy=shiftdim(Policy,-1);

end
