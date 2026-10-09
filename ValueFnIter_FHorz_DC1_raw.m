function [V,Policy]=ValueFnIter_FHorz_DC1_raw(n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

has_d = ~isempty(n_d) && prod(n_d) > 0;
has_z = ~isempty(n_z) && prod(n_z) > 0;

N_d = max(1, prod(n_d));
N_a = prod(n_a);
N_z = max(1, prod(n_z));

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
else
    zind=shiftdim((0:1:N_z-1),-1); % already includes -1
end

zBind=shiftdim(gpuArray(0:1:N_z-1),-2);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);

        % Store
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Second level based on monotonicity
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex = level1ii(ii)+1:1:level1ii(ii+1)-1;
            loweredge = min(maxindex1(:,1,ii,:), n_a-maxgap(ii));
            
            % 1. Package the handle and shape
            ReturnFnHandle = @(ap) CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, reshape(a_grid(ap), size(ap)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            reshape_size = [N_d*(maxgap(ii)+1), 1, N_z];

            % 2. Call the helper!
            [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size);
            
            % 3. Assign results
            V(curraindex,:,N_j) = shiftdim(Vtempii,1);
            allind = dind + N_d*zind; 
            Policy(curraindex,:,N_j) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex = level1ii(ii)+1:1:level1ii(ii+1)-1;
                loweredge = min(maxindex1(:,1,ii,:), n_a-maxgap(ii));

                % 1. Package the handle and shape
                ReturnFnHandle = @(ap) CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, reshape(a_grid(ap), size(ap)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec, 2);
                reshape_size = [N_d*(maxgap(ii)+1), 1, 1];

                % 2. Call the helper!
                [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size);

                % 3. Assign results
                V(curraindex,z_c,N_j) = shiftdim(Vtempii,1);
                Policy(curraindex,z_c,N_j) = shiftdim(maxindex + N_d*(loweredge(dind)-1), 1);
            end
        end
    end
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    EVinf=(EV==-Inf);
    EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
    EV=EV*pi_z_J(:,:,N_j)';
    EV(EVinf*(pi_z_J(:,:,N_j)'>0)>0)=-Inf; % exact -Inf restoration
    EV=reshape(EV,[N_a,1,N_z]);

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1);

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);

        % Store
        V(level1ii,:,N_j)=shiftdim(Vtempii,1);
        Policy(level1ii,:,N_j)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex = level1ii(ii)+1:1:level1ii(ii+1)-1;
            loweredge = min(maxindex1(:,1,ii,:), n_a-maxgap(ii));
            
            % 1. Package the handle and shape
            ReturnFnHandle = @(ap) CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, reshape(a_grid(ap), size(ap)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            reshape_size = [N_d*(maxgap(ii)+1), 1, N_z];
            
            % 2. Extract the Expected Value subset (and multiply by DiscountFactor)
            aprimez = loweredge + (0:1:maxgap(ii)) + N_a*zBind;
            EV_RHS_slice = DiscountFactorParamsVec * reshape(EV(aprimez), reshape_size);
            
            % 3. Call the helper!
            [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);
            
            % 4. Assign results
            V(curraindex,:,N_j) = shiftdim(Vtempii,1);
            allind = dind + N_d*zind; 
            Policy(curraindex,:,N_j) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            EV_z=EV(:,:,z_c);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV_z,-1);

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,N_j)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1),[],1);
            for ii=1:(vfoptions.level1n-1)
                curraindex = level1ii(ii)+1:1:level1ii(ii+1)-1;
                loweredge = min(maxindex1(:,1,ii,:), n_a-maxgap(ii));

                % 1. Package the handle and shape
                ReturnFnHandle = @(ap) CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, reshape(a_grid(ap), size(ap)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec, 2);
                reshape_size = [N_d*(maxgap(ii)+1), 1, 1];

                % 2. Extract the Expected Value subset (and multiply by DiscountFactor)
                aprimez = loweredge + (0:1:maxgap(ii));
                EV_RHS_slice = DiscountFactorParamsVec * reshape(EV_z(aprimez), reshape_size);

                % 3. Call the helper!
                [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);

                % 4. Assign results
                V(curraindex,z_c,N_j) = shiftdim(Vtempii,1);
                Policy(curraindex,z_c,N_j) = shiftdim(maxindex + N_d*(loweredge(dind)-1), 1);
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

    EV=V(:,:,jj+1);

    EVinf=(EV==-Inf);
    EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
    EV=EV*pi_z_J(:,:,jj)';
    EV(EVinf*(pi_z_J(:,:,jj)'>0)>0)=-Inf; % exact -Inf restoration
    EV=reshape(EV,[N_a,1,N_z]);

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV,-1);

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n,N_z]),[],1);

        % Store
        V(level1ii,:,jj)=shiftdim(Vtempii,1);
        Policy(level1ii,:,jj)=shiftdim(maxindex2,1); % d,aprime

        % Attempt for improved version
        maxgap=max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1);
        for ii=1:(vfoptions.level1n-1)
            curraindex = level1ii(ii)+1:1:level1ii(ii+1)-1;
            loweredge = min(maxindex1(:,1,ii,:), n_a-maxgap(ii));
            
            % 1. Package the handle and shape
            ReturnFnHandle = @(ap) CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, reshape(a_grid(ap), size(ap)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec, 2);
            reshape_size = [N_d*(maxgap(ii)+1), 1, N_z];
            
            % 2. Extract the Expected Value subset (and multiply by DiscountFactor)
            aprimez = loweredge + (0:1:maxgap(ii)) + N_a*zBind;
            EV_RHS_slice = DiscountFactorParamsVec * reshape(EV(aprimez), reshape_size);
            
            % 3. Call the helper!
            [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);
            
            % 4. Assign results
            V(curraindex,:,jj) = shiftdim(Vtempii,1);
            allind = dind + N_d*zind; 
            Policy(curraindex,:,jj) = shiftdim(maxindex + N_d*(loweredge(allind)-1), 1);
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*shiftdim(EV_z,-1);

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);

            % Store
            V(level1ii,z_c,jj)=shiftdim(Vtempii,1);
            Policy(level1ii,z_c,jj)=shiftdim(maxindex2,1); % d,aprime

            % Attempt for improved version
            maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
            for ii=1:(vfoptions.level1n-1)
                curraindex = level1ii(ii)+1:1:level1ii(ii+1)-1;
                loweredge = min(maxindex1(:,1,ii,:), n_a-maxgap(ii));

                % 1. Package the handle and shape
                ReturnFnHandle = @(ap) CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, special_n_z, d_gridvals, reshape(a_grid(ap), size(ap)), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec, 2);
                reshape_size = [N_d*(maxgap(ii)+1), 1, N_z];

                % 2. Extract the Expected Value subset (and multiply by DiscountFactor)
                aprimez = loweredge + (0:1:maxgap(ii));
                EV_RHS_slice = DiscountFactorParamsVec * reshape(EV_z(aprimez), reshape_size);

                % 3. Call the helper!
                [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, loweredge, maxgap(ii), N_d, reshape_size, EV_RHS_slice);

                % 4. Assign results
                V(curraindex,z_c,jj) = shiftdim(Vtempii,1);
                Policy(curraindex,z_c,jj) = shiftdim(maxindex + N_d*(loweredge(dind)-1), 1);
            end

        end
    end
end

%%
Policy=shiftdim(Policy,-1);

sz_V = N_a;
if has_z; sz_V = [sz_V, N_z]; end
sz_V = [sz_V, N_j];

V = reshape(V, sz_V);
Policy = reshape(Policy, [1, sz_V]); % DC1 Policy is just a combined index block


end
