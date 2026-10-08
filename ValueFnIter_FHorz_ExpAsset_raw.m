function [V,Policy]=ValueFnIter_FHorz_ExpAsset_raw(n_d1,n_d2,n_a1,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1_raw=prod(n_d1);
N_d2_raw=prod(n_d2);
has_d1=(N_d1_raw > 0);
N_d1=max(N_d1_raw, 1);
N_d2=max(N_d2_raw, 1);

if ~has_d1
    d_gridvals = d2_gridvals;
    n_d1 = 0; % ensures CreateReturnFnMatrix handles it correctly as having no d1
end

N_a1_raw = prod(n_a1);
% has_a1 = (N_a1 > 0);
N_a1 = max(N_a1_raw, 1);
N_a2 = prod(n_a2);
N_a = N_a1 * N_a2;

N_z_raw=prod(n_z);
N_z = max(N_z_raw, 1);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); % indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory > 0
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0
        %Calc the max and its index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            %Calc the max and its index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    if vfoptions.experienceassetz
        [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2,n_z,z_gridvals_J(:,:,N_j)); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    else
        [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    end
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]); % First, switch V_Jplus1 into Kron form
    EV=InterpolateExpAssetEV(EVpre, n_a2, N_d2, N_a1, N_a2, N_z, a2primeIndex, a2primeProbs);

    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    if vfoptions.lowmemory==0
        DiscountedEV=DiscountFactorParamsVec*reshape(EV, [1, N_d2*N_a1, 1, N_a2, N_z]);
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2,n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0

        % Split ReturnMatrix dimensions so implicit expansion can broadcast d1 and a1 cleanly
        ReturnMatrix_reshaped = reshape(ReturnMatrix, [N_d1, N_d2*N_a1, N_a1, N_a2, N_z]);
        entireRHS = reshape(ReturnMatrix_reshaped + DiscountedEV, [N_d1*N_d2*N_a1, N_a1*N_a2, N_z]);

        %Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z = DiscountFactorParamsVec*reshape(EV(:,:,z_c), [1, N_d2*N_a1, 1, N_a2]);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

            ReturnMatrix_reshaped = reshape(ReturnMatrix_z, [N_d1, N_d2*N_a1, N_a1, N_a2]);
            entireRHS_z = reshape(ReturnMatrix_reshaped + DiscountedEV_z, [N_d1*N_d2*N_a1, N_a1*N_a2]);

            %Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
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
    if vfoptions.experienceassetz
        [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2,n_z,z_gridvals_J(:,:,jj)); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    else
        [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    end
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    EVpre=V(:,:,jj+1); % Extract  the 2D slice for this age
    EV=InterpolateExpAssetEV(EVpre, n_a2, N_d2, N_a1, N_a2, N_z, a2primeIndex, a2primeProbs);

    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    if vfoptions.lowmemory==0
        DiscountedEV=DiscountFactorParamsVec*reshape(EV, [1, N_d2*N_a1, 1, N_a2, N_z]);
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2,n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0); % Level=0, Refine=0
        
        % Split ReturnMatrix dimensions so implicit expansion can broadcast d1 and a1 cleanly
        ReturnMatrix_reshaped = reshape(ReturnMatrix, [N_d1, N_d2*N_a1, N_a1, N_a2, N_z]);
        entireRHS = reshape(ReturnMatrix_reshaped + DiscountedEV, [N_d1*N_d2*N_a1, N_a1*N_a2, N_z]);
        
        %Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
        
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z = DiscountFactorParamsVec*reshape(EV(:,:,z_c), [1, N_d2*N_a1, 1, N_a2]);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            
            ReturnMatrix_reshaped = reshape(ReturnMatrix_z, [N_d1, N_d2*N_a1, N_a1, N_a2]);
            entireRHS_z = reshape(ReturnMatrix_reshaped + DiscountedEV_z, [N_d1*N_d2*N_a1, N_a1*N_a2]);
            
            %Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
    end
end

%%
Policy=shiftdim(Policy,-1);


end
