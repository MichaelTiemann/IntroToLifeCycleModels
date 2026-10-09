function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d1=max(1, prod(n_d1(n_d1 > 0)));
N_d2=max(1, prod(n_d2(n_d2 > 0)));
N_d3=max(1, prod(n_d3(n_d3 > 0)));
N_a1=max(1, prod(n_a1(n_a1 > 0)));
N_a2=max(1, prod(n_a2(n_a2 > 0)));
N_z =max(1, prod(n_z(n_z > 0)));
N_e =max(1, prod(n_e(n_e > 0)));
N_u =max(1, prod(n_u(n_u > 0)));

N_d=N_d1*N_d2*N_d3;
N_a=N_a1*N_a2;

% For ReturnFn
n_d13=[n_d1, n_d3];
% For aprimeFn
n_d23=[n_d2, n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j, 'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j, 'gpuArray'); % d1, d2, d3, a1prime

%%
u_grid=gpuArray(u_grid);

n_d13a1=[n_d1, n_d3, n_a1];
grid_d13a1=[d1_grid; d3_grid; a1_grid];
d13a1_gridvals=CreateGridvals(n_d13a1(n_d13a1 > 0), grid_d13a1,1);

n_a12=[n_a1, n_a2];
grid_a12=[a1_grid; a2_grid];
a12_gridvals=CreateGridvals(n_a12(n_a12 > 0), grid_a12,1);

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_z-1), -1);
eind=shiftdim(gpuArray(0:1:N_e-1), -2);

pi_e_J=shiftdim(pi_e_J, -2); % Move to third dimension

%% Iterate backwards through j
for jj=N_j:-1:1
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', jj,N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);

    is_terminal=(jj == N_j) && ~isfield(vfoptions, 'V_Jplus1');

    if ~is_terminal
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);

        if jj == N_j
            % Using V_Jplus1
            EV_base=reshape(vfoptions.V_Jplus1, [N_a,N_z,N_e]);
            % Apply e transition
            EV_base=sum(EV_base .* pi_e_J(1,1,:,N_j+1), 3);
        else
            EV_base=V(:,:,:, jj+1);
            EV_base=sum(EV_base .* pi_e_J(1,1,:, jj+1), 3);
        end

        [a2primeIndex, a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec, 2);
        % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

        aprimeIndex=repelem((1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
        aprimeplus1Index=repelem((1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]

        % Base probability matrix (d23*a1, u)
        baseProbs=repmat(a2primeProbs,N_a1,1);
    end

    % Setup Evaluation Blocks based on lowmemory
    if vfoptions.lowmemory == 0
        e_iter=1; special_n_e=n_e;
        z_iter=1; special_n_z=n_z;
    elseif vfoptions.lowmemory == 1
        e_iter=1:N_e; special_n_e=ones(1, length(n_e));
        z_iter=1;     special_n_z=n_z;
    else % lowmemory == 2
        e_iter=1:N_e; special_n_e=ones(1, length(n_e));
        z_iter=1:N_z; special_n_z=ones(1, length(n_z));
    end

    for e_c=e_iter
        if vfoptions.lowmemory == 0
            e_val=e_gridvals_J(:,:, jj);
            e_idx=1:N_e; e_offset=eind;
        else
            e_val=e_gridvals_J(e_c,:, jj);
            e_idx=e_c; e_offset=0;
        end

        for z_c=z_iter
            if vfoptions.lowmemory <= 1
                z_val=z_gridvals_J(:,:, jj);
                z_idx=1:N_z; z_offset=zind;
            else
                z_val=z_gridvals_J(z_c,:, jj);
                z_idx=z_c; z_offset=0;
            end

            % (d,aprime,a,z,e)
            ReturnMatrix_block=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13, n_a1], [n_a1, n_a2], special_n_z, special_n_e, d13a1_gridvals, a12_gridvals, z_val, e_val, ReturnFnParamsVec);

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            [ReturnMatrix_onlyd3, d1index]=max(reshape(ReturnMatrix_block, [N_d1,N_d3*N_a1,N_a, length(z_idx), length(e_idx)]), [],1);

            if is_terminal
                % Terminal Period Logic (No Continuation Value)
                entireRHS=shiftdim(ReturnMatrix_onlyd3,1);

                % Calc the max and it's index
                [Vtemp, maxindex]=max(entireRHS, [],1);

                V(:, z_idx, e_idx, jj)=shiftdim(Vtemp,1);
                Policy(3,:, z_idx, e_idx, jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
                Policy(4,:, z_idx, e_idx, jj)=shiftdim(ceil(maxindex/N_d3), -1);
                Policy(1,:, z_idx, e_idx, jj)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*z_offset+N_d3*N_a1*N_a*N_z*e_offset),1);
                Policy(2,:, z_idx, e_idx, jj)=1; % is meaningless anyway
            else
                % B. Compute Continuation Value (EV)
                if vfoptions.lowmemory <= 1
                    EV_z=EV_base .* shiftdim(pi_z_J(:,:, jj)', -1);
                else
                    EV_z=EV_base .* pi_z_J(z_c,:, jj);
                end

                EV_z(isnan(EV_z))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z, 2); % sum over z', leaving a singular second dimension

                % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
                % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
                skipinterp=logical(EV_z(aprimeIndex(:)+N_a*((1:length(z_idx))-1)) == EV_z(aprimeplus1Index(:)+N_a*((1:length(z_idx))-1)));

                % Expand probabilities locally for this block ONLY
                blockProbs=repmat(baseProbs,1, length(z_idx));
                blockProbs(skipinterp)=0;
                blockProbs=reshape(blockProbs, [N_d23*N_a1,N_u, length(z_idx)]);

                % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
                EV1=reshape(EV_z(aprimeIndex(:)+N_a*((1:length(z_idx))-1)), [N_d23*N_a1,N_u, length(z_idx)]) .* blockProbs; % probability of lower grid point
                EV2=reshape(EV_z(aprimeplus1Index(:)+N_a*((1:length(z_idx))-1)), [N_d23*N_a1,N_u, length(z_idx)]) .* (1 - blockProbs); % probability of upper grid point

                EV1(isnan(EV1))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so the term contributes nothing
                EV2(isnan(EV2))=0;

                % Expectation over u (using pi_u), and then add the lower and upper
                EV_block=sum((EV1 .* pi_u'), 2)+sum((EV2 .* pi_u'), 2); % (d&a1prime,u,z), sum over u
                % EV is over (d&a1prime,1,z)

                % Second (out of order): EV, we can refine out d2
                [EV_onlyd3, d2index]=max(reshape(DiscountFactorParamsVec*EV_block, [N_d2,N_d3*N_a1,1, length(z_idx)]), [],1);

                % Now put together entireRHS, which just depends on d3
                entireRHS=shiftdim(ReturnMatrix_onlyd3+EV_onlyd3,1);

                % Calc the max and it's index
                [Vtemp, maxindex]=max(entireRHS, [],1);

                V(:, z_idx, e_idx, jj)=shiftdim(Vtemp,1);
                Policy(3,:, z_idx, e_idx, jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
                Policy(4,:, z_idx, e_idx, jj)=shiftdim(ceil(maxindex/N_d3), -1);
                Policy(1,:, z_idx, e_idx, jj)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*z_offset+N_d3*N_a1*N_a*N_z*e_offset),1);
                Policy(2,:, z_idx, e_idx, jj)=shiftdim(d2index(maxindex+N_d3*z_offset),1);
            end
        end
    end
end


end
