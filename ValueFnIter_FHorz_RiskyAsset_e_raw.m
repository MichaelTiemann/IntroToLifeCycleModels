function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

% Safely calculate N_d dimensions, treating 0 as a singleton (1) for math
N_d1 = max(1, prod(n_d1(n_d1 > 0)));
N_d2 = max(1, prod(n_d2(n_d2 > 0)));
N_d3 = max(1, prod(n_d3(n_d3 > 0)));
N_a1 = max(1, prod(n_a1(n_a1 > 0)));
N_a2 = max(1, prod(n_a2(n_a2 > 0)));
N_z  = max(1, prod(n_z(n_z > 0)));
N_e  = max(1, prod(n_e(n_e > 0)));
N_u  = max(1, prod(n_u(n_u > 0)));
N_d  = N_d1 * N_d2 * N_d3;
N_a  = N_a1 * N_a2;

% For ReturnFn
n_d13 = [n_d1(n_d1 > 0), n_d3(n_d3 > 0)];

% For aprimeFn
n_d23 = [n_d2(n_d2 > 0), n_d3(n_d3 > 0)];
N_d23 = N_d2 * N_d3;
d23_grid = [d2_grid; d3_grid];

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

%% Unified Time Loop
for jj = N_j:-1:1
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', jj, N_j)
    end

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    is_terminal = (jj == N_j) && ~isfield(vfoptions, 'V_Jplus1');

    %% Compute EV (Only if not terminal)
    if ~is_terminal
        DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));
        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);

        % 1. Get next period's V and integrate out 'e' immediately!
        if jj == N_j
            V_next = reshape(vfoptions.V_Jplus1, [N_a, N_z, N_e]);
        else
            V_next = V(:,:,:,jj+1);
        end

        % Safely get the shock distribution (cap at max J to avoid out-of-bounds)
        pi_e_step = pi_e_J(:, min(jj+1, size(pi_e_J, 2)));
        
        % shiftdim aligns pi_e to dim 3 for exact tensor broadcasting
        EV_base = sum(V_next .* shiftdim(pi_e_step, -2), 3); % Now it is just [N_a, N_z]!

        [a2primeIndex, a2primeProbs] = CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec, 2);
        aprimeIndex = repelem((1:N_a1)', N_d23, N_u) + N_a1 * repmat(a2primeIndex-1, N_a1, 1);
        aprimeplus1Index = repelem((1:N_a1)', N_d23, N_u) + N_a1 * repmat(a2primeIndex, N_a1, 1);
        baseProbs = repmat(a2primeProbs, N_a1, 1);
    end

    %% Setup Evaluation Loops
    if vfoptions.lowmemory == 0
        e_iter = 1; special_n_e = n_e;
        z_iter = 1; special_n_z = n_z;
    elseif vfoptions.lowmemory == 1
        e_iter = 1:N_e; special_n_e = ones(1, length(n_e));
        z_iter = 1;     special_n_z = n_z;
    else
        e_iter = 1:N_e; special_n_e = ones(1, length(n_e));
        z_iter = 1:N_z; special_n_z = ones(1, length(n_z));
    end

    for e_c = e_iter
        if vfoptions.lowmemory == 0
            e_val = e_gridvals_J(:, :, jj);
            e_idx = 1:N_e; e_offset = eind;
        else
            e_val = e_gridvals_J(e_c, :, jj);
            e_idx = e_c; e_offset = 0;
        end

        for z_c = z_iter
            if vfoptions.lowmemory <= 1
                z_val = z_gridvals_J(:, :, jj);
                z_idx = 1:N_z; z_offset = zind;
            else
                z_val = z_gridvals_J(z_c, :, jj);
                z_idx = z_c; z_offset = 0;
            end

            % 1. Evaluate Return Matrix
            ReturnMatrix_block = CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13, n_a1], [n_a1, n_a2], special_n_z, special_n_e, d13a1_gridvals, a12_gridvals, z_val, e_val, ReturnFnParamsVec);

            % 2. Refine out d1
            [ReturnMatrix_onlyd3, d1index] = max(reshape(ReturnMatrix_block, [N_d1, N_d3*N_a1, N_a, length(z_idx), length(e_idx)]), [], 1);

            if is_terminal
                entireRHS = shiftdim(ReturnMatrix_onlyd3, 1);
                [Vtemp, maxindex] = max(entireRHS, [], 1);
                V(:, z_idx, e_idx, jj) = shiftdim(Vtemp, 1);
                Policy(3, :, z_idx, e_idx, jj) = shiftdim(rem(maxindex-1, N_d3)+1, 1);
                Policy(4, :, z_idx, e_idx, jj) = shiftdim(ceil(maxindex/N_d3), -1);
                Policy(1, :, z_idx, e_idx, jj) = shiftdim(d1index(maxindex + N_d3*N_a1*aind + N_d3*N_a1*N_a*z_offset + N_d3*N_a1*N_a*N_z*e_offset), 1);
                Policy(2, :, z_idx, e_idx, jj) = 1;
            else
                % 3. JUST-IN-TIME Expectation over z
                if vfoptions.lowmemory <= 1
                    EV_z = EV_base .* shiftdim(pi_z_J(:, :, jj)', -1);
                else
                    EV_z = EV_base .* pi_z_J(z_c, :, jj);
                end
                EV_z(isnan(EV_z)) = 0;
                EV_z = sum(EV_z, 2);

                skipinterp = logical(EV_z(aprimeIndex(:) + N_a*((1:length(z_idx))-1)) == EV_z(aprimeplus1Index(:) + N_a*((1:length(z_idx))-1)));
                blockProbs = repmat(baseProbs, 1, length(z_idx));
                blockProbs(skipinterp) = 0;
                blockProbs = reshape(blockProbs, [N_d23*N_a1, N_u, length(z_idx)]);

                EV1 = reshape(EV_z(aprimeIndex(:) + N_a*((1:length(z_idx))-1)), [N_d23*N_a1, N_u, length(z_idx)]) .* blockProbs;
                EV2 = reshape(EV_z(aprimeplus1Index(:) + N_a*((1:length(z_idx))-1)), [N_d23*N_a1, N_u, length(z_idx)]) .* (1 - blockProbs);
                EV1(isnan(EV1)) = 0; EV2(isnan(EV2)) = 0;

                EV_block = sum((EV1 .* pi_u'), 2) + sum((EV2 .* pi_u'), 2);

                % 4. Refine out d2
                [EV_onlyd3, d2index] = max(reshape(DiscountFactorParamsVec * EV_block, [N_d2, N_d3*N_a1, 1, length(z_idx)]), [], 1);

                entireRHS = shiftdim(ReturnMatrix_onlyd3 + EV_onlyd3, 1);
                [Vtemp, maxindex] = max(entireRHS, [], 1);
                V(:, z_idx, e_idx, jj) = shiftdim(Vtemp, 1);
                Policy(3, :, z_idx, e_idx, jj) = shiftdim(rem(maxindex-1, N_d3)+1, 1);
                Policy(4, :, z_idx, e_idx, jj) = shiftdim(ceil(maxindex/N_d3), -1);
                Policy(1, :, z_idx, e_idx, jj) = shiftdim(d1index(maxindex + N_d3*N_a1*aind + N_d3*N_a1*N_a*z_offset + N_d3*N_a1*N_a*N_z*e_offset), 1);
                Policy(2, :, z_idx, e_idx, jj) = shiftdim(d2index(maxindex + N_d3*z_offset), 1);
            end
        end
    end
end

%% Shrink-wrap Policy to remove inactive choice dimensions
has_d1 = (sum(n_d1) > 0);
has_d2 = (sum(n_d2) > 0);
has_d3 = (sum(n_d3) > 0);
has_a1 = (sum(n_a1) > 0);

active_rows = [];
if has_d1, active_rows(end+1) = 1; end
if has_d2, active_rows(end+1) = 2; end
if has_d3, active_rows(end+1) = 3; end
if has_a1, active_rows(end+1) = 4; end

slice_idx = repmat({':'}, 1, ndims(Policy));
slice_idx{1} = active_rows;
Policy = Policy(slice_idx{:});

end
