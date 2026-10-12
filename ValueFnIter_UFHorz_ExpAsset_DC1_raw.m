function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;

has_semiz = Dims.N_semiz_raw > 0;
has_e     = Dims.N_e_raw > 0;
has_z     = Dims.special_n_z(1) > 0;
has_bothz = has_semiz || has_z;

n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_e = Dims.n_e;

N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
n_d1 = Dims.n_d1; n_d2 = Dims.n_d2;
n_d3 = Dims.n_d3; N_d3 = Dims.N_d3;
N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;

% --- Divide-and-Conquer Initialization ---
if N_a1 == 0; error('Cannot use vfoptions.divideandconquer with experience asset if there is no standard endogenous state'); end
if ~isfield(vfoptions, 'level1n'); vfoptions.level1n = min(floor(sqrt(n_a1(1))), n_a1(1)); end
level1ii = round(linspace(1, n_a1(1), vfoptions.level1n));

% Grid Extracts
a1_gridvals = CreateGridvals(n_a1, vfoptions.a1_grid, 1);
a2_gridvals = CreateGridvals(n_a2, vfoptions.a2_grid, 1);

% --- 2. Determine Iteration Strategy ---
if vfoptions.lowmemory == 0
    IterStrategy = 'Full';
elseif vfoptions.lowmemory == 1
    if has_e; IterStrategy = 'Loop_e';
    elseif has_semiz && has_z; IterStrategy = 'Loop_z_parallel_semiz';
    else; IterStrategy = 'Loop_bothz'; end
elseif vfoptions.lowmemory == 2
    if has_e && has_semiz && has_z; IterStrategy = 'Loop_z_parallel_semiz_e';
    elseif has_e; IterStrategy = 'Loop_bothz_e';
    elseif has_semiz && has_z; IterStrategy = 'Loop_bothz';
    else; error('Invalid lowmemory=2 for this shock combination.'); end
elseif vfoptions.lowmemory == 3
    if has_e && has_semiz && has_z; IterStrategy = 'Loop_bothz_e';
    else; error('Invalid lowmemory=3 for this shock combination.'); end
end

% --- 3. Introspect ExpAsset Parameters ---
if Dims.exp_flag > 0
    l_d2_len = length(n_d2); if prod(n_d2) == 0; l_d2_len = 0; end
    l_a2_len = length(n_a2); if prod(n_a2) == 0; l_a2_len = 0; end
    l_z_len  = length(n_z);  if prod(n_z) == 0;  l_z_len = 0; end
    l_e_len  = length(n_e);  if prod(n_e) == 0;  l_e_len = 0; end
    l_bothz_len = length(Dims.n_bothz); if prod(Dims.n_bothz) == 0; l_bothz_len = 0; end
    if isfield(vfoptions, 'n_u'); l_u_len = length(vfoptions.n_u); if prod(vfoptions.n_u) == 0; l_u_len = 0; end; else; l_u_len = 0; end

    if Dims.exp_flag == 3 || Dims.exp_flag == 4; num_state_args = l_d2_len + l_a2_len + l_z_len + l_e_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 2 || Dims.exp_flag == 5; num_state_args = l_d2_len + l_a2_len + l_bothz_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 6; num_state_args = l_d2_len + l_a2_len + l_u_len + (l_a2_len >= 2);
    else; num_state_args = l_d2_len + l_a2_len + (l_a2_len >= 2); end

    temp_args = getAnonymousFnInputNames(vfoptions.aprimeFn);
    if length(temp_args) > num_state_args; aprimeFnParamNames = temp_args(num_state_args + 1 : end); else; aprimeFnParamNames = {}; end
else
    aprimeFnParamNames = {};
end

% --- 4. Preallocate Outputs ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

a2_idx = reshape(gpuArray(0:1:N_a2-1), 1, 1, 1, N_a2);
bothz_idx = reshape(gpuArray(0:1:N_bothz-1), 1, 1, 1, 1, N_bothz);
semiz_idx = reshape(gpuArray(0:1:N_semiz-1), 1, 1, 1, 1, N_semiz);
e_idx_tensor = reshape(gpuArray(0:N_e-1), 1, 1, 1, 1, 1, N_e);

a1p_offsets_base = gpuArray(0:N_a1);

% --- 5. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]); else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    has_cont = ~isempty(V_next);
    if has_cont
        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, j);
        d2_gridvals = CreateGridvals(n_d2, vfoptions.d2_grid, 1);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, Grids.z_gridvals_J(:,:,j), n_e, get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e));
        elseif Dims.exp_flag == 2 || Dims.exp_flag == 5
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, Dims.n_bothz, get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz));
        elseif Dims.exp_flag == 6
            u_gridvals = CreateGridvals(vfoptions.n_u, vfoptions.u_grid, 1);
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, vfoptions.n_u, u_gridvals);
        else
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2);
        end
        ExpAsset_J.a2primeIndex = a2primeIndex;
        ExpAsset_J.a2primeProbs = a2primeProbs;
    else
        ExpAsset_J = [];
    end

    switch IterStrategy
        case 'Full'; [V(:,:,:,j), Pol_j] = solve_period_full(j, V_next, ExpAsset_J);
        case 'Loop_e'; [V(:,:,:,j), Pol_j] = solve_period_loop_e(j, V_next, ExpAsset_J);
        case 'Loop_z_parallel_semiz'; [V(:,:,:,j), Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next, ExpAsset_J);
        case 'Loop_bothz'; [V(:,:,:,j), Pol_j] = solve_period_loop_bothz(j, V_next, ExpAsset_J);
        case 'Loop_z_parallel_semiz_e'; [V(:,:,:,j), Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next, ExpAsset_J);
        case 'Loop_bothz_e'; [V(:,:,:,j), Pol_j] = solve_period_loop_bothz_e(j, V_next, ExpAsset_J);
    end
    Policy(:,:,:,:,j) = Pol_j;
end

% =====================================================================
% SPECIALIZED SOLVE FUNCTIONS
% =====================================================================
    function [V_j, Pol_j] = solve_period_full(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');
        Policy_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp_douter = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1, n_a2, N_d2, N_bothz, vfoptions, Dims);
            end

            for d2_c = 1:N_d2
                d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                d_val = Grids.d12_gridvals( (d_outer_c-1)*N_d_inner + (d2_c-1)*N_d1 + (1:N_d1), : );

                if has_cont
                    EV_interp_d2 = EV_interp_douter(d2_c : N_d2 : end, :, :);
                    DiscountedEV_dense = df * reshape(EV_interp_d2, [N_a1, N_a2, N_bothz, N_e_interp]);
                    DiscountedEV = reshape(DiscountedEV_dense, [1, N_a1, 1, N_a2, N_bothz, N_e_interp]);
                end

                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d1, N_a1, vfoptions.level1n, N_a2, N_bothz, N_e]);

                if has_cont; entireRHS_L1 = RM_L1 + DiscountedEV; else; entireRHS_L1 = RM_L1; end

                [~, maxindex1] = max(entireRHS_L1, [], 2);
                [Vtempii, maxindex2] = max(reshape(entireRHS_L1, [N_d1*N_a1, vfoptions.level1n, N_a2, N_bothz, N_e]), [], 1);
                V_ford23(level1ii,:,:,:,d23_idx) = shiftdim(Vtempii, 1);
                Policy_ford23(level1ii,:,:,:,d23_idx) = shiftdim(maxindex2, 1);

                diff_a1p = maxindex1(:, 1, 2:end, :, :, :) - maxindex1(:, 1, 1:end-1, :, :, :);
                maxgap_vec = gather(squeeze(max(max(max(max(diff_a1p, [], 6), [], 5), [], 4), [], 1)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(vfoptions.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = min(maxgap_vec(ii), N_a1 - 1);

                    loweredge = reshape(min(maxindex1(:, 1, ii, :, :, :), N_a1 - mg), [N_d1, 1, 1, N_a2, N_bothz, N_e]);

                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                    a1p_idx = loweredge + a1p_offsets;

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                        a1_gridvals(a1p_idx), a1_gridvals(curraindex,:), a2_gridvals, ...
                        bz_val, e_val, ReturnFnParamsVec, 2);

                    if has_cont
                        EV_idx = a1p_idx + N_a1 * a2_idx;
                        if N_bothz > 1; EV_idx = EV_idx + N_a1*N_a2*bothz_idx; end
                        if N_e_interp > 1; EV_idx = EV_idx + N_a1*N_a2*N_bothz*e_idx_tensor; end
                        entireRHS_L2 = RM_L2 + DiscountedEV_dense(EV_idx);
                    else
                        entireRHS_L2 = RM_L2;
                    end

                    entireRHS_L2_flat = reshape(entireRHS_L2, [N_d1*(mg+1), L, N_a2, N_bothz, N_e]);
                    [Vtempii, maxindexL2] = max(entireRHS_L2_flat, [], 1);

                    dind = rem(maxindexL2 - 1, N_d1) + 1;
                    a1p_offset = ceil(maxindexL2 / N_d1) - 1;

                    a2_offset = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2]);
                    bz_offset = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, N_bothz]);
                    e_offset  = reshape(gpuArray(0:N_e-1), [1, 1, 1, 1, N_e]);

                    le_idx = dind + N_d1 * a2_offset;
                    if N_bothz > 1; le_idx = le_idx + N_d1 * N_a2 * bz_offset; end
                    if N_e > 1; le_idx = le_idx + N_d1 * N_a2 * N_bothz * e_offset; end

                    pol_flat = dind + N_d1 * (loweredge(le_idx) + a1p_offset - 1);

                    V_ford23(curraindex, :, :, :, d23_idx) = reshape(Vtempii, [L, N_a2, N_bothz, N_e]);
                    Policy_ford23(curraindex, :, :, :, d23_idx) = reshape(pol_flat, [L, N_a2, N_bothz, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford23, Policy_ford23, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');
        Policy_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);

        for d_outer_c = 1:N_d_outer
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp_douter = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1, n_a2, N_d2, N_bothz, vfoptions, Dims);
            end

            for d2_c = 1:N_d2
                d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                d_val = Grids.d12_gridvals( (d_outer_c-1)*N_d_inner + (d2_c-1)*N_d1 + (1:N_d1), : );
                if has_cont; EV_interp_d2 = reshape(EV_interp_douter(d2_c : N_d2 : end, :, :), [N_a1, N_a2, N_bothz, N_e_interp]); end

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    if has_cont
                        e_idx = min(e_c, N_e_interp);
                        DiscountedEV_dense = df * EV_interp_d2(:, :, :, e_idx);
                        DiscountedEV = reshape(DiscountedEV_dense, [1, N_a1, 1, N_a2, N_bothz, 1]);
                    end

                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d1, N_a1, vfoptions.level1n, N_a2, N_bothz, 1]);

                    if has_cont; entireRHS_L1 = RM_L1 + DiscountedEV; else; entireRHS_L1 = RM_L1; end

                    [~, maxindex1] = max(entireRHS_L1, [], 2);
                    [Vtempii, maxindex2] = max(reshape(entireRHS_L1, [N_d1*N_a1, vfoptions.level1n, N_a2, N_bothz]), [], 1);
                    V_ford23(level1ii,:,:,e_c,d23_idx) = shiftdim(Vtempii, 1);
                    Policy_ford23(level1ii,:,:,e_c,d23_idx) = shiftdim(maxindex2, 1);

                    diff_a1p = maxindex1(:, 1, 2:end, :, :, 1) - maxindex1(:, 1, 1:end-1, :, :, 1);
                    maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 5), [], 4), [], 1)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = min(maxgap_vec(ii), N_a1 - 1);

                        loweredge = reshape(min(maxindex1(:, 1, ii, :, :, 1), N_a1 - mg), [N_d1, 1, 1, N_a2, N_bothz, 1]);

                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;

                        RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                            a1_gridvals(a1p_idx), a1_gridvals(curraindex,:), a2_gridvals, ...
                            bz_val, e_val, ReturnFnParamsVec, 2);

                        if has_cont
                            EV_idx = a1p_idx + N_a1 * a2_idx;
                            if N_bothz > 1; EV_idx = EV_idx + N_a1*N_a2*bothz_idx; end
                            entireRHS_L2 = RM_L2 + DiscountedEV_dense(EV_idx);
                        else
                            entireRHS_L2 = RM_L2;
                        end

                        entireRHS_L2_flat = reshape(entireRHS_L2, [N_d1*(mg+1), L, N_a2, N_bothz]);
                        [Vtempii, maxindexL2] = max(entireRHS_L2_flat, [], 1);

                        dind = rem(maxindexL2 - 1, N_d1) + 1;
                        a1p_offset = ceil(maxindexL2 / N_d1) - 1;

                        a2_offset = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2]);
                        bz_offset = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, N_bothz]);

                        le_idx = dind + N_d1 * a2_offset;
                        if N_bothz > 1; le_idx = le_idx + N_d1 * N_a2 * bz_offset; end

                        pol_flat = dind + N_d1 * (loweredge(le_idx) + a1p_offset - 1);

                        V_ford23(curraindex, :, :, e_c, d23_idx) = reshape(Vtempii, [L, N_a2, N_bothz, 1]);
                        Policy_ford23(curraindex, :, :, e_c, d23_idx) = reshape(pol_flat, [L, N_a2, N_bothz, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford23, Policy_ford23, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');
        Policy_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp_douter = reshape(precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1, n_a2, N_d2, N_bothz, vfoptions, Dims), [N_d2 * N_a1 * N_a2, N_semiz, N_z, N_e_interp]);
            end

            for d2_c = 1:N_d2
                d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                d_val = d_gridvals( (d_outer_c-1)*N_d_inner + (d2_c-1)*N_d1 + (1:N_d1), : );
                if has_cont; EV_interp_d2 = EV_interp_douter(d2_c : N_d2 : end, :, :, :); end

                for z_c = 1:N_z
                    sb = (z_c-1)*N_semiz + (1:N_semiz);
                    bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                    if has_cont
                        DiscountedEV_dense = df * reshape(EV_interp_d2(:,:,z_c,:), [N_a1, N_a2, N_semiz, N_e_interp]);
                        DiscountedEV = reshape(DiscountedEV_dense, [1, N_a1, 1, N_a2, N_semiz, N_e_interp]);
                    end

                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d1, N_a1, vfoptions.level1n, N_a2, N_semiz, N_e]);

                    if has_cont; entireRHS_L1 = RM_L1 + DiscountedEV; else; entireRHS_L1 = RM_L1; end

                    [~, maxindex1] = max(entireRHS_L1, [], 2);
                    [Vtempii, maxindex2] = max(reshape(entireRHS_L1, [N_d1*N_a1, vfoptions.level1n, N_a2, N_semiz, N_e]), [], 1);
                    V_ford23(level1ii, :, sb, :, d23_idx) = shiftdim(Vtempii, 1);
                    Policy_ford23(level1ii, :, sb, :, d23_idx) = shiftdim(maxindex2, 1);

                    diff_a1p = maxindex1(:, 1, 2:end, :, :, :) - maxindex1(:, 1, 1:end-1, :, :, :);
                    maxgap_vec = gather(squeeze(max(max(max(max(diff_a1p, [], 6), [], 5), [], 4), [], 1)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = min(maxgap_vec(ii), N_a1 - 1);

                        loweredge = reshape(min(maxindex1(:, 1, ii, :, :, :), N_a1 - mg), [N_d1, 1, 1, N_a2, N_semiz, N_e]);

                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;

                        RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, ...
                            a1_gridvals(a1p_idx), a1_gridvals(curraindex,:), a2_gridvals, ...
                            bz_val, e_val, ReturnFnParamsVec, 2);

                        if has_cont
                            EV_idx = a1p_idx + N_a1 * a2_idx;
                            if N_semiz > 1; EV_idx = EV_idx + N_a1*N_a2*semiz_idx; end
                            if N_e_interp > 1; EV_idx = EV_idx + N_a1*N_a2*N_semiz*e_idx_tensor; end
                            entireRHS_L2 = RM_L2 + DiscountedEV_dense(EV_idx);
                        else
                            entireRHS_L2 = RM_L2;
                        end

                        entireRHS_L2_flat = reshape(entireRHS_L2, [N_d1*(mg+1), L, N_a2, N_semiz, N_e]);
                        [Vtempii, maxindexL2] = max(entireRHS_L2_flat, [], 1);

                        dind = rem(maxindexL2 - 1, N_d1) + 1;
                        a1p_offset = ceil(maxindexL2 / N_d1) - 1;

                        a2_offset = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2]);
                        semiz_offset = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, N_semiz]);
                        e_offset = reshape(gpuArray(0:N_e-1), [1, 1, 1, 1, N_e]);

                        le_idx = dind + N_d1 * a2_offset;
                        if N_semiz > 1; le_idx = le_idx + N_d1 * N_a2 * semiz_offset; end
                        if N_e > 1; le_idx = le_idx + N_d1 * N_a2 * N_semiz * e_offset; end

                        pol_flat = dind + N_d1 * (loweredge(le_idx) + a1p_offset - 1);

                        V_ford23(curraindex, :, sb, :, d23_idx) = reshape(Vtempii, [L, N_a2, N_semiz, N_e]);
                        Policy_ford23(curraindex, :, sb, :, d23_idx) = reshape(pol_flat, [L, N_a2, N_semiz, N_e]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford23, Policy_ford23, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');
        Policy_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp_douter = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1, n_a2, N_d2, N_bothz, vfoptions, Dims);
            end

            for d2_c = 1:N_d2
                d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                d_val = d_gridvals( (d_outer_c-1)*N_d_inner + (d2_c-1)*N_d1 + (1:N_d1), : );
                if has_cont; EV_interp_d2 = reshape(EV_interp_douter(d2_c : N_d2 : end, :, :), [N_a1, N_a2, N_bothz, N_e_interp]); end

                for z_c = 1:N_bothz
                    bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                    if has_cont
                        DiscountedEV_dense = df * EV_interp_d2(:, :, z_c, :);
                        DiscountedEV = reshape(DiscountedEV_dense, [1, N_a1, 1, N_a2, 1, N_e_interp]);
                    end

                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d1, N_a1, vfoptions.level1n, N_a2, 1, N_e]);

                    if has_cont; entireRHS_L1 = RM_L1 + DiscountedEV; else; entireRHS_L1 = RM_L1; end

                    [~, maxindex1] = max(entireRHS_L1, [], 2);
                    [Vtempii, maxindex2] = max(reshape(entireRHS_L1, [N_d1*N_a1, vfoptions.level1n, N_a2, N_e]), [], 1);
                    V_ford23(level1ii, :, z_c, :, d23_idx) = shiftdim(Vtempii, 1);
                    Policy_ford23(level1ii, :, z_c, :, d23_idx) = shiftdim(maxindex2, 1);

                    diff_a1p = maxindex1(:, 1, 2:end, :, 1, :) - maxindex1(:, 1, 1:end-1, :, 1, :);
                    maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 6), [], 4)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = min(maxgap_vec(ii), N_a1 - 1);

                        loweredge = reshape(min(maxindex1(:, 1, ii, :, 1, :), N_a1 - mg), [N_d1, 1, 1, N_a2, 1, N_e]);

                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;

                        RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                            a1_gridvals(a1p_idx), a1_gridvals(curraindex,:), a2_gridvals, ...
                            bz_val, e_val, ReturnFnParamsVec, 2);

                        if has_cont
                            EV_idx = a1p_idx + N_a1 * a2_idx;
                            if N_e_interp > 1; EV_idx = EV_idx + N_a1*N_a2*e_idx_tensor; end
                            entireRHS_L2 = RM_L2 + DiscountedEV_dense(EV_idx);
                        else
                            entireRHS_L2 = RM_L2;
                        end

                        entireRHS_L2_flat = reshape(entireRHS_L2, [N_d1*(mg+1), L, N_a2, 1, N_e]);
                        [Vtempii, maxindexL2] = max(entireRHS_L2_flat, [], 1);

                        dind = rem(maxindexL2 - 1, N_d1) + 1;
                        a1p_offset = ceil(maxindexL2 / N_d1) - 1;

                        a2_offset = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2]);
                        e_offset = reshape(gpuArray(0:N_e-1), [1, 1, 1, 1, N_e]);

                        le_idx = dind + N_d1 * a2_offset;
                        if N_e > 1; le_idx = le_idx + N_d1 * N_a2 * e_offset; end

                        pol_flat = dind + N_d1 * (loweredge(le_idx) + a1p_offset - 1);

                        V_ford23(curraindex, :, z_c, :, d23_idx) = reshape(Vtempii, [L, N_a2, 1, N_e]);
                        Policy_ford23(curraindex, :, z_c, :, d23_idx) = reshape(pol_flat, [L, N_a2, 1, N_e]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford23, Policy_ford23, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');
        Policy_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');
        semiz_idx = reshape(gpuArray(0:1:N_semiz-1), 1, 1, 1, N_semiz);

        for d_outer_c = 1:N_d_outer
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp_douter = reshape(precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1, n_a2, N_d2, N_bothz, vfoptions, Dims), [N_d2 * N_a1 * N_a2, N_semiz, N_z, N_e_interp]);
            end

            for d2_c = 1:N_d2
                d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                d_val = d_gridvals( (d_outer_c-1)*N_d_inner + (d2_c-1)*N_d1 + (1:N_d1), : );
                if has_cont; EV_interp_d2 = EV_interp_douter(d2_c : N_d2 : end, :, :, :, :); end

                for z_c = 1:N_z
                    sb = (z_c-1)*N_semiz + (1:N_semiz);
                    bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);
                    for e_c = 1:N_e
                        e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                        if has_cont
                            e_idx = min(e_c, N_e_interp);
                            DiscountedEV_dense = df * reshape(EV_interp_d2(:,:,sb,z_c,e_idx), [N_a1, N_a2, N_semiz, 1]);
                            DiscountedEV = reshape(DiscountedEV_dense, [1, N_a1, 1, N_a2, N_semiz, 1]);
                        end

                        RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                        RM_L1 = reshape(RM_L1, [N_d1, N_a1, vfoptions.level1n, N_a2, N_semiz, 1]);

                        if has_cont; entireRHS_L1 = RM_L1 + DiscountedEV; else; entireRHS_L1 = RM_L1; end

                        [~, maxindex1] = max(entireRHS_L1, [], 2);
                        [Vtempii, maxindex2] = max(reshape(entireRHS_L1, [N_d1*N_a1, vfoptions.level1n, N_a2, N_semiz]), [], 1);
                        V_ford23(level1ii, :, sb, e_c, d23_idx) = shiftdim(Vtempii, 1);
                        Policy_ford23(level1ii, :, sb, e_c, d23_idx) = shiftdim(maxindex2, 1);

                        diff_a1p = maxindex1(:, 1, 2:end, :, :, 1) - maxindex1(:, 1, 1:end-1, :, :, 1);
                        maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 5), [], 4)));
                        if isempty(maxgap_vec); maxgap_vec = 0; end

                        for ii = 1:(vfoptions.level1n-1)
                            curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                            L = length(curraindex);
                            if L == 0; continue; end

                            mg = min(maxgap_vec(ii), N_a1 - 1);

                            loweredge = reshape(min(maxindex1(:, 1, ii, :, :, 1), N_a1 - mg), [N_d1, 1, 1, N_a2, N_semiz, 1]);

                            a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                            a1p_idx = loweredge + a1p_offsets;

                            RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, ...
                                a1_gridvals(a1p_idx), a1_gridvals(curraindex,:), a2_gridvals, ...
                                bz_val, e_val, ReturnFnParamsVec, 2);

                            if has_cont
                                EV_idx = a1p_idx + N_a1 * a2_idx;
                                if N_semiz > 1; EV_idx = EV_idx + N_a1*N_a2*semiz_idx; end
                                entireRHS_L2 = RM_L2 + DiscountedEV_dense(EV_idx);
                            else
                                entireRHS_L2 = RM_L2;
                            end

                            entireRHS_L2_flat = reshape(entireRHS_L2, [N_d1*(mg+1), L, N_a2, N_semiz]);
                            [Vtempii, maxindexL2] = max(entireRHS_L2_flat, [], 1);

                            dind = rem(maxindexL2 - 1, N_d1) + 1;
                            a1p_offset = ceil(maxindexL2 / N_d1) - 1;

                            a2_offset = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2]);
                            semiz_offset = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, N_semiz]);

                            le_idx = dind + N_d1 * a2_offset;
                            if N_semiz > 1; le_idx = le_idx + N_d1 * N_a2 * semiz_offset; end

                            pol_flat = dind + N_d1 * (loweredge(le_idx) + a1p_offset - 1);

                            V_ford23(curraindex, :, sb, e_c, d23_idx) = reshape(Vtempii, [L, N_a2, N_semiz, 1]);
                            Policy_ford23(curraindex, :, sb, e_c, d23_idx) = reshape(pol_flat, [L, N_a2, N_semiz, 1]);
                        end
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford23, Policy_ford23, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');
        Policy_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp_douter = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1, n_a2, N_d2, N_bothz, vfoptions, Dims);
            end

            for d2_c = 1:N_d2
                d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                d_val = d_gridvals( (d_outer_c-1)*N_d_inner + (d2_c-1)*N_d1 + (1:N_d1), : );
                if has_cont; EV_interp_d2 = reshape(EV_interp_douter(d2_c : N_d2 : end, :, :), [N_a1, N_a2, N_bothz, N_e_interp]); end

                for z_c = 1:N_bothz
                    bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);
                    for e_c = 1:N_e
                        e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                        if has_cont
                            e_idx = min(e_c, N_e_interp);
                            DiscountedEV_dense = df * EV_interp_d2(:, :, z_c, e_idx);
                            DiscountedEV = reshape(DiscountedEV_dense, [1, N_a1, 1, N_a2, 1, 1]);
                        end

                        RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                        RM_L1 = reshape(RM_L1, [N_d1, N_a1, vfoptions.level1n, N_a2, 1, 1]);

                        if has_cont; entireRHS_L1 = RM_L1 + DiscountedEV; else; entireRHS_L1 = RM_L1; end

                        [~, maxindex1] = max(entireRHS_L1, [], 2);
                        [Vtempii, maxindex2] = max(reshape(entireRHS_L1, [N_d1*N_a1, vfoptions.level1n, N_a2]), [], 1);
                        V_ford23(level1ii, :, z_c, e_c, d23_idx) = shiftdim(Vtempii, 1);
                        Policy_ford23(level1ii, :, z_c, e_c, d23_idx) = shiftdim(maxindex2, 1);

                        diff_a1p = maxindex1(:, 1, 2:end, :, 1, 1) - maxindex1(:, 1, 1:end-1, :, 1, 1);
                        maxgap_vec = gather(squeeze(max(diff_a1p, [], 4)));
                        if isempty(maxgap_vec); maxgap_vec = 0; end

                        for ii = 1:(vfoptions.level1n-1)
                            curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                            L = length(curraindex);
                            if L == 0; continue; end

                            mg = min(maxgap_vec(ii), N_a1 - 1);

                            loweredge = reshape(min(maxindex1(:, 1, ii, :, 1, 1), N_a1 - mg), [N_d1, 1, 1, N_a2, 1, 1]);

                            a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                            a1p_idx = loweredge + a1p_offsets;

                            RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                                a1_gridvals(a1p_idx), a1_gridvals(curraindex,:), a2_gridvals, ...
                                bz_val, e_val, ReturnFnParamsVec, 2);

                            if has_cont
                                EV_idx = a1p_idx + N_a1 * a2_idx;
                                entireRHS_L2 = RM_L2 + DiscountedEV_dense(EV_idx);
                            else
                                entireRHS_L2 = RM_L2;
                            end

                            entireRHS_L2_flat = reshape(entireRHS_L2, [N_d1*(mg+1), L, N_a2]);
                            [Vtempii, maxindexL2] = max(entireRHS_L2_flat, [], 1);

                            dind = rem(maxindexL2 - 1, N_d1) + 1;
                            a1p_offset = ceil(maxindexL2 / N_d1) - 1;

                            a2_offset = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2]);
                            le_idx = dind + N_d1 * a2_offset;

                            pol_flat = dind + N_d1 * (loweredge(le_idx) + a1p_offset - 1);

                            V_ford23(curraindex, :, z_c, e_c, d23_idx) = reshape(Vtempii, [L, N_a2, 1, 1]);
                            Policy_ford23(curraindex, :, z_c, e_c, d23_idx) = reshape(pol_flat, [L, N_a2, 1, 1]);
                        end
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford23, Policy_ford23, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function val = get_grid_slice(grid, idx, j, has_flag)
        if has_flag
            if ischar(idx)
                val = grid(:,:,j);
            else
                val = grid(idx,:,j);
            end
        else
            val = [];
        end
    end

    function [V_out, Pol_out] = extract_pol(V_douter, Pol_douter, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer)
        [V_tmp, maxindex_douter] = max(V_douter, [], 5);
        V_out = reshape(V_tmp, [N_a1 * N_a2, N_bothz, N_e]);

        linear_douter = reshape(maxindex_douter, [N_a1 * N_a2 * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a1 * N_a2 * N_bothz * N_e))' + (N_a1 * N_a2 * N_bothz * N_e) * (linear_douter - 1);
        inner_pol = reshape(Pol_douter(lin_idx), [1, N_a1 * N_a2, N_bothz, N_e]);

        d12_idx = rem(inner_pol - 1, N_d_inner) + 1;
        a1prime_idx = ceil(inner_pol / N_d_inner);
        douter_reshaped = reshape(maxindex_douter - 1, [1, N_a1 * N_a2, N_bothz, N_e]);

        Pol_out = d12_idx + douter_reshaped * N_d_inner + (a1prime_idx - 1) * (N_d_inner * N_d_outer);
    end

    function EV_interp = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1, n_a2, N_d2, N_bothz, vfoptions, Dims)
        % For experienceassetz, we MUST Interpolate FIRST, then take the Expectation over z.
        if Dims.exp_flag == 6
            N_u = prod(vfoptions.n_u); EV_interp_next = zeros(N_d2 * prod(n_a1) * prod(n_a2), N_bothz, 'gpuArray');
            for u_c = 1:N_u
                EV_interp_u = InterpolateExpAssetEV(EVpre_flat, n_a2, N_d2, n_a1, n_a2, N_bothz, ExpAsset_J.a2primeIndex(:,:,u_c), ExpAsset_J.a2primeProbs(:,:,u_c));
                EV_interp_next = EV_interp_next + vfoptions.pi_u(u_c) * reshape(EV_interp_u, [N_d2 * prod(n_a1) * prod(n_a2), N_bothz]);
            end
        else
            num_z_interp = N_bothz * N_e_interp;
            EVpre_rep = repmat(EVpre_flat, [1, N_e_interp]);

            a2p_idx = ExpAsset_J.a2primeIndex;
            a2p_prb = ExpAsset_J.a2primeProbs;
            if size(a2p_idx, 3) < num_z_interp && size(a2p_idx, 3) > 1
                a2p_idx = repmat(a2p_idx, [1, 1, num_z_interp / size(a2p_idx, 3)]);
                a2p_prb = repmat(a2p_prb, [1, 1, num_z_interp / size(a2p_prb, 3)]);
            end

            a2p_idx_flat = reshape(a2p_idx, [N_d2, prod(n_a2), num_z_interp]);
            a2p_prb_flat = reshape(a2p_prb, [N_d2, prod(n_a2), num_z_interp]);

            EV_interp_next = InterpolateExpAssetEV(EVpre_rep, n_a2, N_d2, n_a1, n_a2, num_z_interp, a2p_idx_flat, a2p_prb_flat);
        end

        EV_interp_next = reshape(EV_interp_next, [N_d2 * prod(n_a1) * prod(n_a2), N_bothz, N_e_interp]);
        EV_expected = zeros(size(EV_interp_next), 'like', EV_interp_next);

        for e_c = 1:N_e_interp
            EVinf = (EV_interp_next(:,:,e_c) == -Inf);
            EV_val = EV_interp_next(:,:,e_c);
            EV_val(EVinf) = -1e250;
            EV_exp = EV_val * pi_bz';
            EV_exp(EVinf * (pi_bz' > 0) > 0) = -Inf;
            EV_expected(:,:,e_c) = EV_exp;
        end

        EV_interp = EV_expected;
    end


end
