function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;

has_semiz = Dims.N_semiz_raw > 0;
has_e     = Dims.N_e_raw > 0;
has_z     = Dims.special_n_z(1) > 0;
has_bothz = has_semiz || has_z;

n_semiz = Dims.n_semiz; n_e = Dims.n_e;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
n_d1 = Dims.n_d1; n_d2 = Dims.n_d2;
N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;

% ExpAsset DC2A specific split: a1_c, a2_c (choices) vs a_exp
n_a1_c = Dims.n_a1(1);
n_a2_c = max(prod(Dims.n_a1(2:end)), 1);
n_exp = Dims.n_a2; N_exp = Dims.N_a2;

a1_c_gridvals = CreateGridvals(n_a1_c, vfoptions.a1_grid(1:n_a1_c), 1);
a2_c_gridvals = CreateGridvals(Dims.n_a1(2:end), vfoptions.a1_grid(n_a1_c+1:end), 1);
exp_gridvals  = CreateGridvals(n_exp, vfoptions.a2_grid, 1);

level1ii = round(linspace(1, n_a1_c, Dims.level1n));

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

if Dims.exp_flag > 0
    l_d2_len = length(n_d2); if sum(n_d2) == 0; l_d2_len = 0; end
    l_exp_len = length(n_exp); if sum(n_exp) == 0; l_exp_len = 0; end
    l_z_len  = length(n_z);  if sum(n_z) == 0;  l_z_len = 0; end
    l_e_len  = length(n_e);  if sum(n_e) == 0;  l_e_len = 0; end
    l_semiz_len = length(n_semiz); if sum(n_semiz) == 0; l_semiz_len = 0; end
    if isfield(vfoptions, 'n_u'); l_u_len = length(vfoptions.n_u); if sum(vfoptions.n_u) == 0; l_u_len = 0; end; else; l_u_len = 0; end

    if Dims.exp_flag == 3 || Dims.exp_flag == 4; num_state_args = l_d2_len + l_exp_len + l_z_len + l_e_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 2; num_state_args = l_d2_len + l_exp_len + l_z_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 5; num_state_args = l_d2_len + l_exp_len + l_semiz_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 6; num_state_args = l_d2_len + l_exp_len + l_u_len + (l_exp_len >= 2);
    else; num_state_args = l_d2_len + l_exp_len + (l_exp_len >= 2); end

    temp_args = getAnonymousFnInputNames(vfoptions.aprimeFn);
    if length(temp_args) > num_state_args; aprimeFnParamNames = temp_args(num_state_args + 1 : end); else; aprimeFnParamNames = {}; end
else
    aprimeFnParamNames = {};
end

% --- 3. Preallocate Outputs ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:n_a1_c);

% --- 4. Backward Induction Loop ---
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
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, Grids.z_gridvals_J(:,:,j), n_e, get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e));
        elseif Dims.exp_flag == 2
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, z_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 5
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_semiz, get_grid_slice(vfoptions.semiz_gridvals_J, 'all', j, has_semiz));
        elseif Dims.exp_flag == 6
            u_gridvals = CreateGridvals(vfoptions.n_u, vfoptions.u_grid, 1);
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, vfoptions.n_u, u_gridvals);
        else
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2);
        end
        ExpAsset_J.a2primeIndex = expIndex;
        ExpAsset_J.a2primeProbs = expProbs;
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
% SPECIALIZED SOLVE FUNCTIONS (No d2 loop, natively vectorizes over DC envelope)
% =====================================================================
    function [V_j, Pol_j] = solve_period_full(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));
        d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1, 1, 1]);
        a2p_off_L2 = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1, 1, 1, 1, 1]);
        exp_off_L2 = reshape(gpuArray(0:N_exp-1), [1, 1, 1, 1, 1, N_exp, 1, 1]);
        bz_off_L2  = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, 1, N_bothz, 1]);
        e_off_L2   = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, 1, 1, N_e_interp]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims);
                DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_bothz, N_e_interp]);
                EV_flat = reshape(EV_coarse, [N_d2 * n_a1_c * n_a2_c * N_exp * N_bothz * N_e_interp, 1]);
            end

            RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals(level1ii,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
            RM_L1 = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, Dims.level1n, n_a2_c, N_exp, N_bothz, N_e]);

            if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

            RHS_L1_flat = reshape(RHS_L1, [N_d_inner * n_a1_c * n_a2_c, Dims.level1n, n_a2_c, N_exp, N_bothz, N_e]);
            [~, maxindex1] = max(RHS_L1_flat, [], 1);

            a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
            a1p_idx_L1 = rem(a1p_a2p_ind - 1, n_a1_c) + 1;

            [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*n_a1_c*n_a2_c, Dims.level1n, n_a2_c, N_exp, N_bothz, N_e]), [], 1);
            V_fordouter(level1ii,:,:,:,:,d_outer_c) = reshape(v_tmp, [Dims.level1n, n_a2_c, N_exp, N_bothz, N_e]);
            Policy_fordouter(level1ii,:,:,:,:,d_outer_c) = reshape(p_tmp, [Dims.level1n, n_a2_c, N_exp, N_bothz, N_e]);

            diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, :, :) - a1p_idx_L1(1, 1:end-1, :, :, :, :));
            maxgap_vec = gather(squeeze(max(max(max(max(diff_a1p, [], 6), [], 5), [], 4), [], 3)));
            if isempty(maxgap_vec); maxgap_vec = 0; end

            for ii = 1:(Dims.level1n-1)
                curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                L = length(curraindex);
                if L == 0; continue; end

                mg = max(min(maxgap_vec(ii), n_a1_c - 1), 0);
                loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, :, :), n_a1_c - mg), [1, 1, 1, 1, n_a2_c, N_exp, N_bothz, N_e]);
                a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1, 1]);
                a1p_idx = loweredge + a1p_offsets;
                a1p_grid_safe = reshape(a1_c_gridvals(a1p_idx), size(a1p_idx));

                RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                    a1p_grid_safe, a2_c_gridvals, a1_c_gridvals(curraindex,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, n_a2_c, L, n_a2_c, N_exp, N_bothz, N_e]);

                if has_cont
                    EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*n_a1_c*a2p_off_L2 + N_d2*n_a1_c*n_a2_c*exp_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*bz_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*N_bothz*e_off_L2;
                    RHS_L2 = RM_L2 + df * EV_flat(EV_idx);
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner*(mg+1)*n_a2_c, L, n_a2_c, N_exp, N_bothz, N_e]);
                [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                dind = rem(maxindex2 - 1, N_d_inner) + 1;
                a1p_a2p_idx = ceil(maxindex2 / N_d_inner);
                a1p_offset = rem(a1p_a2p_idx - 1, mg+1) + 1;
                a2p_ind = ceil(a1p_a2p_idx / (mg+1));

                le_flat = reshape(loweredge, [1, 1, n_a2_c, N_exp, N_bothz, N_e]);
                a1p_abs = le_flat + a1p_offset - 1;
                pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * n_a1_c * (a2p_ind - 1);

                V_fordouter(curraindex, :, :, :, :, d_outer_c) = reshape(v_tmp, [L, n_a2_c, N_exp, N_bothz, N_e]);
                Policy_fordouter(curraindex, :, :, :, :, d_outer_c) = reshape(pol_flat, [L, n_a2_c, N_exp, N_bothz, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));
        d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1, 1]);
        a2p_off_L2 = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1, 1, 1, 1]);
        exp_off_L2 = reshape(gpuArray(0:N_exp-1), [1, 1, 1, 1, 1, N_exp, 1]);
        bz_off_L2  = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, 1, N_bothz]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims);
            end

            for e_c = 1:N_e
                e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                if has_cont
                    e_idx = min(e_c, N_e_interp);
                    e_base_idx = N_d2*n_a1_c*n_a2_c*N_exp*N_bothz * (e_idx - 1);
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, e_idx), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_bothz, 1]);
                    EV_flat_e = reshape(EV_coarse(:,:,:,:,:,e_idx), [N_d2 * n_a1_c * n_a2_c * N_exp * N_bothz, 1]);
                end

                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals(level1ii,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, Dims.level1n, n_a2_c, N_exp, N_bothz, 1]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * n_a1_c * n_a2_c, Dims.level1n, n_a2_c, N_exp, N_bothz, 1]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, n_a1_c) + 1;

                [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*n_a1_c*n_a2_c, Dims.level1n, n_a2_c, N_exp, N_bothz, 1]), [], 1);
                V_fordouter(level1ii,:,:,:,e_c,d_outer_c) = reshape(v_tmp, [Dims.level1n, n_a2_c, N_exp, N_bothz, 1]);
                Policy_fordouter(level1ii,:,:,:,e_c,d_outer_c) = reshape(p_tmp, [Dims.level1n, n_a2_c, N_exp, N_bothz, 1]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, :, 1) - a1p_idx_L1(1, 1:end-1, :, :, :, 1));
                maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 5), [], 4), [], 3)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(Dims.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), n_a1_c - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, :, 1), n_a1_c - mg), [1, 1, 1, 1, n_a2_c, N_exp, N_bothz, 1]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                    a1p_idx = loweredge + a1p_offsets;
                    a1p_grid_safe = reshape(a1_c_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                        a1p_grid_safe, a2_c_gridvals, a1_c_gridvals(curraindex,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, n_a2_c, L, n_a2_c, N_exp, N_bothz, 1]);

                    if has_cont
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*n_a1_c*a2p_off_L2 + N_d2*n_a1_c*n_a2_c*exp_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*bz_off_L2;
                        RHS_L2 = RM_L2 + df * EV_flat_e(EV_idx);
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner*(mg+1)*n_a2_c, L, n_a2_c, N_exp, N_bothz, 1]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_a2p_idx = ceil(maxindex2 / N_d_inner);
                    a1p_offset = rem(a1p_a2p_idx - 1, mg+1) + 1;
                    a2p_ind = ceil(a1p_a2p_idx / (mg+1));

                    le_flat = reshape(loweredge, [1, 1, n_a2_c, N_exp, N_bothz, 1]);
                    a1p_abs = le_flat + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * n_a1_c * (a2p_ind - 1);

                    V_fordouter(curraindex, :, :, :, e_c, d_outer_c) = reshape(v_tmp, [L, n_a2_c, N_exp, N_bothz, 1]);
                    Policy_fordouter(curraindex, :, :, :, e_c, d_outer_c) = reshape(pol_flat, [L, n_a2_c, N_exp, N_bothz, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);
        d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));
        d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1, 1]);
        a2p_off_L2 = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1, 1, 1, 1]);
        exp_off_L2 = reshape(gpuArray(0:N_exp-1), [1, 1, 1, 1, 1, N_exp, 1]);
        sb_off_L2  = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, 1, N_semiz]);
        e_off_L2   = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, 1, 1, N_e_interp]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims);
                EV_coarse = reshape(EV_coarse, [N_d2, n_a1_c, n_a2_c, N_exp, N_semiz, N_z, N_e_interp]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                if has_cont
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, z_c, :), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_semiz, N_e_interp]);
                    EV_flat_z = reshape(EV_coarse(:,:,:,:,:,z_c,:), [N_d2 * n_a1_c * n_a2_c * N_exp * N_semiz * N_e_interp, 1]);
                end

                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals(level1ii,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, Dims.level1n, n_a2_c, N_exp, N_semiz, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * n_a1_c * n_a2_c, Dims.level1n, n_a2_c, N_exp, N_semiz, N_e]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, n_a1_c) + 1;

                [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*n_a1_c*n_a2_c, Dims.level1n, n_a2_c, N_exp, N_semiz, N_e]), [], 1);
                V_fordouter(level1ii,:,:,sb,:,d_outer_c) = reshape(v_tmp, [Dims.level1n, n_a2_c, N_exp, N_semiz, N_e]);
                Policy_fordouter(level1ii,:,:,sb,:,d_outer_c) = reshape(p_tmp, [Dims.level1n, n_a2_c, N_exp, N_semiz, N_e]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, :, :) - a1p_idx_L1(1, 1:end-1, :, :, :, :));
                maxgap_vec = gather(squeeze(max(max(max(max(diff_a1p, [], 6), [], 5), [], 4), [], 3)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(Dims.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), n_a1_c - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, :, :), n_a1_c - mg), [1, 1, 1, 1, n_a2_c, N_exp, N_semiz, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1, 1]);
                    a1p_idx = loweredge + a1p_offsets;
                    a1p_grid_safe = reshape(a1_c_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, ...
                        a1p_grid_safe, a2_c_gridvals, a1_c_gridvals(curraindex,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, n_a2_c, L, n_a2_c, N_exp, N_semiz, N_e]);

                    if has_cont
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*n_a1_c*a2p_off_L2 + N_d2*n_a1_c*n_a2_c*exp_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*sb_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*N_semiz*e_off_L2;
                        RHS_L2 = RM_L2 + df * EV_flat_z(EV_idx);
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner*(mg+1)*n_a2_c, L, n_a2_c, N_exp, N_semiz, N_e]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_a2p_idx = ceil(maxindex2 / N_d_inner);
                    a1p_offset = rem(a1p_a2p_idx - 1, mg+1) + 1;
                    a2p_ind = ceil(a1p_a2p_idx / (mg+1));

                    le_flat = reshape(loweredge, [1, 1, n_a2_c, N_exp, N_semiz, N_e]);
                    a1p_abs = le_flat + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * n_a1_c * (a2p_ind - 1);

                    V_fordouter(curraindex, :, :, sb, :, d_outer_c) = reshape(v_tmp, [L, n_a2_c, N_exp, N_semiz, N_e]);
                    Policy_fordouter(curraindex, :, :, sb, :, d_outer_c) = reshape(pol_flat, [L, n_a2_c, N_exp, N_semiz, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);
        d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));
        d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1, 1]);
        a2p_off_L2 = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1, 1, 1, 1]);
        exp_off_L2 = reshape(gpuArray(0:N_exp-1), [1, 1, 1, 1, 1, N_exp, 1]);
        e_off_L2   = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, 1, N_e_interp]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                if has_cont
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, z_c, :), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, 1, N_e_interp]);
                    EV_flat_bz = reshape(EV_coarse(:,:,:,:,z_c,:), [N_d2 * n_a1_c * n_a2_c * N_exp * N_e_interp, 1]);
                end

                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals(level1ii,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, Dims.level1n, n_a2_c, N_exp, 1, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * n_a1_c * n_a2_c, Dims.level1n, n_a2_c, N_exp, 1, N_e]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, n_a1_c) + 1;

                [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*n_a1_c*n_a2_c, Dims.level1n, n_a2_c, N_exp, 1, N_e]), [], 1);
                V_fordouter(level1ii,:,:,z_c,:,d_outer_c) = reshape(v_tmp, [Dims.level1n, n_a2_c, N_exp, 1, N_e]);
                Policy_fordouter(level1ii,:,:,z_c,:,d_outer_c) = reshape(p_tmp, [Dims.level1n, n_a2_c, N_exp, 1, N_e]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, 1, :) - a1p_idx_L1(1, 1:end-1, :, :, 1, :));
                maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 6), [], 4), [], 3)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(Dims.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), n_a1_c - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, 1, :), n_a1_c - mg), [1, 1, 1, 1, n_a2_c, N_exp, 1, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1, 1]);
                    a1p_idx = loweredge + a1p_offsets;
                    a1p_grid_safe = reshape(a1_c_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                        a1p_grid_safe, a2_c_gridvals, a1_c_gridvals(curraindex,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, n_a2_c, L, n_a2_c, N_exp, 1, N_e]);

                    if has_cont
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*n_a1_c*a2p_off_L2 + N_d2*n_a1_c*n_a2_c*exp_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*e_off_L2;
                        RHS_L2 = RM_L2 + df * EV_flat_bz(EV_idx);
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner*(mg+1)*n_a2_c, L, n_a2_c, N_exp, 1, N_e]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_a2p_idx = ceil(maxindex2 / N_d_inner);
                    a1p_offset = rem(a1p_a2p_idx - 1, mg+1) + 1;
                    a2p_ind = ceil(a1p_a2p_idx / (mg+1));

                    le_flat = reshape(loweredge, [1, 1, n_a2_c, N_exp, 1, N_e]);
                    a1p_abs = le_flat + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * n_a1_c * (a2p_ind - 1);

                    V_fordouter(curraindex, :, :, z_c, :, d_outer_c) = reshape(v_tmp, [L, n_a2_c, N_exp, 1, N_e]);
                    Policy_fordouter(curraindex, :, :, z_c, :, d_outer_c) = reshape(pol_flat, [L, n_a2_c, N_exp, 1, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');

        d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));
        d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1]);
        a2p_off_L2 = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1, 1, 1]);
        exp_off_L2 = reshape(gpuArray(0:N_exp-1), [1, 1, 1, 1, 1, N_exp]);
        sb_off_L2  = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims);
                EV_coarse = reshape(EV_coarse, [N_d2, n_a1_c, n_a2_c, N_exp, N_semiz, N_z, N_e_interp]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    if has_cont
                        e_idx = min(e_c, N_e_interp);
                        DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, z_c, e_idx), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_semiz, 1]);
                        EV_flat_ze = reshape(EV_coarse(:,:,:,:,:,z_c,e_idx), [N_d2 * n_a1_c * n_a2_c * N_exp * N_semiz, 1]);
                    end

                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals(level1ii,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, Dims.level1n, n_a2_c, N_exp, N_semiz, 1]);

                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * n_a1_c * n_a2_c, Dims.level1n, n_a2_c, N_exp, N_semiz, 1]);
                    [~, maxindex1] = max(RHS_L1_flat, [], 1);

                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1p_idx_L1 = rem(a1p_a2p_ind - 1, n_a1_c) + 1;

                    [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*n_a1_c*n_a2_c, Dims.level1n, n_a2_c, N_exp, N_semiz, 1]), [], 1);
                    V_fordouter(level1ii,:,:,sb,e_c,d_outer_c) = reshape(v_tmp, [Dims.level1n, n_a2_c, N_exp, N_semiz, 1]);
                    Policy_fordouter(level1ii,:,:,sb,e_c,d_outer_c) = reshape(p_tmp, [Dims.level1n, n_a2_c, N_exp, N_semiz, 1]);

                    diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, :, 1) - a1p_idx_L1(1, 1:end-1, :, :, :, 1));
                    maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 5), [], 4), [], 3)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(Dims.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), n_a1_c - 1), 0);
                        loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, :, 1), n_a1_c - mg), [1, 1, 1, 1, n_a2_c, N_exp, N_semiz, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;
                        a1p_grid_safe = reshape(a1_c_gridvals(a1p_idx), size(a1p_idx));

                        RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, ...
                            a1p_grid_safe, a2_c_gridvals, a1_c_gridvals(curraindex,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, n_a2_c, L, n_a2_c, N_exp, N_semiz, 1]);

                        if has_cont
                            EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*n_a1_c*a2p_off_L2 + N_d2*n_a1_c*n_a2_c*exp_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*sb_off_L2;
                            RHS_L2 = RM_L2 + df * EV_flat_ze(EV_idx);
                        else
                            RHS_L2 = RM_L2;
                        end

                        RHS_L2_flat = reshape(RHS_L2, [N_d_inner*(mg+1)*n_a2_c, L, n_a2_c, N_exp, N_semiz, 1]);
                        [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                        dind = rem(maxindex2 - 1, N_d_inner) + 1;
                        a1p_a2p_idx = ceil(maxindex2 / N_d_inner);
                        a1p_offset = rem(a1p_a2p_idx - 1, mg+1) + 1;
                        a2p_ind = ceil(a1p_a2p_idx / (mg+1));

                        le_flat = reshape(loweredge, [1, 1, n_a2_c, N_exp, N_semiz, 1]);
                        a1p_abs = le_flat + a1p_offset - 1;
                        pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * n_a1_c * (a2p_ind - 1);

                        V_fordouter(curraindex, :, :, sb, e_c, d_outer_c) = reshape(v_tmp, [L, n_a2_c, N_exp, N_semiz, 1]);
                        Policy_fordouter(curraindex, :, :, sb, e_c, d_outer_c) = reshape(pol_flat, [L, n_a2_c, N_exp, N_semiz, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');

        d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));
        d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1, 1]);
        a2p_off_L2 = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1, 1, 1, 1]);
        exp_off_L2 = reshape(gpuArray(0:N_exp-1), [1, 1, 1, 1, 1, N_exp, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    if has_cont
                        e_idx = min(e_c, N_e_interp);
                        DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, z_c, e_idx), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, 1, 1]);
                        EV_flat_bze = reshape(EV_coarse(:,:,:,:,z_c,e_idx), [N_d2 * n_a1_c * n_a2_c * N_exp, 1]);
                    end

                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals(level1ii,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, Dims.level1n, n_a2_c, N_exp, 1, 1]);

                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * n_a1_c * n_a2_c, Dims.level1n, n_a2_c, N_exp, 1, 1]);
                    [~, maxindex1] = max(RHS_L1_flat, [], 1);

                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1p_idx_L1 = rem(a1p_a2p_ind - 1, n_a1_c) + 1;

                    [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*n_a1_c*n_a2_c, Dims.level1n, n_a2_c, N_exp, 1, 1]), [], 1);
                    V_fordouter(level1ii,:,:,z_c,e_c,d_outer_c) = reshape(v_tmp, [Dims.level1n, n_a2_c, N_exp, 1, 1]);
                    Policy_fordouter(level1ii,:,:,z_c,e_c,d_outer_c) = reshape(p_tmp, [Dims.level1n, n_a2_c, N_exp, 1, 1]);

                    diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, 1, 1) - a1p_idx_L1(1, 1:end-1, :, :, 1, 1));
                    maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 4), [], 3)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(Dims.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), n_a1_c - 1), 0);
                        loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, 1, 1), n_a1_c - mg), [1, 1, 1, 1, n_a2_c, N_exp, 1, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;
                        a1p_grid_safe = reshape(a1_c_gridvals(a1p_idx), size(a1p_idx));

                        RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                            a1p_grid_safe, a2_c_gridvals, a1_c_gridvals(curraindex,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, n_a2_c, L, n_a2_c, N_exp, 1, 1]);

                        if has_cont
                            EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*n_a1_c*a2p_off_L2 + N_d2*n_a1_c*n_a2_c*exp_off_L2;
                            RHS_L2 = RM_L2 + df * EV_flat_bze(EV_idx);
                        else
                            RHS_L2 = RM_L2;
                        end

                        RHS_L2_flat = reshape(RHS_L2, [N_d_inner*(mg+1)*n_a2_c, L, n_a2_c, N_exp, 1, 1]);
                        [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                        dind = rem(maxindex2 - 1, N_d_inner) + 1;
                        a1p_a2p_idx = ceil(maxindex2 / N_d_inner);
                        a1p_offset = rem(a1p_a2p_idx - 1, mg+1) + 1;
                        a2p_ind = ceil(a1p_a2p_idx / (mg+1));

                        le_flat = reshape(loweredge, [1, 1, n_a2_c, N_exp, 1, 1]);
                        a1p_abs = le_flat + a1p_offset - 1;
                        pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * n_a1_c * (a2p_ind - 1);

                        V_fordouter(curraindex, :, :, z_c, e_c, d_outer_c) = reshape(v_tmp, [L, n_a2_c, N_exp, 1, 1]);
                        Policy_fordouter(curraindex, :, :, z_c, e_c, d_outer_c) = reshape(pol_flat, [L, n_a2_c, N_exp, 1, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_inner, N_d_outer);
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

    function [V_out, Pol_out] = extract_pol(V_douter, Policy_douter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_inner, N_d_outer)
        N_a_full = n_a1_c * n_a2_c * N_exp;
        V_douter_reshaped = reshape(V_douter, [N_a_full, N_bothz, N_e, N_d_outer]);
        Policy_douter_reshaped = reshape(Policy_douter, [N_a_full, N_bothz, N_e, N_d_outer]);

        if N_d_outer > 1
            [V_tmp, maxindex_douter] = max(V_douter_reshaped, [], 4);
            V_out = reshape(V_tmp, [N_a_full, N_bothz, N_e]);

            linear_douter = reshape(maxindex_douter, [N_a_full * N_bothz * N_e, 1]);
            lin_idx = (1:(N_a_full * N_bothz * N_e))' + (N_a_full * N_bothz * N_e) * (linear_douter - 1);
            inner_pol = reshape(Policy_douter_reshaped(lin_idx), [1, N_a_full, N_bothz, N_e]);

            d12_idx = rem(inner_pol - 1, N_d_inner) + 1;
            a12prime_idx = ceil(inner_pol / N_d_inner);
            douter_reshaped = reshape(maxindex_douter - 1, [1, N_a_full, N_bothz, N_e]);

            Pol_out = d12_idx + douter_reshaped * N_d_inner + (a12prime_idx - 1) * (N_d_inner * N_d_outer);
        else
            V_out = reshape(V_douter_reshaped, [N_a_full, N_bothz, N_e]);
            Pol_out = reshape(Policy_douter_reshaped, [1, N_a_full, N_bothz, N_e]);
        end
    end

    function EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims)
        EVinf = (EVpre_flat == -Inf);
        EV_val = EVpre_flat;
        EV_val(EVinf) = -1e250;
        EV_expected = EV_val * pi_bz';
        EV_expected((EVinf * pi_bz') > 0) = -Inf;

        num_z_interp = N_bothz * N_e_interp;

        if Dims.exp_flag == 6
            N_u = prod(vfoptions.n_u);
            EV_coarse_next = zeros(N_d2 * n_a1_c * n_a2_c * n_exp, N_bothz, 'gpuArray');
            for u_c = 1:N_u
                exp_idx = ExpAsset_J.a2primeIndex(:,:,u_c);
                exp_prb = ExpAsset_J.a2primeProbs(:,:,u_c);
                if size(exp_idx, 3) < N_bothz
                    exp_idx = repmat(exp_idx, [1, 1, N_bothz]);
                    exp_prb = repmat(exp_prb, [1, 1, N_bothz]);
                end
                EV_u = InterpolateExpAssetEV(EV_expected, n_exp, N_d2, n_a1_c * n_a2_c, n_exp, N_bothz, exp_idx, exp_prb);
                EV_coarse_next = EV_coarse_next + vfoptions.pi_u(u_c) * reshape(EV_u, [N_d2 * n_a1_c * n_a2_c * n_exp, N_bothz]);
            end
            EV_coarse = reshape(EV_coarse_next, [N_d2, n_a1_c, n_a2_c, n_exp, N_bothz, 1]);
        else
            EV_expected_rep = repmat(EV_expected, [1, N_e_interp]);

            exp_idx = ExpAsset_J.a2primeIndex;
            exp_prb = ExpAsset_J.a2primeProbs;

            if size(exp_idx, 3) < num_z_interp && size(exp_idx, 3) > 1
                exp_idx = repmat(exp_idx, [1, 1, num_z_interp / size(exp_idx, 3)]);
                exp_prb = repmat(exp_prb, [1, 1, num_z_interp / size(exp_prb, 3)]);
            end

            exp_idx_flat = reshape(exp_idx, [N_d2, n_exp, num_z_interp]);
            exp_prb_flat = reshape(exp_prb, [N_d2, n_exp, num_z_interp]);

            EV_coarse_flat = InterpolateExpAssetEV(EV_expected_rep, n_exp, N_d2, n_a1_c * n_a2_c, n_exp, num_z_interp, exp_idx_flat, exp_prb_flat);
            EV_coarse = reshape(EV_coarse_flat, [N_d2, n_a1_c, n_a2_c, n_exp, N_bothz, N_e_interp]);
        end
    end


end
