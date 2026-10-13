function [V, Policy] = ValueFnIter_UFHorz_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;

has_semiz = Dims.N_semiz_raw > 0;
has_e     = Dims.N_e_raw > 0;
has_z     = Dims.special_n_z(1) > 0;
has_bothz = has_semiz || has_z;

N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;

a_gridvals = gpuArray(a_grid);
level1ii = round(linspace(1, N_a, Dims.level1n));

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

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 3. Preallocate Outputs ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:N_a);

% --- 4. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]); else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    switch IterStrategy
        case 'Full'; [V(:,:,:,j), Pol_j] = solve_period_full(j, V_next);
        case 'Loop_e'; [V(:,:,:,j), Pol_j] = solve_period_loop_e(j, V_next);
        case 'Loop_z_parallel_semiz'; [V(:,:,:,j), Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next);
        case 'Loop_bothz'; [V(:,:,:,j), Pol_j] = solve_period_loop_bothz(j, V_next);
        case 'Loop_z_parallel_semiz_e'; [V(:,:,:,j), Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next);
        case 'Loop_bothz_e'; [V(:,:,:,j), Pol_j] = solve_period_loop_bothz_e(j, V_next);
    end
    Policy(:,:,:,:,j) = Pol_j;
end

% =====================================================================
% SPECIALIZED SOLVE FUNCTIONS
% =====================================================================
    function [V_j, Pol_j] = solve_period_full(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);
        bz_off_L2 = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, N_bothz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = Grids.d12_gridvals(:,:,d_outer_c); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_expected = df * compute_expected_value(EV, pi_bz, N_a, N_bothz);
                EV_expected_reshaped = reshape(EV_expected, [N_a, N_bothz]);
                DiscountedEV_L1 = reshape(EV_expected, [1, N_a, 1, N_bothz, 1]);
            end

            RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
            RM_L1 = reshape(RM_L1, [N_d_inner, N_a, Dims.level1n, N_bothz, N_e]);

            if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

            RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, Dims.level1n, N_bothz, N_e]);
            [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

            a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
            a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;

            V_fordouter(level1ii,:,:,d_outer_c) = reshape(v_tmp, [Dims.level1n, N_bothz, N_e]);
            Policy_fordouter(level1ii,:,:,d_outer_c) = reshape(maxindex1, [Dims.level1n, N_bothz, N_e]);

            diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :) - a1p_idx_L1(1, 1:end-1, :, :));
            maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 4), [], 3), [], 1)));
            if isempty(maxgap_vec); maxgap_vec = 0; end

            for ii = 1:(Dims.level1n-1)
                curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                L = length(curraindex);
                if L == 0; continue; end

                mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                loweredge = reshape(min(a1p_idx_L1(1, ii, :, :), N_a - mg), [1, 1, 1, N_bothz, N_e]);
                a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                le_flat = repmat(loweredge, [1, 1, L, 1, 1]);
                a1p_idx = le_flat + a1p_offsets;
                a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, N_bothz, N_e]);

                RM_L2 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, L, N_bothz, N_e]);

                if has_cont
                    EV_idx = a1p_idx + N_a * bz_off_L2;
                    EV_safe = EV_expected_reshaped(EV_idx);
                    RHS_L2 = RM_L2 + EV_safe;
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1), L, N_bothz, N_e]);
                [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                dind = rem(maxindex2 - 1, N_d_inner) + 1;
                a1p_offset = ceil(maxindex2 / N_d_inner);
                le_flat_clean = reshape(loweredge, [1, 1, N_bothz, N_e]);
                a1p_abs = le_flat_clean + a1p_offset - 1;
                pol_flat = dind + N_d_inner * (a1p_abs - 1);

                V_fordouter(curraindex, :, :, d_outer_c) = reshape(v_tmp, [L, N_bothz, N_e]);
                Policy_fordouter(curraindex, :, :, d_outer_c) = reshape(pol_flat, [L, N_bothz, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        bz_off_L2 = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, N_bothz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = Grids.d12_gridvals(:,:,d_outer_c); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_expected = df * compute_expected_value(EV, pi_bz, N_a, N_bothz);
                EV_expected_reshaped = reshape(EV_expected, [N_a, N_bothz]);
                DiscountedEV_L1 = reshape(EV_expected, [1, N_a, 1, N_bothz, 1]);
            end

            for e_c = 1:N_e
                e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a, Dims.level1n, N_bothz, 1]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, Dims.level1n, N_bothz, 1]);
                [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;

                V_fordouter(level1ii,:,e_c,d_outer_c) = reshape(v_tmp, [Dims.level1n, N_bothz, 1]);
                Policy_fordouter(level1ii,:,e_c,d_outer_c) = reshape(maxindex1, [Dims.level1n, N_bothz, 1]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, 1) - a1p_idx_L1(1, 1:end-1, :, 1));
                maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 3), [], 1)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(Dims.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, 1), N_a - mg), [1, 1, 1, N_bothz, 1]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                    le_flat = repmat(loweredge, [1, 1, L, 1, 1]);
                    a1p_idx = le_flat + a1p_offsets;
                    a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, N_bothz, 1]);

                    RM_L2 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, L, N_bothz, 1]);

                    if has_cont
                        EV_idx = a1p_idx + N_a * bz_off_L2;
                        EV_safe = reshape(EV_expected_reshaped(EV_idx), size(a1p_idx));
                        RHS_L2 = RM_L2 + EV_safe;
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1), L, N_bothz, 1]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    le_flat_clean = reshape(loweredge, [1, 1, N_bothz, 1]);
                    a1p_abs = le_flat_clean + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1);

                    V_fordouter(curraindex, :, e_c, d_outer_c) = reshape(v_tmp, [L, N_bothz, 1]);
                    Policy_fordouter(curraindex, :, e_c, d_outer_c) = reshape(pol_flat, [L, N_bothz, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = Grids.d12_gridvals(:,:,d_outer_c); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                if has_cont
                    EV_expected = df * compute_expected_value_cond(EV, pi_bz(z_c,:)', N_a);
                    DiscountedEV_L1 = reshape(EV_expected, [1, N_a, 1, 1, 1]);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a, Dims.level1n, 1, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, Dims.level1n, 1, N_e]);
                [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;

                V_fordouter(level1ii,z_c,:,d_outer_c) = reshape(v_tmp, [Dims.level1n, 1, N_e]);
                Policy_fordouter(level1ii,z_c,:,d_outer_c) = reshape(maxindex1, [Dims.level1n, 1, N_e]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, 1, :) - a1p_idx_L1(1, 1:end-1, 1, :));
                maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 4), [], 1)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(Dims.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, 1, :), N_a - mg), [1, 1, 1, 1, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                    le_flat = repmat(loweredge, [1, 1, L, 1, 1]);
                    a1p_idx = le_flat + a1p_offsets;
                    a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, 1, N_e]);

                    RM_L2 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, L, 1, N_e]);

                    if has_cont
                        EV_idx = a1p_idx;
                        EV_safe = reshape(EV_expected(EV_idx), size(a1p_idx));
                        RHS_L2 = RM_L2 + EV_safe;
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1), L, 1, N_e]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    le_flat_clean = reshape(loweredge, [1, 1, 1, N_e]);
                    a1p_abs = le_flat_clean + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1);

                    V_fordouter(curraindex, z_c, :, d_outer_c) = reshape(v_tmp, [L, 1, N_e]);
                    Policy_fordouter(curraindex, z_c, :, d_outer_c) = reshape(pol_flat, [L, 1, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);
        sb_off_L2 = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, N_semiz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = Grids.d12_gridvals(:,:,d_outer_c); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                if has_cont
                    EV_expected = df * compute_expected_value_cond(EV, pi_bz(sb,:)', N_a);
                    EV_expected_reshaped = reshape(EV_expected, [N_a, N_semiz]);
                    DiscountedEV_L1 = reshape(EV_expected, [1, N_a, 1, N_semiz, 1]);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a, Dims.level1n, N_semiz, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, Dims.level1n, N_semiz, N_e]);
                [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;

                V_fordouter(level1ii,sb,:,d_outer_c) = reshape(v_tmp, [Dims.level1n, N_semiz, N_e]);
                Policy_fordouter(level1ii,sb,:,d_outer_c) = reshape(maxindex1, [Dims.level1n, N_semiz, N_e]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :) - a1p_idx_L1(1, 1:end-1, :, :));
                maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 4), [], 3), [], 1)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(Dims.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, :), N_a - mg), [1, 1, 1, N_semiz, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                    le_flat = repmat(loweredge, [1, L, 1, 1]);
                    a1p_idx = le_flat + a1p_offsets;
                    a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, N_semiz, N_e]);

                    RM_L2 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, L, N_semiz, N_e]);

                    if has_cont
                        EV_idx = a1p_idx + N_a * sb_off_L2;
                        EV_safe = reshape(EV_expected_reshaped(EV_idx), size(a1p_idx));
                        RHS_L2 = RM_L2 + EV_safe;
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1), L, N_semiz, N_e]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    le_flat_clean = reshape(loweredge, [1, 1, N_semiz, N_e]);
                    a1p_abs = le_flat_clean + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1);

                    V_fordouter(curraindex, sb, :, d_outer_c) = reshape(v_tmp, [L, N_semiz, N_e]);
                    Policy_fordouter(curraindex, sb, :, d_outer_c) = reshape(pol_flat, [L, N_semiz, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        sb_off_L2 = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, N_semiz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = Grids.d12_gridvals(:,:,d_outer_c); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                if has_cont
                    EV_expected = df * compute_expected_value_cond(EV, pi_bz(sb,:)', N_a);
                    EV_expected_reshaped = reshape(EV_expected, [N_a, N_semiz]);
                    DiscountedEV_L1 = reshape(EV_expected, [1, N_a, 1, N_semiz, 1]);
                end

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d_inner, N_a, Dims.level1n, N_semiz, 1]);

                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, Dims.level1n, N_semiz, 1]);
                    [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;

                    V_fordouter(level1ii,sb,e_c,d_outer_c) = reshape(v_tmp, [Dims.level1n, N_semiz, 1]);
                    Policy_fordouter(level1ii,sb,e_c,d_outer_c) = reshape(maxindex1, [Dims.level1n, N_semiz, 1]);

                    diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, 1) - a1p_idx_L1(1, 1:end-1, :, 1));
                    maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 3), [], 1)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(Dims.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                        loweredge = reshape(min(a1p_idx_L1(1, ii, :, 1), N_a - mg), [1, 1, 1, N_semiz, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                        le_flat = repmat(loweredge, [1, L, 1, 1]);
                        a1p_idx = le_flat + a1p_offsets;
                        a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, N_semiz, 1]);

                        RM_L2 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, L, N_semiz, 1]);

                        if has_cont
                            EV_idx = a1p_idx + N_a * sb_off_L2;
                            EV_safe = reshape(EV_expected_reshaped(EV_idx), size(a1p_idx));
                            RHS_L2 = RM_L2 + EV_safe;
                        else
                            RHS_L2 = RM_L2;
                        end

                        RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1), L, N_semiz, 1]);
                        [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                        dind = rem(maxindex2 - 1, N_d_inner) + 1;
                        a1p_offset = ceil(maxindex2 / N_d_inner);
                        le_flat_clean = reshape(loweredge, [1, 1, N_semiz, 1]);
                        a1p_abs = le_flat_clean + a1p_offset - 1;
                        pol_flat = dind + N_d_inner * (a1p_abs - 1);

                        V_fordouter(curraindex, sb, e_c, d_outer_c) = reshape(v_tmp, [L, N_semiz, 1]);
                        Policy_fordouter(curraindex, sb, e_c, d_outer_c) = reshape(pol_flat, [L, N_semiz, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = Grids.d12_gridvals(:,:,d_outer_c); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                if has_cont
                    EV_expected = df * compute_expected_value_cond(EV, pi_bz(z_c,:)', N_a);
                    DiscountedEV_L1 = reshape(EV_expected, [1, N_a, 1, 1, 1]);
                end

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d_inner, N_a, Dims.level1n, 1, 1]);

                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, Dims.level1n, 1, 1]);
                    [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;

                    V_fordouter(level1ii,z_c,e_c,d_outer_c) = reshape(v_tmp, [Dims.level1n, 1, 1]);
                    Policy_fordouter(level1ii,z_c,e_c,d_outer_c) = reshape(maxindex1, [Dims.level1n, 1, 1]);

                    diff_a1p = max(0, a1p_idx_L1(1, 2:end, 1, 1) - a1p_idx_L1(1, 1:end-1, 1, 1));
                    maxgap_vec = gather(squeeze(max(diff_a1p, [], 2)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(Dims.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                        loweredge = reshape(min(a1p_idx_L1(1, ii, 1, 1), N_a - mg), [1, 1, 1, 1, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                        le_flat = repmat(loweredge, [1, L, 1, 1]);
                        a1p_idx = le_flat + a1p_offsets;
                        a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, 1, 1]);

                        RM_L2 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, L, 1, 1]);

                        if has_cont
                            EV_idx = a1p_idx;
                            EV_safe = reshape(EV_expected(EV_idx), size(a1p_idx));
                            RHS_L2 = RM_L2 + EV_safe;
                        else
                            RHS_L2 = RM_L2;
                        end

                        RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1), L, 1, 1]);
                        [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                        dind = rem(maxindex2 - 1, N_d_inner) + 1;
                        a1p_offset = ceil(maxindex2 / N_d_inner);
                        le_flat_clean = reshape(loweredge, [1, 1, 1, 1]);
                        a1p_abs = le_flat_clean + a1p_offset - 1;
                        pol_flat = dind + N_d_inner * (a1p_abs - 1);

                        V_fordouter(curraindex, z_c, e_c, d_outer_c) = reshape(v_tmp, [L, 1, 1]);
                        Policy_fordouter(curraindex, z_c, e_c, d_outer_c) = reshape(pol_flat, [L, 1, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a, N_bothz, N_e, N_d_inner, N_d_outer);
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

    function [V_out, Pol_out] = extract_pol(V_douter, P_douter, N_a, N_bothz, N_e, N_d_inner, N_d_outer)
        if N_d_outer > 1
            [V_out, maxidx] = max(V_douter, [], 5);
            lin_idx = (1:(N_a*N_bothz*N_e))' + (N_a*N_bothz*N_e)*(maxidx(:) - 1);
            P_douter_flat = reshape(P_douter, [N_a*N_bothz*N_e*N_d_outer, 1]);
            raw_pol = P_douter_flat(lin_idx);

            d_inner_idx = rem(raw_pol - 1, N_d_inner) + 1;
            a1p_idx = ceil(raw_pol / (N_d_inner * N_d_outer));
            d_outer_idx = maxidx(:);
            linear_combined = d_inner_idx + N_d_inner * (d_outer_idx - 1) + (N_d_inner * N_d_outer) * (a1p_idx - 1);
            Pol_out = reshape(linear_combined, [1, N_a, N_bothz, N_e]);
        else
            V_out = reshape(V_douter, [N_a, N_bothz, N_e]);
            Pol_out = reshape(P_douter, [1, N_a, N_bothz, N_e]);
        end
    end

    function EV_out = compute_expected_value(EV, pi_bothz, N_a, N_bothz)
        EVinf = (EV == -Inf);
        EV_val = EV;
        EV_val(EVinf) = -1e250;
        EV_val = EV_val * pi_bothz';
        EV_val(EVinf * (pi_bothz' > 0) > 0) = -Inf;
        EV_out = reshape(EV_val, [N_a, 1, N_bothz]);
    end

    function EV_out = compute_expected_value_cond(EV, pi_slice, N_a)
        EV_val = EV .* shiftdim(pi_slice, -1);
        EV_val(isnan(EV_val)) = 0;
        EV_out = sum(EV_val, 2);
    end


end
