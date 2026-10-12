function [V, Policy] = ValueFnIter_UFHorz_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;

N_semiz_raw = Dims.N_semiz_raw > 0;
N_e_raw     = Dims.N_e_raw > 0;
N_z_raw     = Dims.special_n_z(1) > 0;
has_bothz   = N_semiz_raw || N_z_raw;

N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
n_d1 = Dims.n_d1; n_d2 = Dims.n_d2;
N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;

N_a1 = n_a(1);
N_a2 = max(prod(n_a(2:end)), 1);
N_a = N_a1 * N_a2;

a1_gridvals = gpuArray(a_grid(1:N_a1));
a2_gridvals = gpuArray(a_grid(N_a1+1:end));

if ~isfield(vfoptions, 'level1n'); vfoptions.level1n = min(floor(sqrt(N_a1)), N_a1); end
level1ii = round(linspace(1, N_a1, vfoptions.level1n));

% --- 2. Determine Iteration Strategy ---
if vfoptions.lowmemory == 0
    IterStrategy = 'Full';
elseif vfoptions.lowmemory == 1
    if N_e_raw; IterStrategy = 'Loop_e';
    elseif N_semiz_raw && N_z_raw; IterStrategy = 'Loop_z_parallel_semiz';
    else; IterStrategy = 'Loop_bothz'; end
elseif vfoptions.lowmemory == 2
    if N_e_raw && N_semiz_raw && N_z_raw; IterStrategy = 'Loop_z_parallel_semiz_e';
    elseif N_e_raw; IterStrategy = 'Loop_bothz_e';
    elseif N_semiz_raw && N_z_raw; IterStrategy = 'Loop_bothz';
    else; error('Invalid lowmemory=2 for this shock combination.'); end
elseif vfoptions.lowmemory == 3
    if N_e_raw && N_semiz_raw && N_z_raw; IterStrategy = 'Loop_bothz_e';
    else; error('Invalid lowmemory=3 for this shock combination.'); end
end

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 3. Preallocate Outputs ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:N_a1);

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
% SPECIALIZED SOLVE FUNCTIONS (Vectorised over N_d_inner and N_a2_choice)
% =====================================================================
    function [V_j, Pol_j] = solve_period_full(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1, 1]);
        bz_off_L2  = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, N_bothz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_expected = df * (EVpre_flat * pi_bz');
                EV_expected_reshaped = reshape(EV_expected, [N_a1, N_a2, N_bothz, 1]);
                DiscountedEV_L1 = reshape(EV_expected, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
            end

            RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
            RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, vfoptions.level1n, N_a2, N_bothz, N_e]);

            if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

            [~, maxindex1] = max(RHS_L1, [], 2);
            [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, vfoptions.level1n, N_a2, N_bothz, N_e]), [], 1);
            V_fordouter(level1ii,:,:,:,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, N_a2, N_bothz, N_e]);
            Policy_fordouter(level1ii,:,:,:,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, N_a2, N_bothz, N_e]);

            min_a1prime = min(min(maxindex1, [], 1), [], 3);
            max_a1prime = max(max(maxindex1, [], 1), [], 3);
            diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,:) - min_a1prime(1,1,1,1:end-1,:,:,:));
            maxgap_vec = gather(squeeze(max(max(max(max(diff_a1p, [], 7), [], 6), [], 5), [], 1)));
            if isempty(maxgap_vec); maxgap_vec = 0; end

            for ii = 1:(vfoptions.level1n-1)
                curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                L = length(curraindex);
                if L == 0; continue; end

                mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                loweredge = reshape(min(min_a1prime(1, 1, 1, ii, :, :, :), N_a1 - mg), [1, 1, 1, 1, N_a2, N_bothz, N_e]);
                a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                a1p_idx = loweredge + a1p_offsets;
                a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals(curraindex,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);

                if has_cont
                    EV_idx = a1p_idx + N_a1 * a2p_off_L2 + N_a1 * N_a2 * bz_off_L2;
                    EV_safe = EV_expected_reshaped(EV_idx);
                    RHS_L2 = RM_L2 + EV_safe;
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1) * N_a2, L, N_a2, N_bothz, N_e]);
                [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                dind = rem(maxindex2 - 1, N_d_inner) + 1;
                a1p_offset = rem(ceil(maxindex2 / N_d_inner) - 1, mg+1) + 1;
                a2p_ind = ceil(maxindex2 / (N_d_inner * (mg+1)));

                le_flat = reshape(loweredge, [1, 1, N_a2, N_bothz, N_e]);
                a1p_abs = le_flat + a1p_offset - 1;
                pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * N_a1 * (a2p_ind - 1);

                V_fordouter(curraindex, :, :, :, d_outer_c) = reshape(v_tmp, [L, N_a2, N_bothz, N_e]);
                Policy_fordouter(curraindex, :, :, :, d_outer_c) = reshape(pol_flat, [L, N_a2, N_bothz, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1, 1]);
        bz_off_L2  = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, N_bothz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_expected = df * (EVpre_flat * pi_bz');
                EV_expected_reshaped = reshape(EV_expected, [N_a1, N_a2, N_bothz, 1]);
                DiscountedEV_L1 = reshape(EV_expected, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
            end

            for e_c = 1:N_e
                e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);

                RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, vfoptions.level1n, N_a2, N_bothz, 1]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                [~, maxindex1] = max(RHS_L1, [], 2);
                [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, vfoptions.level1n, N_a2, N_bothz]), [], 1);
                V_fordouter(level1ii,:,:,e_c,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, N_a2, N_bothz, 1]);
                Policy_fordouter(level1ii,:,:,e_c,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, N_a2, N_bothz, 1]);

                min_a1prime = min(min(maxindex1, [], 1), [], 3);
                max_a1prime = max(max(maxindex1, [], 1), [], 3);
                diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,1) - min_a1prime(1,1,1,1:end-1,:,:,1));
                maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 6), [], 5), [], 1)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(vfoptions.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                    loweredge = reshape(min(min_a1prime(1, 1, 1, ii, :, :, 1), N_a1 - mg), [1, 1, 1, 1, N_a2, N_bothz, 1]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                    a1p_idx = loweredge + a1p_offsets;
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals(curraindex,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);

                    if has_cont
                        EV_idx = a1p_idx + N_a1 * a2p_off_L2 + N_a1 * N_a2 * bz_off_L2;
                        EV_safe = EV_expected_reshaped(EV_idx);
                        RHS_L2 = RM_L2 + EV_safe;
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1) * N_a2, L, N_a2, N_bothz, 1]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = rem(ceil(maxindex2 / N_d_inner) - 1, mg+1) + 1;
                    a2p_ind = ceil(maxindex2 / (N_d_inner * (mg+1)));

                    le_flat = reshape(loweredge, [1, 1, N_a2, N_bothz, 1]);
                    a1p_abs = le_flat + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * N_a1 * (a2p_ind - 1);

                    V_fordouter(curraindex, :, :, e_c, d_outer_c) = reshape(v_tmp, [L, N_a2, N_bothz, 1]);
                    Policy_fordouter(curraindex, :, :, e_c, d_outer_c) = reshape(pol_flat, [L, N_a2, N_bothz, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);
        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1, 1]);
        sb_off_L2  = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, N_semiz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_expected = df * (EVpre_flat * pi_bz');
                EV_expected_reshaped = reshape(EV_expected, [N_a1, N_a2, N_semiz, N_z]);
                DiscountedEV_pre = reshape(EV_expected, [1, N_a1, N_a2, 1, 1, N_semiz, N_z]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                if has_cont
                    DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,:,:,z_c);
                    EV_safe_z = reshape(EV_expected_reshaped(:,:,:,z_c), [N_a1, N_a2, N_semiz, 1]);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, vfoptions.level1n, N_a2, N_semiz, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                [~, maxindex1] = max(RHS_L1, [], 2);
                [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, vfoptions.level1n, N_a2, N_semiz, N_e]), [], 1);
                V_fordouter(level1ii,:,sb,:,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, N_a2, N_semiz, N_e]);
                Policy_fordouter(level1ii,:,sb,:,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, N_a2, N_semiz, N_e]);

                min_a1prime = min(min(maxindex1, [], 1), [], 3);
                max_a1prime = max(max(maxindex1, [], 1), [], 3);
                diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,:) - min_a1prime(1,1,1,1:end-1,:,:,:));
                maxgap_vec = gather(squeeze(max(max(max(max(diff_a1p, [], 7), [], 6), [], 5), [], 1)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(vfoptions.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                    loweredge = reshape(min(min_a1prime(1, 1, 1, ii, :, :, :), N_a1 - mg), [1, 1, 1, 1, N_a2, N_semiz, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                    a1p_idx = loweredge + a1p_offsets;
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals(curraindex,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);

                    if has_cont
                        EV_idx = a1p_idx + N_a1 * a2p_off_L2 + N_a1 * N_a2 * sb_off_L2;
                        EV_safe = EV_safe_z(EV_idx);
                        RHS_L2 = RM_L2 + EV_safe;
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1) * N_a2, L, N_a2, N_semiz, N_e]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = rem(ceil(maxindex2 / N_d_inner) - 1, mg+1) + 1;
                    a2p_ind = ceil(maxindex2 / (N_d_inner * (mg+1)));

                    le_flat = reshape(loweredge, [1, 1, N_a2, N_semiz, N_e]);
                    a1p_abs = le_flat + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * N_a1 * (a2p_ind - 1);

                    V_fordouter(curraindex, :, sb, :, d_outer_c) = reshape(v_tmp, [L, N_a2, N_semiz, N_e]);
                    Policy_fordouter(curraindex, :, sb, :, d_outer_c) = reshape(pol_flat, [L, N_a2, N_semiz, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);
        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_expected = df * (EVpre_flat * pi_bz');
                EV_expected_reshaped = reshape(EV_expected, [N_a1, N_a2, N_bothz, 1]);
                DiscountedEV_pre = reshape(EV_expected, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                if has_cont
                    DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,:,z_c,1);
                    EV_safe_z = reshape(EV_expected_reshaped(:,:,z_c,1), [N_a1, N_a2, 1, 1]);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, vfoptions.level1n, N_a2, 1, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                [~, maxindex1] = max(RHS_L1, [], 2);
                [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, vfoptions.level1n, N_a2, 1, N_e]), [], 1);
                V_fordouter(level1ii,:,z_c,:,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, N_a2, 1, N_e]);
                Policy_fordouter(level1ii,:,z_c,:,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, N_a2, 1, N_e]);

                min_a1prime = min(min(maxindex1, [], 1), [], 3);
                max_a1prime = max(max(maxindex1, [], 1), [], 3);
                diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,:) - min_a1prime(1,1,1,1:end-1,:,:,:));
                maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 7), [], 6), [], 5)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(vfoptions.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                    loweredge = reshape(min(min_a1prime(1, 1, 1, ii, :, :, :), N_a1 - mg), [1, 1, 1, 1, N_a2, 1, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                    a1p_idx = loweredge + a1p_offsets;
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals(curraindex,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);

                    if has_cont
                        EV_idx = a1p_idx + N_a1 * a2p_off_L2;
                        EV_safe = EV_safe_z(EV_idx);
                        RHS_L2 = RM_L2 + EV_safe;
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1) * N_a2, L, N_a2, 1, N_e]);
                    [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                    dind = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = rem(ceil(maxindex2 / N_d_inner) - 1, mg+1) + 1;
                    a2p_ind = ceil(maxindex2 / (N_d_inner * (mg+1)));

                    le_flat = reshape(loweredge, [1, 1, N_a2, 1, N_e]);
                    a1p_abs = le_flat + a1p_offset - 1;
                    pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * N_a1 * (a2p_ind - 1);

                    V_fordouter(curraindex, :, z_c, :, d_outer_c) = reshape(v_tmp, [L, N_a2, 1, N_e]);
                    Policy_fordouter(curraindex, :, z_c, :, d_outer_c) = reshape(pol_flat, [L, N_a2, 1, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1, 1]);
        sb_off_L2  = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, N_semiz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_expected = df * (EVpre_flat * pi_bz');
                EV_expected_reshaped = reshape(EV_expected, [N_a1, N_a2, N_semiz, N_z]);
                DiscountedEV_pre = reshape(EV_expected, [1, N_a1, N_a2, 1, 1, N_semiz, N_z]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);

                    if has_cont
                        DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,:,:,z_c);
                        EV_safe_z = reshape(EV_expected_reshaped(:,:,:,z_c), [N_a1, N_a2, N_semiz, 1]);
                    end

                    RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                    RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, vfoptions.level1n, N_a2, N_semiz, 1]);

                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    [~, maxindex1] = max(RHS_L1, [], 2);
                    [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, vfoptions.level1n, N_a2, N_semiz]), [], 1);
                    V_fordouter(level1ii,:,sb,e_c,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, N_a2, N_semiz, 1]);
                    Policy_fordouter(level1ii,:,sb,e_c,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, N_a2, N_semiz, 1]);

                    min_a1prime = min(min(maxindex1, [], 1), [], 3);
                    max_a1prime = max(max(maxindex1, [], 1), [], 3);
                    diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,1) - min_a1prime(1,1,1,1:end-1,:,:,1));
                    maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 6), [], 5), [], 1)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                        loweredge = reshape(min(min_a1prime(1, 1, 1, ii, :, :, 1), N_a1 - mg), [1, 1, 1, 1, N_a2, N_semiz, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;
                        a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                        RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals(curraindex,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);

                        if has_cont
                            EV_idx = a1p_idx + N_a1 * a2p_off_L2 + N_a1 * N_a2 * sb_off_L2;
                            EV_safe = EV_safe_z(EV_idx);
                            RHS_L2 = RM_L2 + EV_safe;
                        else
                            RHS_L2 = RM_L2;
                        end

                        RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1) * N_a2, L, N_a2, N_semiz, 1]);
                        [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                        dind = rem(maxindex2 - 1, N_d_inner) + 1;
                        a1p_offset = rem(ceil(maxindex2 / N_d_inner) - 1, mg+1) + 1;
                        a2p_ind = ceil(maxindex2 / (N_d_inner * (mg+1)));

                        le_flat = reshape(loweredge, [1, 1, N_a2, N_semiz, 1]);
                        a1p_abs = le_flat + a1p_offset - 1;
                        pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * N_a1 * (a2p_ind - 1);

                        V_fordouter(curraindex, :, sb, e_c, d_outer_c) = reshape(v_tmp, [L, N_a2, N_semiz, 1]);
                        Policy_fordouter(curraindex, :, sb, e_c, d_outer_c) = reshape(pol_flat, [L, N_a2, N_semiz, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        Policy_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_expected = df * (EVpre_flat * pi_bz');
                EV_expected_reshaped = reshape(EV_expected, [N_a1, N_a2, N_bothz, 1]);
                DiscountedEV_pre = reshape(EV_expected, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);

                    if has_cont
                        DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,:,z_c,1);
                        EV_safe_z = reshape(EV_expected_reshaped(:,:,z_c,1), [N_a1, N_a2, 1, 1]);
                    end

                    RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                    RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, vfoptions.level1n, N_a2, 1, 1]);

                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    [~, maxindex1] = max(RHS_L1, [], 2);
                    [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, vfoptions.level1n, N_a2, 1]), [], 1);
                    V_fordouter(level1ii,:,z_c,e_c,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, N_a2, 1, 1]);
                    Policy_fordouter(level1ii,:,z_c,e_c,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, N_a2, 1, 1]);

                    min_a1prime = min(min(maxindex1, [], 1), [], 3);
                    max_a1prime = max(max(maxindex1, [], 1), [], 3);
                    diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,1) - min_a1prime(1,1,1,1:end-1,:,:,1));
                    maxgap_vec = gather(squeeze(max(diff_a1p, [], 4)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                        loweredge = reshape(min(min_a1prime(1, 1, 1, ii, :, :, 1), N_a1 - mg), [1, 1, 1, 1, N_a2, 1, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;
                        a1p_grid_safe = reshape(a_gridvals(a1p_idx), size(a1p_idx));

                        RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals(curraindex,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);

                        if has_cont
                            EV_idx = a1p_idx + N_a1 * a2p_off_L2;
                            EV_safe = EV_safe_z(EV_idx);
                            RHS_L2 = RM_L2 + EV_safe;
                        else
                            RHS_L2 = RM_L2;
                        end

                        RHS_L2_flat = reshape(RHS_L2, [N_d_inner * (mg+1) * N_a2, L, N_a2, 1, 1]);
                        [v_tmp, maxindex2] = max(RHS_L2_flat, [], 1);

                        dind = rem(maxindex2 - 1, N_d_inner) + 1;
                        a1p_offset = rem(ceil(maxindex2 / N_d_inner) - 1, mg+1) + 1;
                        a2p_ind = ceil(maxindex2 / (N_d_inner * (mg+1)));

                        le_flat = reshape(loweredge, [1, 1, N_a2, 1, 1]);
                        a1p_abs = le_flat + a1p_offset - 1;
                        pol_flat = dind + N_d_inner * (a1p_abs - 1) + N_d_inner * N_a1 * (a2p_ind - 1);

                        V_fordouter(curraindex, :, z_c, e_c, d_outer_c) = reshape(v_tmp, [L, N_a2, 1, 1]);
                        Policy_fordouter(curraindex, :, z_c, e_c, d_outer_c) = reshape(pol_flat, [L, N_a2, 1, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Policy_fordouter, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer);
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

    function [V_out, Pol_out] = extract_pol(V_douter, Policy_douter, N_a1, N_a2, N_bothz, N_e, N_d_inner, N_d_outer)
        if N_d_outer > 1
            [V_tmp, maxindex_douter] = max(V_douter, [], 5);
            V_out = reshape(V_tmp, [N_a1 * N_a2, N_bothz, N_e]);

            linear_douter = reshape(maxindex_douter, [N_a1 * N_a2 * N_bothz * N_e, 1]);
            lin_idx = (1:(N_a1 * N_a2 * N_bothz * N_e))' + (N_a1 * N_a2 * N_bothz * N_e) * (linear_douter - 1);
            inner_pol = reshape(Policy_douter(lin_idx), [1, N_a1 * N_a2, N_bothz, N_e]);

            d12_idx = rem(inner_pol - 1, N_d_inner) + 1;
            a12prime_idx = ceil(inner_pol / N_d_inner);
            douter_reshaped = reshape(maxindex_douter - 1, [1, N_a1 * N_a2, N_bothz, N_e]);

            Pol_out = d12_idx + douter_reshaped * N_d_inner + (a12prime_idx - 1) * (N_d_inner * N_d_outer);
        else
            V_out = reshape(V_douter, [N_a1 * N_a2, N_bothz, N_e]);
            Pol_out = reshape(Policy_douter, [1, N_a1 * N_a2, N_bothz, N_e]);
        end
    end


end
