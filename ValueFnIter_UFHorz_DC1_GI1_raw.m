function [V, Policy] = ValueFnIter_UFHorz_DC1_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_semiz_raw = Dims.N_semiz_raw > 0;
N_e_raw     = Dims.N_e_raw > 0;
N_z_raw     = Dims.special_n_z(1) > 0;
has_bothz   = N_semiz_raw || N_z_raw;

N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
n_d1 = Dims.n_d1; n_d2 = Dims.n_d2;
N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;

a_gridvals = gpuArray(a_grid);

n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
a1prime_grid = gpuArray(interp1(1:1:N_a, gather(a_gridvals), linspace(1, N_a, N_a+(N_a-1)*n2short))');
N_a1fine = length(a1prime_grid);

if ~isfield(vfoptions, 'level1n'); vfoptions.level1n = min(floor(sqrt(N_a)), N_a); end
level1ii = round(linspace(1, N_a, vfoptions.level1n));

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
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy(4,:,:,:,:) = 2;

n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [n2long, 1, 1, 1]);
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
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);

        bz_off_L2  = reshape(gpuArray(0:N_bothz-1), [1, 1, N_bothz, 1]);
        a1_off_flat = reshape(gpuArray(0:N_a-1), [1, N_a, 1, 1]);
        bz_off_flat = reshape(gpuArray(0:N_bothz-1), [1, 1, N_bothz, 1]);
        e_off_flat  = reshape(gpuArray(0:N_e-1), [1, 1, 1, N_e]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end
            midpt_local = zeros(1, N_a, N_bothz, N_e, 'gpuArray');

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_bothz, a_gridvals, a1prime_grid);
                EV_expected = df * compute_expected_value(EVpre_flat, pi_bz, N_a, N_bothz);
                EV_expected_reshaped = reshape(EV_expected, [N_a, N_bothz]);
                DiscountedEV_L1 = reshape(EV_expected, [1, N_a, 1, N_bothz, 1]);
            end

            RM_L1 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
            RM_L1 = reshape(RM_L1, [N_d_inner, N_a, vfoptions.level1n, N_bothz, N_e]);

            if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

            RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, vfoptions.level1n, N_bothz, N_e]);
            [~, maxindex1] = max(RHS_L1_flat, [], 1);

            a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
            a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;
            midpt_local(1, level1ii, :, :) = a1p_idx_L1;

            diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :) - a1p_idx_L1(1, 1:end-1, :, :));
            maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 4), [], 3), [], 1)));
            if isempty(maxgap_vec); maxgap_vec = 0; end

            for ii = 1:(vfoptions.level1n-1)
                curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                L = length(curraindex);
                if L == 0; continue; end

                mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                loweredge = reshape(min(a1p_idx_L1(1, ii, :, :), N_a - mg), [1, 1, N_bothz, N_e]);
                a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [mg+1, 1, 1, 1]);
                le_flat = repmat(loweredge, [1, L, 1, 1]);
                a1p_idx = le_flat + a1p_offsets;

                a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, N_bothz, N_e]);

                RM_L2_DC = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                    a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_bothz, N_e]);

                if has_cont
                    EV_idx = a1p_idx + N_a * bz_off_L2;
                    RHS_L2_DC = RM_L2_DC + reshape(EV_expected_reshaped(EV_idx), [1, mg+1, L, N_bothz, N_e]);
                else
                    RHS_L2_DC = RM_L2_DC;
                end

                RHS_L2_flat_DC = reshape(RHS_L2_DC, [N_d_inner*(mg+1), L, N_bothz, N_e]);
                [~, maxindex2] = max(RHS_L2_flat_DC, [], 1);

                a1p_offset = ceil(maxindex2 / N_d_inner);
                midpt_local(1, curraindex, :, :) = le_flat(1,:,:,:) + a1p_offset - 1;
            end

            midpoint = max(min(midpt_local, N_a-1), 2);
            a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
            a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, N_a, N_bothz, N_e]);

            RM_GI = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
            RM_GI = reshape(RM_GI, [N_d_inner, n2long, N_a, N_bothz, N_e]);

            if has_cont
                EV_idx_GI = a1primeindexes + N_a1fine * bz_off_L2;
                RHS_GI = RM_GI + df * reshape(EV_interp(EV_idx_GI), [1, n2long, N_a, N_bothz, N_e]);
            else
                RHS_GI = RM_GI;
            end

            RHS_GI_flat = reshape(RHS_GI, [N_d_inner * n2long, N_a, N_bothz, N_e]);
            [v_tmp, p_tmp] = max(RHS_GI_flat, [], 1);

            d_ind = rem(p_tmp - 1, N_d_inner) + 1;
            L2_and_a2p = ceil(p_tmp / N_d_inner);
            L2offset = rem(L2_and_a2p - 1, n2long) + 1;

            linidx_lower = d_ind + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*bz_off_flat + N_d_inner*n2long*N_a*N_bothz*e_off_flat;
            linidx_upper = d_ind + N_d_inner*(n2long-1) + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*bz_off_flat + N_d_inner*n2long*N_a*N_bothz*e_off_flat;

            isInfLower = (RM_GI(linidx_lower) == -Inf);
            isInfUpper = (RM_GI(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

            V_fordouter(:,:,:,d_outer_c) = reshape(v_tmp, [N_a, N_bothz, N_e]);
            P1_fordouter(:,:,:,d_outer_c) = reshape(d_idx_val, [N_a, N_bothz, N_e]);
            P2_fordouter(:,:,:,d_outer_c) = reshape(midpoint, [N_a, N_bothz, N_e]);
            P3_fordouter(:,:,:,d_outer_c) = reshape(L2offset, [N_a, N_bothz, N_e]);
            P4_fordouter(:,:,:,d_outer_c) = reshape(flag_val, [N_a, N_bothz, N_e]);
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);

        bz_off_L2  = reshape(gpuArray(0:N_bothz-1), [1, 1, N_bothz]);
        a1_off_flat = reshape(gpuArray(0:N_a-1), [1, N_a, 1]);
        bz_off_flat = reshape(gpuArray(0:N_bothz-1), [1, 1, N_bothz]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_bothz, a_gridvals, a1prime_grid);
                EV_expected = df * compute_expected_value(EVpre_flat, pi_bz, N_a, N_bothz);
                EV_expected_reshaped = reshape(EV_expected, [N_a, N_bothz]);
                DiscountedEV_L1 = reshape(EV_expected, [1, N_a, 1, N_bothz]);
            end

            for e_c = 1:N_e
                e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);
                midpt_local = zeros(1, N_a, N_bothz, 1, 'gpuArray');

                RM_L1 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a, vfoptions.level1n, N_bothz]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, vfoptions.level1n, N_bothz]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;
                midpt_local(1, level1ii, :) = a1p_idx_L1;

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :) - a1p_idx_L1(1, 1:end-1, :));
                maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 3), [], 1)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(vfoptions.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :), N_a - mg), [1, 1, N_bothz]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [mg+1, 1, 1]);
                    le_flat = repmat(loweredge, [1, L, 1]);
                    a1p_idx = le_flat + a1p_offsets;

                    a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, N_bothz]);

                    RM_L2_DC = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                        a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_bothz]);

                    if has_cont
                        EV_idx = a1p_idx + N_a * bz_off_L2;
                        RHS_L2_DC = RM_L2_DC + reshape(EV_expected_reshaped(EV_idx), [1, mg+1, L, N_bothz]);
                    else
                        RHS_L2_DC = RM_L2_DC;
                    end

                    RHS_L2_flat_DC = reshape(RHS_L2_DC, [N_d_inner*(mg+1), L, N_bothz]);
                    [~, maxindex2] = max(RHS_L2_flat_DC, [], 1);

                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    midpt_local(1, curraindex, :) = le_flat(1,:,:) + a1p_offset - 1;
                end

                midpoint = max(min(midpt_local, N_a-1), 2);
                a1primeindexes = (midpoint + (midpoint-1)*n2short) + reshape(gpuArray(-n2short-1:1:n2short+1), [n2long, 1, 1]);
                a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, N_a, N_bothz]);

                RM_GI = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_GI = reshape(RM_GI, [N_d_inner, n2long, N_a, N_bothz]);

                if has_cont
                    EV_idx_GI = a1primeindexes + N_a1fine * bz_off_L2;
                    RHS_GI = RM_GI + df * reshape(EV_interp(EV_idx_GI), [1, n2long, N_a, N_bothz]);
                else
                    RHS_GI = RM_GI;
                end

                RHS_GI_flat = reshape(RHS_GI, [N_d_inner * n2long, N_a, N_bothz]);
                [v_tmp, p_tmp] = max(RHS_GI_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;

                linidx_lower = d_ind + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*bz_off_flat;
                linidx_upper = d_ind + N_d_inner*(n2long-1) + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*bz_off_flat;

                isInfLower = (RM_GI(linidx_lower) == -Inf);
                isInfUpper = (RM_GI(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:,:,e_c,d_outer_c) = reshape(v_tmp, [N_a, N_bothz, 1]);
                P1_fordouter(:,:,e_c,d_outer_c) = reshape(d_idx_val, [N_a, N_bothz, 1]);
                P2_fordouter(:,:,e_c,d_outer_c) = reshape(midpoint, [N_a, N_bothz, 1]);
                P3_fordouter(:,:,e_c,d_outer_c) = reshape(L2offset, [N_a, N_bothz, 1]);
                P4_fordouter(:,:,e_c,d_outer_c) = reshape(flag_val, [N_a, N_bothz, 1]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);
        sb_off_L2  = reshape(gpuArray(0:N_semiz-1), [1, 1, N_semiz, 1]);

        a1_off_flat = reshape(gpuArray(0:N_a-1), [1, N_a, 1, 1]);
        sb_off_flat = reshape(gpuArray(0:N_semiz-1), [1, 1, N_semiz, 1]);
        e_off_flat  = reshape(gpuArray(0:N_e-1), [1, 1, 1, N_e]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_bothz, a_gridvals, a1prime_grid);
                EV_interp = reshape(EV_interp, [N_a1fine, N_semiz, N_z]);
                EV_expected = df * compute_expected_value(EVpre_flat, pi_bz, N_a, N_bothz);
                EV_expected_reshaped = reshape(EV_expected, [N_a, N_semiz, N_z]);
                DiscountedEV_pre = reshape(EV_expected, [1, N_a, 1, N_semiz, N_z]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);
                midpt_local = zeros(1, N_a, N_semiz, N_e, 'gpuArray');

                if has_cont
                    DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,z_c);
                    EV_safe_z = reshape(EV_expected_reshaped(:,:,z_c), [N_a, N_semiz]);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a, vfoptions.level1n, N_semiz, N_e]);
                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, vfoptions.level1n, N_semiz, N_e]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);
                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;
                midpt_local(1, level1ii, :, :) = a1p_idx_L1;

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :) - a1p_idx_L1(1, 1:end-1, :, :));
                maxgap_vec = gather(squeeze(max(max(max(diff_a1p, [], 4), [], 3), [], 1)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(vfoptions.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, :), N_a - mg), [1, 1, N_semiz, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [mg+1, 1, 1, 1]);
                    le_flat = repmat(loweredge, [1, L, 1, 1]);
                    a1p_idx = le_flat + a1p_offsets;

                    a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, N_semiz, N_e]);

                    RM_L2_DC = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, ...
                        a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_semiz, N_e]);

                    if has_cont
                        EV_idx = a1p_idx + N_a * sb_off_L2;
                        RHS_L2_DC = RM_L2_DC + reshape(EV_safe_z(EV_idx), [1, mg+1, L, N_semiz, 1]);
                    else
                        RHS_L2_DC = RM_L2_DC;
                    end

                    RHS_L2_flat_DC = reshape(RHS_L2_DC, [N_d_inner*(mg+1), L, N_semiz, N_e]);
                    [~, maxindex2] = max(RHS_L2_flat_DC, [], 1);

                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    midpt_local(1, curraindex, :, :) = le_flat(1,:,:,:) + a1p_offset - 1;
                end

                midpoint = max(min(midpt_local, N_a-1), 2);
                a1primeindexes = (midpoint + (midpoint-1)*n2short) + reshape(gpuArray(-n2short-1:1:n2short+1), [n2long, 1, 1, 1]);
                a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, N_a, N_semiz, N_e]);

                RM_GI = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_GI = reshape(RM_GI, [N_d_inner, n2long, N_a, N_semiz, N_e]);

                if has_cont
                    EV_idx_GI = a1primeindexes + N_a1fine * sb_off_L2;
                    RHS_GI = RM_GI + df * reshape(EV_interp(EV_idx_GI, z_c), [1, n2long, N_a, N_semiz, 1]);
                else
                    RHS_GI = RM_GI;
                end

                RHS_GI_flat = reshape(RHS_GI, [N_d_inner * n2long, N_a, N_semiz, N_e]);
                [v_tmp, p_tmp] = max(RHS_GI_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;

                linidx_lower = d_ind + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*sb_off_flat + N_d_inner*n2long*N_a*N_semiz*e_off_flat;
                linidx_upper = d_ind + N_d_inner*(n2long-1) + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*sb_off_flat + N_d_inner*n2long*N_a*N_semiz*e_off_flat;

                isInfLower = (RM_GI(linidx_lower) == -Inf);
                isInfUpper = (RM_GI(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:,:,sb,:,d_outer_c) = reshape(v_tmp, [N_a, N_semiz, N_e]);
                P1_fordouter(:,:,sb,:,d_outer_c) = reshape(d_idx_val, [N_a, N_semiz, N_e]);
                P2_fordouter(:,:,sb,:,d_outer_c) = reshape(midpoint, [N_a, N_semiz, N_e]);
                P3_fordouter(:,:,sb,:,d_outer_c) = reshape(L2offset, [N_a, N_semiz, N_e]);
                P4_fordouter(:,:,sb,:,d_outer_c) = reshape(flag_val, [N_a, N_semiz, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);
        a1_off_flat = reshape(gpuArray(0:N_a-1), [1, N_a, 1]);
        e_off_flat  = reshape(gpuArray(0:N_e-1), [1, 1, N_e]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_bothz, a_gridvals, a1prime_grid);
                EV_expected = df * compute_expected_value(EVpre_flat, pi_bz, N_a, N_bothz);
                EV_expected_reshaped = reshape(EV_expected, [N_a, N_bothz]);
                DiscountedEV_pre = reshape(EV_expected, [1, N_a, 1, N_bothz]);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);
                midpt_local = zeros(1, N_a, 1, N_e, 'gpuArray');

                if has_cont
                    DiscountedEV_L1 = DiscountedEV_pre(:,:,:,z_c);
                    EV_safe_z = reshape(EV_expected_reshaped(:,:,z_c), [N_a, 1]);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a, vfoptions.level1n, 1, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, vfoptions.level1n, 1, N_e]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);
                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;
                midpt_local(1, level1ii, 1, :) = a1p_idx_L1;

                [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*N_a, vfoptions.level1n, 1, N_e]), [], 1);
                V_fordouter(level1ii,z_c,:,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, 1, N_e]);
                P1_fordouter(level1ii,z_c,:,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, 1, N_e]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, 1, :) - a1p_idx_L1(1, 1:end-1, 1, :));
                maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 4), [], 2)));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(vfoptions.level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, 1, :), N_a - mg), [1, 1, 1, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [mg+1, 1, 1, 1]);
                    le_flat = repmat(loweredge, [1, L, 1, 1]);
                    a1p_idx = le_flat + a1p_offsets;

                    a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, 1, N_e]);

                    RM_L2_DC = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                        a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, 1, N_e]);

                    if has_cont
                        RHS_L2_DC = RM_L2_DC + reshape(EV_safe_z(a1p_idx), [1, mg+1, L, 1, N_e]);
                    else
                        RHS_L2_DC = RM_L2_DC;
                    end

                    RHS_L2_flat_DC = reshape(RHS_L2_DC, [N_d_inner*(mg+1), L, 1, N_e]);
                    [~, maxindex2] = max(RHS_L2_flat_DC, [], 1);

                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    midpt_local(1, curraindex, 1, :) = le_flat(1,:,1,:) + a1p_offset - 1;
                end

                midpoint = max(min(midpt_local, N_a-1), 2);
                a1primeindexes = (midpoint + (midpoint-1)*n2short) + reshape(gpuArray(-n2short-1:1:n2short+1), [n2long, 1, 1, 1]);
                a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, N_a, 1, N_e]);

                RM_GI = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_GI = reshape(RM_GI, [N_d_inner, n2long, N_a, 1, N_e]);

                if has_cont
                    RHS_GI = RM_GI + df * reshape(EV_interp(a1primeindexes, z_c), [1, n2long, N_a, 1, N_e]);
                else
                    RHS_GI = RM_GI;
                end

                RHS_GI_flat = reshape(RHS_GI, [N_d_inner * n2long, N_a, 1, N_e]);
                [v_tmp, p_tmp] = max(RHS_GI_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;

                linidx_lower = d_ind + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*e_off_flat;
                linidx_upper = d_ind + N_d_inner*(n2long-1) + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*e_off_flat;

                isInfLower = (RM_GI(linidx_lower) == -Inf);
                isInfUpper = (RM_GI(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:,:,z_c,:,d_outer_c) = reshape(v_tmp, [N_a, 1, N_e]);
                P1_fordouter(:,:,z_c,:,d_outer_c) = reshape(d_idx_val, [N_a, 1, N_e]);
                P2_fordouter(:,:,z_c,:,d_outer_c) = reshape(midpoint, [N_a, 1, N_e]);
                P3_fordouter(:,:,z_c,:,d_outer_c) = reshape(L2offset, [N_a, 1, N_e]);
                P4_fordouter(:,:,z_c,:,d_outer_c) = reshape(flag_val, [N_a, 1, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        sb_off_L2  = reshape(gpuArray(0:N_semiz-1), [1, 1, N_semiz]);
        a1_off_flat = reshape(gpuArray(0:N_a-1), [1, N_a, 1]);
        sb_off_flat = reshape(gpuArray(0:N_semiz-1), [1, 1, N_semiz]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_bothz, a_gridvals, a1prime_grid);
                EV_interp = reshape(EV_interp, [N_a1fine, N_semiz, N_z]);
                EV_expected = df * compute_expected_value(EVpre_flat, pi_bz, N_a, N_bothz);
                EV_expected_reshaped = reshape(EV_expected, [N_a, N_semiz, N_z]);
                DiscountedEV_pre = reshape(EV_expected, [1, N_a, 1, N_semiz, N_z]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                if has_cont
                    DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,z_c);
                    EV_safe_z = reshape(EV_expected_reshaped(:,:,z_c), [N_a, N_semiz]);
                end

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);
                    midpt_local = zeros(1, N_a, N_semiz, 1, 'gpuArray');

                    RM_L1 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d_inner, N_a, vfoptions.level1n, N_semiz, 1]);
                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, vfoptions.level1n, N_semiz, 1]);
                    [~, maxindex1] = max(RHS_L1_flat, [], 1);
                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;
                    midpt_local(1, level1ii, :, 1) = a1p_idx_L1;

                    [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*N_a, vfoptions.level1n, N_semiz, 1]), [], 1);
                    V_fordouter(level1ii,sb,e_c,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, N_semiz, 1]);
                    P1_fordouter(level1ii,sb,e_c,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, N_semiz, 1]);

                    diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, 1) - a1p_idx_L1(1, 1:end-1, :, 1));
                    maxgap_vec = gather(squeeze(max(max(diff_a1p, [], 4), [], 3)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                        loweredge = reshape(min(a1p_idx_L1(1, ii, :, 1), N_a - mg), [1, 1, N_semiz, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [mg+1, 1, 1]);
                        le_flat = repmat(loweredge, [1, L, 1]);
                        a1p_idx = le_flat + a1p_offsets;

                        a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, N_semiz, 1]);

                        RM_L2_DC = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, ...
                            a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_semiz, 1]);

                        if has_cont
                            EV_idx = a1p_idx + N_a * sb_off_L2;
                            RHS_L2_DC = RM_L2_DC + reshape(EV_safe_z(EV_idx), [1, mg+1, L, N_semiz, 1]);
                        else
                            RHS_L2_DC = RM_L2_DC;
                        end

                        RHS_L2_flat_DC = reshape(RHS_L2_DC, [N_d_inner*(mg+1), L, N_semiz, 1]);
                        [~, maxindex2] = max(RHS_L2_flat_DC, [], 1);

                        a1p_offset = ceil(maxindex2 / N_d_inner);
                        midpt_local(1, curraindex, :, 1) = le_flat(1,:,:) + a1p_offset - 1;
                    end

                    midpoint = max(min(midpt_local, N_a-1), 2);
                    a1primeindexes = (midpoint + (midpoint-1)*n2short) + reshape(gpuArray(-n2short-1:1:n2short+1), [n2long, 1, 1]);
                    a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, N_a, N_semiz, 1]);

                    RM_GI = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_GI = reshape(RM_GI, [N_d_inner, n2long, N_a, N_semiz, 1]);

                    if has_cont
                        EV_idx_GI = a1primeindexes + N_a1fine * sb_off_L2;
                        RHS_GI = RM_GI + df * reshape(EV_interp(EV_idx_GI, z_c), [1, n2long, N_a, N_semiz, 1]);
                    else
                        RHS_GI = RM_GI;
                    end

                    RHS_GI_flat = reshape(RHS_GI, [N_d_inner * n2long, N_a, N_semiz, 1]);
                    [v_tmp, p_tmp] = max(RHS_GI_flat, [], 1);

                    d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                    L2_and_a2p = ceil(p_tmp / N_d_inner);
                    L2offset = rem(L2_and_a2p - 1, n2long) + 1;

                    linidx_lower = d_ind + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*sb_off_flat;
                    linidx_upper = d_ind + N_d_inner*(n2long-1) + N_d_inner*n2long*a1_off_flat + N_d_inner*n2long*N_a*sb_off_flat;

                    isInfLower = (RM_GI(linidx_lower) == -Inf);
                    isInfUpper = (RM_GI(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                    V_fordouter(:,:,sb,e_c,d_outer_c) = reshape(v_tmp, [N_a, N_semiz, 1]);
                    P1_fordouter(:,:,sb,e_c,d_outer_c) = reshape(d_idx_val, [N_a, N_semiz, 1]);
                    P2_fordouter(:,:,sb,e_c,d_outer_c) = reshape(midpoint, [N_a, N_semiz, 1]);
                    P3_fordouter(:,:,sb,e_c,d_outer_c) = reshape(L2offset, [N_a, N_semiz, 1]);
                    P4_fordouter(:,:,sb,e_c,d_outer_c) = reshape(flag_val, [N_a, N_semiz, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        a1_off_flat = reshape(gpuArray(0:N_a-1), [1, N_a, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_bothz; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_bothz, a_gridvals, a1prime_grid);
                EV_expected = df * compute_expected_value(EVpre_flat, pi_bz, N_a, N_bothz);
                EV_expected_reshaped = reshape(EV_expected, [N_a, N_bothz]);
                DiscountedEV_pre = reshape(EV_expected, [1, N_a, 1, N_bothz]);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);
                    midpt_local = zeros(1, N_a, 1, 1, 'gpuArray');

                    if has_cont
                        DiscountedEV_L1 = DiscountedEV_pre(:,:,:,z_c);
                        EV_safe_z = reshape(EV_expected_reshaped(:,:,z_c), [N_a, 1]);
                    end

                    RM_L1 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d_inner, N_a, vfoptions.level1n, 1, 1]);
                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a, vfoptions.level1n, 1, 1]);
                    [~, maxindex1] = max(RHS_L1_flat, [], 1);
                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a) + 1;
                    midpt_local(1, level1ii, 1, 1) = a1p_idx_L1;

                    [v_tmp, p_tmp] = max(reshape(RHS_L1, [N_d_inner*N_a, vfoptions.level1n, 1, 1]), [], 1);
                    V_fordouter(level1ii,z_c,e_c,d_outer_c) = reshape(v_tmp, [vfoptions.level1n, 1, 1]);
                    P1_fordouter(level1ii,z_c,e_c,d_outer_c) = reshape(p_tmp, [vfoptions.level1n, 1, 1]);

                    diff_a1p = max(0, a1p_idx_L1(1, 2:end, 1, 1) - a1p_idx_L1(1, 1:end-1, 1, 1));
                    maxgap_vec = gather(squeeze(max(diff_a1p, [], 2)));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                        loweredge = reshape(min(a1p_idx_L1(1, ii, 1, 1), N_a - mg), [1, 1, 1, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [mg+1, 1, 1]);
                        le_flat = repmat(loweredge, [1, L, 1]);
                        a1p_idx = le_flat + a1p_offsets;

                        a1p_grid_safe = reshape(a_gridvals(a1p_idx), [1, mg+1, L, 1, 1]);

                        RM_L2_DC = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                            a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, 1, 1]);

                        if has_cont
                            RHS_L2_DC = RM_L2_DC + reshape(EV_safe_z(a1p_idx), [1, mg+1, L, 1, 1]);
                        else
                            RHS_L2_DC = RM_L2_DC;
                        end

                        RHS_L2_flat_DC = reshape(RHS_L2_DC, [N_d_inner*(mg+1), L, 1, 1]);
                        [~, maxindex2] = max(RHS_L2_flat_DC, [], 1);

                        a1p_offset = ceil(maxindex2 / N_d_inner);
                        midpt_local(1, curraindex, 1, 1) = le_flat(1,:,:) + a1p_offset - 1;
                    end

                    midpoint = max(min(midpt_local, N_a-1), 2);
                    a1primeindexes = (midpoint + (midpoint-1)*n2short) + reshape(gpuArray(-n2short-1:1:n2short+1), [n2long, 1, 1]);
                    a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, N_a, 1, 1]);

                    RM_GI = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_GI = reshape(RM_GI, [N_d_inner, n2long, N_a, 1, 1]);

                    if has_cont
                        RHS_GI = RM_GI + df * reshape(EV_interp(a1primeindexes, z_c), [1, n2long, N_a, 1, 1]);
                    else
                        RHS_GI = RM_GI;
                    end

                    RHS_GI_flat = reshape(RHS_GI, [N_d_inner * n2long, N_a, 1, 1]);
                    [v_tmp, p_tmp] = max(RHS_GI_flat, [], 1);

                    d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                    L2_and_a2p = ceil(p_tmp / N_d_inner);
                    L2offset = rem(L2_and_a2p - 1, n2long) + 1;

                    linidx_lower = d_ind + N_d_inner*n2long*a1_off_flat;
                    linidx_upper = d_ind + N_d_inner*(n2long-1) + N_d_inner*n2long*a1_off_flat;

                    isInfLower = (RM_GI(linidx_lower) == -Inf);
                    isInfUpper = (RM_GI(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                    V_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(v_tmp, [N_a, 1, 1]);
                    P1_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(d_idx_val, [N_a, 1, 1]);
                    P2_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(midpoint, [N_a, 1, 1]);
                    P3_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(L2offset, [N_a, 1, 1]);
                    P4_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(flag_val, [N_a, 1, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
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

    function [V_out, Pol_out] = extract_pol(V_douter, P1, P2, P3, P4, N_a, N_bothz, N_e, N_d_outer)
        if N_d_outer > 1
            [V_out, maxidx] = max(V_douter, [], 4);
            lin_idx = (1:(N_a*N_bothz*N_e))' + (N_a*N_bothz*N_e)*(maxidx(:) - 1);
            Pol_out = zeros(4, N_a, N_bothz, N_e, 'gpuArray');
            Pol_out(1,:,:,:) = reshape(P1(lin_idx), [1, N_a, N_bothz, N_e]);
            Pol_out(2,:,:,:) = reshape(P2(lin_idx), [1, N_a, N_bothz, N_e]);
            Pol_out(3,:,:,:) = reshape(P3(lin_idx), [1, N_a, N_bothz, N_e]);
            Pol_out(4,:,:,:) = reshape(P4(lin_idx), [1, N_a, N_bothz, N_e]);
        else
            V_out = reshape(V_douter, [N_a, N_bothz, N_e]);
            Pol_out = zeros(4, N_a, N_bothz, N_e, 'gpuArray');
            Pol_out(1,:,:,:) = reshape(P1, [1, N_a, N_bothz, N_e]);
            Pol_out(2,:,:,:) = reshape(P2, [1, N_a, N_bothz, N_e]);
            Pol_out(3,:,:,:) = reshape(P3, [1, N_a, N_bothz, N_e]);
            Pol_out(4,:,:,:) = reshape(P4, [1, N_a, N_bothz, N_e]);
        end
    end

    function EV_interp = precalc_ev(EVpre_flat, pi_bz, N_bothz, a_gridvals, a1prime_grid)
        EVinf = (EVpre_flat == -Inf);
        EV_val = EVpre_flat;
        EV_val(EVinf) = -1e250;
        EV_expected = EV_val * pi_bz';
        EV_expected((EVinf * pi_bz') > 0) = -Inf;

        EV_interp_next = interp1(a_gridvals, EV_expected, a1prime_grid);
        EV_interp = reshape(EV_interp_next, [length(a1prime_grid), N_bothz]);
    end

    function EV_out = compute_expected_value(EV, pi_bothz, N_a, N_bothz)
        EVinf = (EV == -Inf);
        EV_val = EV;
        EV_val(EVinf) = -1e250;
        EV_val = EV_val * pi_bothz';
        EV_val(EVinf * (pi_bothz' > 0) > 0) = -Inf;
        EV_out = reshape(EV_val, [N_a, 1, N_bothz]);
    end


end
