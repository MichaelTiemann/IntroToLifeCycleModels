function [V, Policy] = ValueFnIter_UFHorz_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

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

% GI2A specific split: a1 (interpolated) and a2 (discrete)
N_a1 = n_a(1);
N_a2 = max(prod(n_a(2:end)), 1);
N_a = N_a1 * N_a2;

a1_gridvals = gpuArray(a_grid(1:N_a1));
a2_gridvals = gpuArray(a_grid(N_a1+1:end));

n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
a1prime_grid = gpuArray(interp1(1:1:N_a1, gather(a1_gridvals), linspace(1, N_a1, N_a1+(N_a1-1)*n2short))');
N_a1fine = length(a1prime_grid);

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

n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1, 1, 1]);

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
% SPECIALIZED SOLVE FUNCTIONS (Vectorised over N_d_inner and N_a2)
% =====================================================================
    function [V_j, Pol_j] = solve_period_full(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        N_e_interp = 1;

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if N_e_raw; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1, 1]);
        bz_off_L2  = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, N_bothz, 1]);

        a1_off = reshape(gpuArray(0:N_a1-1), [1, 1, 1, N_a1, 1, 1, 1]);
        a2_off = reshape(gpuArray(0:N_a2-1), [1, 1, 1, 1, N_a2, 1, 1]);
        bz_off = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, N_bothz, 1]);
        e_off = reshape(gpuArray(0:N_e-1), [1, 1, 1, 1, 1, 1, N_e]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid);
                DiscountedEV_L1 = df * reshape(EVpre_flat * pi_bz', [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
            end

            RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
            RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, N_a1, N_a2, N_bothz, N_e]);

            if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

            RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, N_a1, N_a2, N_bothz, N_e]);
            [~, maxindex1] = max(RHS_L1_flat, [], 1);

            a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
            a1prime_idx = rem(a1p_a2p_ind - 1, N_a1) + 1;
            midpoint = max(min(a1prime_idx, N_a1-1), 2);

            a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
            a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, 1, N_a1, N_a2, N_bothz, N_e]);

            RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);
            RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a2, N_a1, N_a2, N_bothz, N_e]);

            if has_cont
                EV_idx = a1primeindexes + N_a1fine * a2p_off_L2 + N_a1fine * N_a2 * bz_off_L2;
                RHS_L2 = RM_L2 + df * reshape(EV_interp(EV_idx), size(a1primeindexes));
            else
                RHS_L2 = RM_L2;
            end

            RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * N_a2, N_a1, N_a2, N_bothz, N_e]);
            [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

            d_ind = rem(p_tmp - 1, N_d_inner) + 1;
            L2_and_a2p = ceil(p_tmp / N_d_inner);
            L2offset = rem(L2_and_a2p - 1, n2long) + 1;
            a2p_idx_chosen = ceil(L2_and_a2p / n2long);

            S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a2; S4 = S3 * N_a1; S5 = S4 * N_a2; S6 = S5 * N_bothz; S7 = S6 * N_e;
            linidx_lower = d_ind + S1*(1 - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*bz_off + S6*e_off;
            linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*bz_off + S6*e_off;

            isInfLower = (RM_L2(linidx_lower) == -Inf);
            isInfUpper = (RM_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            SM1 = 1; SM2 = SM1 * N_a1; SM3 = SM2 * N_a2; SM4 = SM3 * N_bothz; SM5 = SM4 * N_e;
            midpt_idx = 1 + SM1*a1_off + SM2*a2_off + SM3*bz_off + SM4*e_off;
            chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, N_bothz, N_e]);
            joint_a_idx = chosen_midpt + N_a1 * (a2p_idx_chosen - 1);

            d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

            V_fordouter(:,:,:,:,d_outer_c) = reshape(v_tmp, [N_a1, N_a2, N_bothz, N_e]);
            P1_fordouter(:,:,:,:,d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, N_bothz, N_e]);
            P2_fordouter(:,:,:,:,d_outer_c) = reshape(joint_a_idx, [N_a1, N_a2, N_bothz, N_e]);
            P3_fordouter(:,:,:,:,d_outer_c) = reshape(L2offset, [N_a1, N_a2, N_bothz, N_e]);
            P4_fordouter(:,:,:,:,d_outer_c) = reshape(flag_val, [N_a1, N_a2, N_bothz, N_e]);
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
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
        P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1]);
        bz_off_L2  = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, N_bothz]);

        a1_off = reshape(gpuArray(0:N_a1-1), [1, 1, 1, N_a1, 1, 1]);
        a2_off = reshape(gpuArray(0:N_a2-1), [1, 1, 1, 1, N_a2, 1]);
        bz_off = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, N_bothz]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid);
                DiscountedEV_L1 = df * reshape(EVpre_flat * pi_bz', [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
            end

            for e_c = 1:N_e
                e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);

                RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, N_a1, N_a2, N_bothz, 1]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, N_a1, N_a2, N_bothz, 1]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1prime_idx = rem(a1p_a2p_ind - 1, N_a1) + 1;
                midpoint = max(min(a1prime_idx, N_a1-1), 2);

                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, 1, N_a1, N_a2, N_bothz, 1]);

                RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a2, N_a1, N_a2, N_bothz, 1]);

                if has_cont
                    EV_idx = a1primeindexes + N_a1fine * a2p_off_L2 + N_a1fine * N_a2 * bz_off_L2;
                    RHS_L2 = RM_L2 + df * reshape(EV_interp(EV_idx), size(a1primeindexes));
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * N_a2, N_a1, N_a2, N_bothz, 1]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a2; S4 = S3 * N_a1; S5 = S4 * N_a2; S6 = S5 * N_bothz;
                linidx_lower = d_ind + S1*(1 - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*bz_off;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*bz_off;

                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                SM1 = 1; SM2 = SM1 * N_a1; SM3 = SM2 * N_a2; SM4 = SM3 * N_bothz;
                midpt_idx = 1 + SM1*a1_off + SM2*a2_off + SM3*bz_off;
                chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, N_bothz, 1]);
                joint_a_idx = chosen_midpt + N_a1 * (a2p_idx_chosen - 1);

                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:,:,:,e_c,d_outer_c) = reshape(v_tmp, [N_a1, N_a2, N_bothz, 1]);
                P1_fordouter(:,:,:,e_c,d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, N_bothz, 1]);
                P2_fordouter(:,:,:,e_c,d_outer_c) = reshape(joint_a_idx, [N_a1, N_a2, N_bothz, 1]);
                P3_fordouter(:,:,:,e_c,d_outer_c) = reshape(L2offset, [N_a1, N_a2, N_bothz, 1]);
                P4_fordouter(:,:,:,e_c,d_outer_c) = reshape(flag_val, [N_a1, N_a2, N_bothz, 1]);
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
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1, 1]);
        sb_off_L2  = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, N_semiz, 1]);

        a1_off = reshape(gpuArray(0:N_a1-1), [1, 1, 1, N_a1, 1, 1, 1]);
        a2_off = reshape(gpuArray(0:N_a2-1), [1, 1, 1, 1, N_a2, 1, 1]);
        sb_off = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, N_semiz, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp = reshape(precalc_ev(EVpre_flat, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid), [N_a1fine, N_a2, N_semiz, N_z]);
                DiscountedEV_pre = df * reshape(EVpre_flat * pi_bz', [1, N_a1, N_a2, 1, 1, N_semiz, N_z]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                if has_cont
                    DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,:,:,z_c);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, N_a1, N_a2, N_semiz, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, N_a1, N_a2, N_semiz, N_e]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1prime_idx = rem(a1p_a2p_ind - 1, N_a1) + 1;
                midpoint = max(min(a1prime_idx, N_a1-1), 2);

                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, 1, N_a1, N_a2, N_semiz, N_e]);

                RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a2, N_a1, N_a2, N_semiz, N_e]);

                if has_cont
                    EV_idx = a1primeindexes + N_a1fine * a2p_off_L2 + N_a1fine * N_a2 * sb_off_L2 + N_a1fine * N_a2 * N_semiz * (z_c - 1);
                    RHS_L2 = RM_L2 + df * reshape(EV_interp(EV_idx), size(a1primeindexes));
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * N_a2, N_a1, N_a2, N_semiz, N_e]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a2; S4 = S3 * N_a1; S5 = S4 * N_a2; S6 = S5 * N_semiz;
                linidx_lower = d_ind + S1*(1 - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*sb_off;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*sb_off;

                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                SM1 = 1; SM2 = SM1 * N_a1; SM3 = SM2 * N_a2; SM4 = SM3 * N_semiz;
                midpt_idx = 1 + SM1*a1_off + SM2*a2_off + SM3*sb_off;
                chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, N_semiz, N_e]);
                joint_a_idx = chosen_midpt + N_a1 * (a2p_idx_chosen - 1);

                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:, :, sb, :, d_outer_c) = reshape(v_tmp, [N_a1, N_a2, N_semiz, N_e]);
                P1_fordouter(:, :, sb, :, d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, N_semiz, N_e]);
                P2_fordouter(:, :, sb, :, d_outer_c) = reshape(joint_a_idx, [N_a1, N_a2, N_semiz, N_e]);
                P3_fordouter(:, :, sb, :, d_outer_c) = reshape(L2offset, [N_a1, N_a2, N_semiz, N_e]);
                P4_fordouter(:, :, sb, :, d_outer_c) = reshape(flag_val, [N_a1, N_a2, N_semiz, N_e]);
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
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw);

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1]);

        a1_off = reshape(gpuArray(0:N_a1-1), [1, 1, 1, N_a1, 1, 1]);
        a2_off = reshape(gpuArray(0:N_a2-1), [1, 1, 1, 1, N_a2, 1]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid);
                DiscountedEV_pre = df * reshape(EVpre_flat * pi_bz', [1, N_a1, N_a2, 1, 1, N_bothz]);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                if has_cont
                    DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,:,z_c);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, N_a1, N_a2, 1, N_e]);

                if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, N_a1, N_a2, 1, N_e]);
                [~, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1prime_idx = rem(a1p_a2p_ind - 1, N_a1) + 1;
                midpoint = max(min(a1prime_idx, N_a1-1), 2);

                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, 1, N_a1, N_a2, 1, N_e]);

                RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a2, N_a1, N_a2, 1, N_e]);

                if has_cont
                    EV_idx = a1primeindexes + N_a1fine * a2p_off_L2 + N_a1fine * N_a2 * (z_c - 1);
                    RHS_L2 = RM_L2 + df * reshape(EV_interp(EV_idx), size(a1primeindexes));
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * N_a2, N_a1, N_a2, 1, N_e]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a2; S4 = S3 * N_a1; S5 = S4 * N_a2;
                linidx_lower = d_ind + S1*(1 - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off;

                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                SM1 = 1; SM2 = SM1 * N_a1; SM3 = SM2 * N_a2;
                midpt_idx = 1 + SM1*a1_off + SM2*a2_off;
                chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, 1, N_e]);
                joint_a_idx = chosen_midpt + N_a1 * (a2p_idx_chosen - 1);

                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:, :, z_c, :, d_outer_c) = reshape(v_tmp, [N_a1, N_a2, 1, N_e]);
                P1_fordouter(:, :, z_c, :, d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, 1, N_e]);
                P2_fordouter(:, :, z_c, :, d_outer_c) = reshape(joint_a_idx, [N_a1, N_a2, 1, N_e]);
                P3_fordouter(:, :, z_c, :, d_outer_c) = reshape(L2offset, [N_a1, N_a2, 1, N_e]);
                P4_fordouter(:, :, z_c, :, d_outer_c) = reshape(flag_val, [N_a1, N_a2, 1, N_e]);
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
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1, 1]);
        sb_off_L2  = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, N_semiz]);

        a1_off = reshape(gpuArray(0:N_a1-1), [1, 1, 1, N_a1, 1, 1]);
        a2_off = reshape(gpuArray(0:N_a2-1), [1, 1, 1, 1, N_a2, 1]);
        sb_off = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, N_semiz]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp = reshape(precalc_ev(EVpre_flat, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid), [N_a1fine, N_a2, N_semiz, N_z]);
                DiscountedEV_pre = df * reshape(EVpre_flat * pi_bz', [1, N_a1, N_a2, 1, 1, N_semiz, N_z]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);

                    if has_cont
                        DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,:,:,z_c);
                    end

                    RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                    RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, N_a1, N_a2, N_semiz, 1]);

                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, N_a1, N_a2, N_semiz, 1]);
                    [~, maxindex1] = max(RHS_L1_flat, [], 1);

                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1prime_idx = rem(a1p_a2p_ind - 1, N_a1) + 1;
                    midpoint = max(min(a1prime_idx, N_a1-1), 2);

                    a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                    a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, 1, N_a1, N_a2, N_semiz, 1]);

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);
                    RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a2, N_a1, N_a2, N_semiz, 1]);

                    if has_cont
                        EV_idx = a1primeindexes + N_a1fine * a2p_off_L2 + N_a1fine * N_a2 * sb_off_L2 + N_a1fine * N_a2 * N_semiz * (z_c - 1);
                        RHS_L2 = RM_L2 + df * reshape(EV_interp(EV_idx), size(a1primeindexes));
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * N_a2, N_a1, N_a2, N_semiz, 1]);
                    [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                    d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                    L2_and_a2p = ceil(p_tmp / N_d_inner);
                    L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                    a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                    S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a2; S4 = S3 * N_a1; S5 = S4 * N_a2; S6 = S5 * N_semiz;
                    linidx_lower = d_ind + S1*(1 - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*sb_off;
                    linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*sb_off;

                    isInfLower = (RM_L2(linidx_lower) == -Inf);
                    isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    SM1 = 1; SM2 = SM1 * N_a1; SM3 = SM2 * N_a2; SM4 = SM3 * N_semiz;
                    midpt_idx = 1 + SM1*a1_off + SM2*a2_off + SM3*sb_off;
                    chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, N_semiz, 1]);
                    joint_a_idx = chosen_midpt + N_a1 * (a2p_idx_chosen - 1);

                    d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                    V_fordouter(:, :, sb, e_c, d_outer_c) = reshape(v_tmp, [N_a1, N_a2, N_semiz, 1]);
                    P1_fordouter(:, :, sb, e_c, d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, N_semiz, 1]);
                    P2_fordouter(:, :, sb, e_c, d_outer_c) = reshape(joint_a_idx, [N_a1, N_a2, N_semiz, 1]);
                    P3_fordouter(:, :, sb, e_c, d_outer_c) = reshape(L2offset, [N_a1, N_a2, N_semiz, 1]);
                    P4_fordouter(:, :, sb, e_c, d_outer_c) = reshape(flag_val, [N_a1, N_a2, N_semiz, 1]);
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
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        a2p_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1]);
        a1_off = reshape(gpuArray(0:N_a1-1), [1, 1, 1, N_a1, 1]);
        a2_off = reshape(gpuArray(0:N_a2-1), [1, 1, 1, 1, N_a2]);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_interp = precalc_ev(EVpre_flat, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid);
                DiscountedEV_pre = df * reshape(EVpre_flat * pi_bz', [1, N_a1, N_a2, 1, 1, N_bothz]);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, N_e_raw);

                    if has_cont
                        DiscountedEV_L1 = DiscountedEV_pre(:,:,:,:,:,z_c);
                    end

                    RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                    RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, N_a1, N_a2, 1, 1]);

                    if has_cont; RHS_L1 = RM_L1 + DiscountedEV_L1; else; RHS_L1 = RM_L1; end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, N_a1, N_a2, 1, 1]);
                    [~, maxindex1] = max(RHS_L1_flat, [], 1);

                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1prime_idx = rem(a1p_a2p_ind - 1, N_a1) + 1;
                    midpoint = max(min(a1prime_idx, N_a1-1), 2);

                    a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                    a1p_grid_safe = reshape(a1prime_grid(a1primeindexes), [1, n2long, 1, N_a1, N_a2, 1, 1]);

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);
                    RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a2, N_a1, N_a2, 1, 1]);

                    if has_cont
                        EV_idx = a1primeindexes + N_a1fine * a2p_off_L2 + N_a1fine * N_a2 * (z_c - 1);
                        RHS_L2 = RM_L2 + df * reshape(EV_interp(EV_idx), size(a1primeindexes));
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * N_a2, N_a1, N_a2, 1, 1]);
                    [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                    d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                    L2_and_a2p = ceil(p_tmp / N_d_inner);
                    L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                    a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                    S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a2; S4 = S3 * N_a1; S5 = S4 * N_a2;
                    linidx_lower = d_ind + S1*(1 - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off;
                    linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off;

                    isInfLower = (RM_L2(linidx_lower) == -Inf);
                    isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    SM1 = 1; SM2 = SM1 * N_a1; SM3 = SM2 * N_a2;
                    midpt_idx = 1 + SM1*a1_off + SM2*a2_off;
                    chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, 1, 1]);
                    joint_a_idx = chosen_midpt + N_a1 * (a2p_idx_chosen - 1);

                    d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                    V_fordouter(:, :, z_c, e_c, d_outer_c) = reshape(v_tmp, [N_a1, N_a2, 1, 1]);
                    P1_fordouter(:, :, z_c, e_c, d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, 1, 1]);
                    P2_fordouter(:, :, z_c, e_c, d_outer_c) = reshape(joint_a_idx, [N_a1, N_a2, 1, 1]);
                    P3_fordouter(:, :, z_c, e_c, d_outer_c) = reshape(L2offset, [N_a1, N_a2, 1, 1]);
                    P4_fordouter(:, :, z_c, e_c, d_outer_c) = reshape(flag_val, [N_a1, N_a2, 1, 1]);
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
            [V_out, maxidx] = max(V_douter, [], 5);
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

    function EV_interp = precalc_ev(EVpre_flat, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid)
        EVinf = (EVpre_flat == -Inf);
        EV_val = EVpre_flat;
        EV_val(EVinf) = -1e250;
        EV_expected = EV_val * pi_bz';
        EV_expected((EVinf * pi_bz') > 0) = -Inf;

        % Standard models expect V(a1', a2', z'). Interpolate over a1' grid.
        EV_reshaped = reshape(EV_expected, [N_a1, N_a2, N_bothz]);
        EV_interp_next = permute(interp1(a1_gridvals, permute(EV_reshaped, [1, 2, 3]), a1prime_grid), [1, 2, 3]);
        EV_interp = reshape(EV_interp_next, [length(a1prime_grid), N_a2, N_bothz]);
    end


end
