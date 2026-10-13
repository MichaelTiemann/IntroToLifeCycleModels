function [V, Policy] = ValueFnIter_UFHorz_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_d = Dims.N_d; N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;

has_d2 = Dims.has_d2; has_d1 = Dims.has_d1;
N_semiz_raw = Dims.N_semiz_raw;
has_d = Dims.has_d1 || Dims.has_d2;

has_semiz = Dims.N_semiz_raw > 0;
has_e     = Dims.N_e_raw > 0;
has_z     = Dims.special_n_z(1) > 0;
has_bothz = has_semiz || has_z;

N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;

a_gridvals = gpuArray(a_grid);

% Grid interpolation setup
n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
aprime_grid = gpuArray(interp1(1:1:N_a, gather(a_gridvals), linspace(1, N_a, N_a+(N_a-1)*n2short))');
n2aprime = length(aprime_grid);

% Preallocate outputs
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');

n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1]);
bothzBind = reshape(gpuArray(0:1:N_bothz-1), [1, 1, 1, N_bothz, 1]);
semizBind = reshape(gpuArray(0:1:N_semiz-1), [1, 1, 1, N_semiz, 1]);

aind     = reshape(gpuArray(0:1:N_a-1),     [1, N_a, 1, 1]);
bothzind = reshape(gpuArray(0:1:N_bothz-1), [1, 1, N_bothz, 1]);
eind     = reshape(gpuArray(0:1:N_e-1),     [1, 1, 1, N_e]);
semizind = reshape(gpuArray(0:1:N_semiz-1), [1, 1, N_semiz, 1]);

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

% --- 3. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]); else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    switch IterStrategy
        case 'Full'; [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period_full(j, V_next);
        case 'Loop_e'; [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period_loop_e(j, V_next);
        case 'Loop_z_parallel_semiz'; [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period_loop_z_parallel_semiz(j, V_next);
        case 'Loop_bothz'; [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period_loop_bothz(j, V_next);
        case 'Loop_z_parallel_semiz_e'; [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period_loop_z_parallel_semiz_e(j, V_next);
        case 'Loop_bothz_e'; [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period_loop_bothz_e(j, V_next);
    end
end

% =====================================================================
% SPECIALIZED SOLVE FUNCTIONS
% =====================================================================
    function [V_j, Pol_j] = solve_period_full(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_continuation = ~isempty(V_next);
        if has_continuation
            DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        maxidx_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        midpt_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        flag_ford2_jj = 2 * ones(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_continuation
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_d2 = compute_expected_value(EV, pi_bz, N_a, N_bothz);
                EVinterp = interp1(a_gridvals, EV_d2, aprime_grid);
            end

            RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
            RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, N_bothz, N_e]);

            if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2, -1); else; entireRHS_L1 = RM_L1; end

            [~, maxindex1] = max(entireRHS_L1, [], 2);
            midpoint = max(min(maxindex1, N_a-1), 2);
            aprimeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;

            a1p_grid_safe = reshape(aprime_grid(aprimeindexes), size(aprimeindexes));

            RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
            RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, N_bothz, N_e]);

            if has_continuation
                aprimez = aprimeindexes + n2aprime * bothzBind;
                entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp(aprimez(:)), [N_d1*n2long, N_a, N_bothz, N_e]);
            else
                entireRHS_L2 = RM_L2;
            end

            [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
            V_ford2_jj(:,:,:,d2_c) = shiftdim(Vtemp, 1);
            maxidx_ford2_jj(:,:,:,d2_c) = shiftdim(maxindexL2, 1);

            d1_ind = rem(maxindexL2-1, N_d1) + 1;
            L2offset = ceil(maxindexL2 / N_d1);

            linidx_lower = d1_ind + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
            isInfLower = (RM_L2(linidx_lower) == -Inf);
            isInfUpper = (RM_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);

            flag_ford2_jj(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            allind = d1_ind + N_d1*aind + N_d1*N_a*bothzind + N_d1*N_a*N_bothz*eind;
            midpt_ford2_jj(:,:,:,d2_c) = shiftdim(midpoint(allind), 1);
        end

        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_GI_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = extract_policy_GI_no_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, N_d1);
        end
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_continuation = ~isempty(V_next);
        if has_continuation
            DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        maxidx_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        midpt_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        flag_ford2_jj = 2 * ones(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_continuation
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
                EV_d2 = compute_expected_value(EV, pi_bz, N_a, N_bothz);
                EVinterp = interp1(a_gridvals, EV_d2, aprime_grid);
            end

            for e_c = 1:N_e
                e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, N_bothz, 1]);

                if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2, -1); else; entireRHS_L1 = RM_L1; end

                [~, maxindex1] = max(entireRHS_L1, [], 2);
                midpoint = max(min(maxindex1, N_a-1), 2);
                aprimeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;

                a1p_grid_safe = reshape(aprime_grid(aprimeindexes), size(aprimeindexes));

                RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, N_bothz, 1]);

                if has_continuation
                    aprimez = aprimeindexes + n2aprime * bothzBind;
                    entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp(aprimez(:)), [N_d1*n2long, N_a, N_bothz, 1]);
                else
                    entireRHS_L2 = RM_L2;
                end

                [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                V_ford2_jj(:,:,e_c,d2_c) = shiftdim(Vtemp, 1);
                maxidx_ford2_jj(:,:,e_c,d2_c) = shiftdim(maxindexL2, 1);

                d1_ind = rem(maxindexL2-1, N_d1) + 1;
                L2offset = ceil(maxindexL2 / N_d1);

                linidx_lower = d1_ind + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);

                flag_ford2_jj(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                allind = d1_ind + N_d1*aind + N_d1*N_a*bothzind;
                midpt_ford2_jj(:,:,e_c,d2_c) = shiftdim(midpoint(allind), 1);
            end
        end

        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_GI_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = extract_policy_GI_no_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, N_d1);
        end
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_continuation = ~isempty(V_next);
        if has_continuation
            DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        maxidx_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        midpt_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        flag_ford2_jj = 2 * ones(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_continuation
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
            end

            for z_c = 1:N_z
                semizblock = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, semizblock, j, has_bothz);

                if has_continuation
                    EV_d2z = compute_expected_value_cond(EV, pi_bz(semizblock,:)', N_a);
                    EVinterp_z = interp1(a_gridvals, EV_d2z, aprime_grid);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, N_semiz, N_e]);

                if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2z, -1); else; entireRHS_L1 = RM_L1; end

                [~, maxindex1] = max(entireRHS_L1, [], 2);
                midpoint = max(min(maxindex1, N_a-1), 2);
                aprimeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;

                a1p_grid_safe = reshape(aprime_grid(aprimeindexes), size(aprimeindexes));

                RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, N_semiz, N_e]);

                if has_continuation
                    aprimez = aprimeindexes + n2aprime * semizBind;
                    entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp_z(aprimez(:)), [N_d1*n2long, N_a, N_semiz, N_e]);
                else
                    entireRHS_L2 = RM_L2;
                end

                [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                V_ford2_jj(:,semizblock,:,d2_c) = shiftdim(Vtemp, 1);
                maxidx_ford2_jj(:,semizblock,:,d2_c) = shiftdim(maxindexL2, 1);

                d1_ind = rem(maxindexL2-1, N_d1) + 1;
                L2offset = ceil(maxindexL2 / N_d1);

                linidx_lower = d1_ind + N_d1*n2long*aind + N_d1*n2long*N_a*semizind + N_d1*n2long*N_a*N_semiz*eind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind + N_d1*n2long*N_a*N_semiz*eind;
                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);

                flag_ford2_jj(:,semizblock,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                allind = d1_ind + N_d1*aind + N_d1*N_a*semizind + N_d1*N_a*N_semiz*eind;
                midpt_ford2_jj(:,semizblock,:,d2_c) = shiftdim(midpoint(allind), 1);
            end
        end

        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_GI_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = extract_policy_GI_no_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, N_d1);
        end
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        maxidx_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        midpt_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        flag_ford2_jj = 2 * ones(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                if has_cont
                    EV_d2z = compute_expected_value_cond(EV, pi_bz(z_c,:)', N_a);
                    EVinterp_z = interp1(a_gridvals, EV_d2z, aprime_grid);
                end

                RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, 1, N_e]);

                if has_cont; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2z, -1); else; entireRHS_L1 = RM_L1; end

                [~, maxindex1] = max(entireRHS_L1, [], 2);
                midpoint = max(min(maxindex1, N_a-1), 2);
                aprimeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;

                a1p_grid_safe = reshape(aprime_grid(aprimeindexes), size(aprimeindexes));

                RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, 1, N_e]);

                if has_cont
                    entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp_z(aprimeindexes(:)), [N_d1*n2long, N_a, 1, N_e]);
                else
                    entireRHS_L2 = RM_L2;
                end

                [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                V_ford2_jj(:,z_c,:,d2_c) = shiftdim(Vtemp, 1);
                maxidx_ford2_jj(:,z_c,:,d2_c) = shiftdim(maxindexL2, 1);

                d1_ind = rem(maxindexL2-1, N_d1) + 1;
                L2offset = ceil(maxindexL2 / N_d1);

                linidx_lower = d1_ind + N_d1*n2long*aind + N_d1*n2long*N_a*eind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*eind;
                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);

                flag_ford2_jj(:,z_c,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                allind = d1_ind + N_d1*aind + N_d1*N_a*eind;
                midpt_ford2_jj(:,z_c,:,d2_c) = shiftdim(midpoint(allind), 1);
            end
        end

        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_GI_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = extract_policy_GI_no_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, N_d1);
        end
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        maxidx_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        midpt_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        flag_ford2_jj = 2 * ones(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
            end

            for z_c = 1:N_z
                semizblock = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, semizblock, j, has_bothz);

                if has_cont
                    EV_d2z = compute_expected_value_cond(EV, pi_bz(semizblock,:)', N_a);
                    EVinterp_z = interp1(a_gridvals, EV_d2z, aprime_grid);
                end

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, N_semiz, 1]);

                    if has_cont; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2z, -1); else; entireRHS_L1 = RM_L1; end

                    [~, maxindex1] = max(entireRHS_L1, [], 2);
                    midpoint = max(min(maxindex1, N_a-1), 2);
                    aprimeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;

                    a1p_grid_safe = reshape(aprime_grid(aprimeindexes), size(aprimeindexes));

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, N_semiz, 1]);

                    if has_cont
                        aprimez = aprimeindexes + n2aprime * semizBind;
                        entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp_z(aprimez(:)), [N_d1*n2long, N_a, N_semiz, 1]);
                    else
                        entireRHS_L2 = RM_L2;
                    end

                    [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                    V_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(Vtemp, 1);
                    maxidx_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(maxindexL2, 1);

                    d1_ind = rem(maxindexL2-1, N_d1) + 1;
                    L2offset = ceil(maxindexL2 / N_d1);

                    linidx_lower = d1_ind + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                    isInfLower = (RM_L2(linidx_lower) == -Inf);
                    isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);

                    flag_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                    allind = d1_ind + N_d1*aind + N_d1*N_a*semizind;
                    midpt_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(midpoint(allind), 1);
                end
            end
        end

        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_GI_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = extract_policy_GI_no_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, N_d1);
        end
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EV = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EV = V_next; end
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        maxidx_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        midpt_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        flag_ford2_jj = 2 * ones(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_cont
                if has_bothz; if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; else; pi_bz = 1; end
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                if has_cont
                    EV_d2z = compute_expected_value_cond(EV, pi_bz(z_c,:)', N_a);
                    EVinterp_z = interp1(a_gridvals, EV_d2z, aprime_grid);
                end

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, 1, 1]);

                    if has_cont; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2z, -1); else; entireRHS_L1 = RM_L1; end

                    [~, maxindex1] = max(entireRHS_L1, [], 2);
                    midpoint = max(min(maxindex1, N_a-1), 2);
                    aprimeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;

                    a1p_grid_safe = reshape(aprime_grid(aprimeindexes), size(aprimeindexes));

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, 1, 1]);

                    if has_cont
                        entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp_z(aprimeindexes(:)), [N_d1*n2long, N_a, 1, 1]);
                    else
                        entireRHS_L2 = RM_L2;
                    end

                    [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                    V_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(Vtemp, 1);
                    maxidx_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(maxindexL2, 1);

                    d1_ind = rem(maxindexL2-1, N_d1) + 1;
                    L2offset = ceil(maxindexL2 / N_d1);

                    linidx_lower = d1_ind + N_d1*n2long*aind;
                    linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                    isInfLower = (RM_L2(linidx_lower) == -Inf);
                    isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);

                    flag_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                    allind = d1_ind + N_d1*aind;
                    midpt_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(midpoint(allind), 1);
                end
            end
        end

        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_GI_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = extract_policy_GI_no_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, N_d1);
        end
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function d_slice = get_d_slice(d12_gridvals, d2_c, has_d2, has_d1)
        if has_d2 && has_d1; d_slice = d12_gridvals(:,:,d2_c); else; d_slice = d12_gridvals; end
    end

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

    function Pol4 = extract_policy_GI_no_d2(maxidx, midpoint, flag, N_d1)
        Pol4 = zeros(4, size(maxidx, 1), size(maxidx, 2), size(maxidx, 3), 'gpuArray');
        Pol4(1,:,:,:) = shiftdim(rem(maxidx(:,:,:,1) - 1, N_d1) + 1, -1); % d
        Pol4(2,:,:,:) = shiftdim(midpoint(:,:,:,1), -1);                  % midpoint
        Pol4(3,:,:,:) = shiftdim(ceil(maxidx(:,:,:,1) / N_d1), -1);       % L2off
        Pol4(4,:,:,:) = shiftdim(flag(:,:,:,1), -1);                      % flag
    end

    function Pol4 = extract_policy_GI_d2(maxidx_ford2, midpoint_ford2, flag_ford2, maxindex_d2, N_a, N_bothz, N_e, N_d1)
        Pol4 = zeros(4, N_a, N_bothz, N_e, 'gpuArray');
        linear_d2 = reshape(maxindex_d2, [N_a * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_d2 - 1);
        maxidx = reshape(maxidx_ford2(lin_idx), [1, N_a, N_bothz, N_e]);
        d1_idx = rem(maxidx - 1, N_d1) + 1;

        d_idx = d1_idx + N_d1 * (shiftdim(maxindex_d2, -1) - 1);

        Pol4(1,:,:,:) = d_idx;
        Pol4(2,:,:,:) = reshape(midpoint_ford2(lin_idx), [1, N_a, N_bothz, N_e]); % midpoint
        Pol4(3,:,:,:) = ceil(maxidx / N_d1);       % L2off
        Pol4(4,:,:,:) = reshape(flag_ford2(lin_idx), [1, N_a, N_bothz, N_e]);     % flag
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
