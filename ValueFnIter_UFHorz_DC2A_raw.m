function [V, Policy] = ValueFnIter_UFHorz_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;
has_d2 = Dims.has_d2; has_d1 = Dims.has_d1;
N_semiz_raw = Dims.N_semiz_raw > 0;

N_a1 = n_a(1); N_a2 = n_a(2:end);
a1_grid = a_grid(1:N_a1); a2_grid = a_grid(N_a1+1:end);

level1ii = round(linspace(1, N_a1, vfoptions.level1n));
level1iidiff = level1ii(2:end) - level1ii(1:end-1) - 1;

V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

% Indexers for refinement steps
a2_col    = gpuArray((0:N_a2-1)'); % Explicit column vector to prevent repelem spreading
a2p_idx   = reshape(gpuArray(0:N_a2-1), 1, 1, N_a2, 1, 1, 1, 1);
bothz_idx = reshape(gpuArray(0:N_bothz-1), 1, 1, 1, 1, 1, N_bothz, 1);
semiz_idx = reshape(gpuArray(0:N_semiz-1), 1, 1, 1, 1, 1, N_semiz, 1);

% --- 2. Determine Iteration Strategy ---
if vfoptions.lowmemory == 0
    IterStrategy = 'Full';
elseif vfoptions.lowmemory == 1
    IterStrategy = 'Loop_e';
elseif vfoptions.lowmemory == 2
    IterStrategy = 'Loop_z_parallel_semiz_e';
elseif vfoptions.lowmemory == 3
    IterStrategy = 'Loop_bothz_e';
end

% --- 3. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]);
        else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    switch IterStrategy
        case 'Full'; [V(:,:,:,j), Pol_j] = solve_period_full(j, V_next);
        case 'Loop_e'; [V(:,:,:,j), Pol_j] = solve_period_loop_e(j, V_next);
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
            EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Policy_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_cont
                if N_semiz_raw; pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                else; pi_bothz = Probs.pi_z_J(:,:,j); end

                EV_expected = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                DiscountedEV_flat = df * reshape(EV_expected, [N_a1 * N_a2 * N_bothz, 1]);
                DiscountedEV = reshape(DiscountedEV_flat, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
            end

            RM_ii = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 1);
            RM_ii = reshape(RM_ii, [N_d1, N_a1, N_a2, vfoptions.level1n, N_a2, N_bothz, N_e]);

            if has_cont; entireRHS_ii = RM_ii + DiscountedEV; else; entireRHS_ii = RM_ii; end

            [~, maxindex1] = max(entireRHS_ii, [], 2);
            min_a1prime = min(min(maxindex1, [], 1), [], 3);
            max_a1prime = max(max(maxindex1, [], 1), [], 3);

            diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,:) - min_a1prime(1,1,1,1:end-1,:,:,:));
            maxgap = squeeze(max(max(max(diff_a1p, [], 7), [], 6), [], 5));

            [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n, N_a2, N_bothz, N_e]), [], 1);

            curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(a2_col, vfoptions.level1n, 1);
            V_ford2_jj(curraindex,:,:,d2_c) = reshape(Vtempii, [vfoptions.level1n * N_a2, N_bothz, N_e]);
            Policy_ford2_jj(curraindex,:,:,d2_c) = reshape(maxindex2, [vfoptions.level1n * N_a2, N_bothz, N_e]);

            for ii = 1:(vfoptions.level1n-1)
                curra_a1_idx = (level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if isempty(curra_a1_idx); continue; end
                L = length(curra_a1_idx);

                loweredge = min(min_a1prime(1,1,1,ii,:,:,:), N_a1-maxgap(ii));

                le_tiled = repmat(loweredge, [1, 1, 1, L, 1, 1, 1]);
                a1p_idx = bsxfun(@plus, le_tiled, reshape(gpuArray(0:maxgap(ii)), 1, maxgap(ii)+1, 1, 1, 1, 1, 1));

                RM = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1_grid(a1p_idx), a2_grid, a1_grid(curra_a1_idx), a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 2);

                if has_cont
                    EV_idx = bsxfun(@plus, a1p_idx, N_a1 * a2p_idx);
                    EV_idx = bsxfun(@plus, EV_idx, N_a1 * N_a2 * bothz_idx);
                    entireRHS = RM + DiscountedEV_flat(EV_idx);
                else
                    entireRHS = RM;
                end

                entireRHS_flat = reshape(entireRHS, [N_d1*(maxgap(ii)+1)*N_a2, L, N_a2, N_bothz, N_e]);
                [Vtempii, maxindex] = max(entireRHS_flat, [], 1);

                dind = rem(maxindex-1, N_d1) + 1;
                a1primeind = rem(ceil(maxindex/N_d1)-1, maxgap(ii)+1) + 1;
                a2primeind = ceil(maxindex/(N_d1*(maxgap(ii)+1)));

                le_flat = reshape(le_tiled, [1, L, N_a2, N_bothz, N_e]);
                a1p_abs = le_flat + a1primeind - 1;
                pol_flat = dind + N_d1 * (a1p_abs - 1) + N_d1 * N_a1 * (a2primeind - 1);

                curraidx_local = repmat(curra_a1_idx, N_a2, 1) + N_a1*repelem(a2_col, L, 1);
                V_ford2_jj(curraidx_local,:,:,d2_c) = reshape(Vtempii, [L * N_a2, N_bothz, N_e]);
                Policy_ford2_jj(curraidx_local,:,:,d2_c) = reshape(pol_flat, [L * N_a2, N_bothz, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2_jj, Policy_ford2_jj);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Policy_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_cont
                if N_semiz_raw; pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                else; pi_bothz = Probs.pi_z_J(:,:,j); end

                EV_expected = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                DiscountedEV_flat = df * reshape(EV_expected, [N_a1 * N_a2 * N_bothz, 1]);
                DiscountedEV = reshape(DiscountedEV_flat, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
            end

            for e_c = 1:N_e
                e_val = Grids.e_gridvals_J(e_c,:,j);
                RM_ii = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 1);
                RM_ii = reshape(RM_ii, [N_d1, N_a1, N_a2, vfoptions.level1n, N_a2, N_bothz, 1]);

                if has_cont; entireRHS_ii = RM_ii + DiscountedEV; else; entireRHS_ii = RM_ii; end

                [~, maxindex1] = max(entireRHS_ii, [], 2);
                min_a1prime = min(min(maxindex1, [], 1), [], 3);
                max_a1prime = max(max(maxindex1, [], 1), [], 3);

                diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,:) - min_a1prime(1,1,1,1:end-1,:,:,:));
                maxgap = squeeze(max(max(diff_a1p, [], 6), [], 5));

                [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n, N_a2, N_bothz]), [], 1);

                curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(a2_col, vfoptions.level1n, 1);
                V_ford2_jj(curraindex,:,e_c,d2_c) = reshape(Vtempii, [vfoptions.level1n * N_a2, N_bothz, 1]);
                Policy_ford2_jj(curraindex,:,e_c,d2_c) = reshape(maxindex2, [vfoptions.level1n * N_a2, N_bothz, 1]);

                for ii = 1:(vfoptions.level1n-1)
                    curra_a1_idx = (level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if isempty(curra_a1_idx); continue; end
                    L = length(curra_a1_idx);

                    loweredge = min(min_a1prime(1,1,1,ii,:,:,:), N_a1-maxgap(ii));

                    le_tiled = repmat(loweredge, [1, 1, 1, L, 1, 1, 1]);
                    a1p_idx = bsxfun(@plus, le_tiled, reshape(gpuArray(0:maxgap(ii)), 1, maxgap(ii)+1, 1, 1, 1, 1, 1));

                    RM = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1_grid(a1p_idx), a2_grid, a1_grid(curra_a1_idx), a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 2);

                    if has_cont
                        EV_idx = bsxfun(@plus, a1p_idx, N_a1 * a2p_idx);
                        EV_idx = bsxfun(@plus, EV_idx, N_a1 * N_a2 * bothz_idx);
                        entireRHS = RM + DiscountedEV_flat(EV_idx);
                    else
                        entireRHS = RM;
                    end

                    entireRHS_flat = reshape(entireRHS, [N_d1*(maxgap(ii)+1)*N_a2, L, N_a2, N_bothz, 1]);
                    [Vtempii, maxindex] = max(entireRHS_flat, [], 1);

                    dind = rem(maxindex-1, N_d1) + 1;
                    a1primeind = rem(ceil(maxindex/N_d1)-1, maxgap(ii)+1) + 1;
                    a2primeind = ceil(maxindex/(N_d1*(maxgap(ii)+1)));

                    le_flat = reshape(le_tiled, [1, L, N_a2, N_bothz, 1]);
                    a1p_abs = le_flat + a1primeind - 1;
                    pol_flat = dind + N_d1 * (a1p_abs - 1) + N_d1 * N_a1 * (a2primeind - 1);

                    curraidx_local = repmat(curra_a1_idx, N_a2, 1) + N_a1*repelem(a2_col, L, 1);
                    V_ford2_jj(curraidx_local,:,e_c,d2_c) = reshape(Vtempii, [L * N_a2, N_bothz, 1]);
                    Policy_ford2_jj(curraidx_local,:,e_c,d2_c) = reshape(pol_flat, [L * N_a2, N_bothz, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2_jj, Policy_ford2_jj);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Policy_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_cont
                if N_semiz_raw; pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                else; pi_bothz = Probs.pi_z_J(:,:,j); end
            end

            for z_c = 1:N_z
                semizblock = (z_c-1)*N_semiz + (1:N_semiz);

                if has_cont
                    EV_d2z = compute_expected_value_cond(EV, pi_bothz(semizblock,:)', N_a);
                    DiscountedEV_flat = df * reshape(EV_d2z, [N_a1 * N_a2 * N_semiz, 1]);
                    DiscountedEV_z = reshape(DiscountedEV_flat, [1, N_a1, N_a2, 1, 1, N_semiz, 1]);
                end

                for e_c = 1:N_e
                    e_val = Grids.e_gridvals_J(e_c,:,j);
                    RM_ii = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 1);
                    RM_ii = reshape(RM_ii, [N_d1, N_a1, N_a2, vfoptions.level1n, N_a2, N_semiz, 1]);

                    if has_cont; entireRHS_ii = RM_ii + DiscountedEV_z; else; entireRHS_ii = RM_ii; end

                    [~, maxindex1] = max(entireRHS_ii, [], 2);
                    min_a1prime = min(min(maxindex1, [], 1), [], 3);
                    max_a1prime = max(max(maxindex1, [], 1), [], 3);

                    diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,:) - min_a1prime(1,1,1,1:end-1,:,:,:));
                    maxgap = squeeze(max(max(diff_a1p, [], 6), [], 5));

                    [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n, N_a2, N_semiz]), [], 1);

                    curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(a2_col, vfoptions.level1n, 1);
                    V_ford2_jj(curraindex,semizblock,e_c,d2_c) = reshape(Vtempii, [vfoptions.level1n * N_a2, N_semiz, 1]);
                    Policy_ford2_jj(curraindex,semizblock,e_c,d2_c) = reshape(maxindex2, [vfoptions.level1n * N_a2, N_semiz, 1]);

                    for ii = 1:(vfoptions.level1n-1)
                        curra_a1_idx = (level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if isempty(curra_a1_idx); continue; end
                        L = length(curra_a1_idx);

                        loweredge = min(min_a1prime(1,1,1,ii,:,:,:), N_a1-maxgap(ii));

                        le_tiled = repmat(loweredge, [1, 1, 1, L, 1, 1, 1]);
                        a1p_idx = bsxfun(@plus, le_tiled, reshape(gpuArray(0:maxgap(ii)), 1, maxgap(ii)+1, 1, 1, 1, 1, 1));

                        RM = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_grid(a1p_idx), a2_grid, a1_grid(curra_a1_idx), a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 2);

                        if has_cont
                            EV_idx = bsxfun(@plus, a1p_idx, N_a1 * a2p_idx);
                            EV_idx = bsxfun(@plus, EV_idx, N_a1 * N_a2 * semiz_idx);
                            entireRHS = RM + DiscountedEV_flat(EV_idx);
                        else
                            entireRHS = RM;
                        end

                        entireRHS_flat = reshape(entireRHS, [N_d1*(maxgap(ii)+1)*N_a2, L, N_a2, N_semiz, 1]);
                        [Vtempii, maxindex] = max(entireRHS_flat, [], 1);

                        dind = rem(maxindex-1, N_d1) + 1;
                        a1primeind = rem(ceil(maxindex/N_d1)-1, maxgap(ii)+1) + 1;
                        a2primeind = ceil(maxindex/(N_d1*(maxgap(ii)+1)));

                        le_flat = reshape(le_tiled, [1, L, N_a2, N_semiz, 1]);
                        a1p_abs = le_flat + a1primeind - 1;
                        pol_flat = dind + N_d1 * (a1p_abs - 1) + N_d1 * N_a1 * (a2primeind - 1);

                        curraidx_local = repmat(curra_a1_idx, N_a2, 1) + N_a1*repelem(a2_col, L, 1);
                        V_ford2_jj(curraidx_local,semizblock,e_c,d2_c) = reshape(Vtempii, [L * N_a2, N_semiz, 1]);
                        Policy_ford2_jj(curraidx_local,semizblock,e_c,d2_c) = reshape(pol_flat, [L * N_a2, N_semiz, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2_jj, Policy_ford2_jj);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Policy_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

            if has_cont
                if N_semiz_raw; pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                else; pi_bothz = Probs.pi_z_J(:,:,j); end
            end

            for z_c = 1:N_bothz
                if has_cont
                    EV_d2z = compute_expected_value_cond(EV, pi_bothz(z_c,:)', N_a);
                    DiscountedEV_flat = df * reshape(EV_d2z, [N_a1 * N_a2, 1]);
                    DiscountedEV_z = reshape(DiscountedEV_flat, [1, N_a1, N_a2, 1, 1, 1, 1]);
                end

                for e_c = 1:N_e
                    e_val = Grids.e_gridvals_J(e_c,:,j);
                    RM_ii = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 1);
                    RM_ii = reshape(RM_ii, [N_d1, N_a1, N_a2, vfoptions.level1n, N_a2, 1, 1]);

                    if has_cont; entireRHS_ii = RM_ii + DiscountedEV_z; else; entireRHS_ii = RM_ii; end

                    [~, maxindex1] = max(entireRHS_ii, [], 2);
                    min_a1prime = min(min(maxindex1, [], 1), [], 3);
                    max_a1prime = max(max(maxindex1, [], 1), [], 3);

                    diff_a1p = max(0, max_a1prime(1,1,1,2:end,:,:,:) - min_a1prime(1,1,1,1:end-1,:,:,:));
                    maxgap = squeeze(max(diff_a1p, [], 5));

                    [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n, N_a2]), [], 1);

                    curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(a2_col, vfoptions.level1n, 1);
                    V_ford2_jj(curraindex,z_c,e_c,d2_c) = reshape(Vtempii, [vfoptions.level1n * N_a2, 1, 1]);
                    Policy_ford2_jj(curraindex,z_c,e_c,d2_c) = reshape(maxindex2, [vfoptions.level1n * N_a2, 1, 1]);

                    for ii = 1:(vfoptions.level1n-1)
                        curra_a1_idx = (level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if isempty(curra_a1_idx); continue; end
                        L = length(curra_a1_idx);

                        loweredge = min(min_a1prime(1,1,1,ii,:,:,:), N_a1-maxgap(ii));

                        le_tiled = repmat(loweredge, [1, 1, 1, L, 1, 1, 1]);
                        a1p_idx = bsxfun(@plus, le_tiled, reshape(gpuArray(0:maxgap(ii)), 1, maxgap(ii)+1, 1, 1, 1, 1, 1));

                        RM = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_grid(a1p_idx), a2_grid, a1_grid(curra_a1_idx), a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 2);

                        if has_cont
                            EV_idx = bsxfun(@plus, a1p_idx, N_a1 * a2p_idx);
                            entireRHS = RM + DiscountedEV_flat(EV_idx);
                        else
                            entireRHS = RM;
                        end

                        entireRHS_flat = reshape(entireRHS, [N_d1*(maxgap(ii)+1)*N_a2, L, N_a2, 1, 1]);
                        [Vtempii, maxindex] = max(entireRHS_flat, [], 1);

                        dind = rem(maxindex-1, N_d1) + 1;
                        a1primeind = rem(ceil(maxindex/N_d1)-1, maxgap(ii)+1) + 1;
                        a2primeind = ceil(maxindex/(N_d1*(maxgap(ii)+1)));

                        le_flat = reshape(le_tiled, [1, L, N_a2, 1, 1]);
                        a1p_abs = le_flat + a1primeind - 1;
                        pol_flat = dind + N_d1 * (a1p_abs - 1) + N_d1 * N_a1 * (a2primeind - 1);

                        curraidx_local = repmat(curra_a1_idx, N_a2, 1) + N_a1*repelem(a2_col, L, 1);
                        V_ford2_jj(curraidx_local,z_c,e_c,d2_c) = reshape(Vtempii, [L * N_a2, 1, 1]);
                        Policy_ford2_jj(curraidx_local,z_c,e_c,d2_c) = reshape(pol_flat, [L * N_a2, 1, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2_jj, Policy_ford2_jj);
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function [V_j, Pol_j] = extract_pol(V_ford2_jj, Policy_ford2_jj)
        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);

        linear_d2 = reshape(maxindex_d2, [N_a * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_d2 - 1);
        d1a1a2_ind = reshape(Policy_ford2_jj(lin_idx), [1, N_a, N_bothz, N_e]);

        d1_idx = rem(d1a1a2_ind - 1, N_d1) + 1;
        a1a2_idx = ceil(d1a1a2_ind / N_d1);
        d2_idx_reshaped = reshape(maxindex_d2, [1, N_a, N_bothz, N_e]);

        Pol_j = d1_idx + (d2_idx_reshaped - 1)*N_d1 + (a1a2_idx - 1)*(N_d1*N_d2);
    end

    function d_slice = get_d_slice(d12_gridvals, d2_c, has_d2, has_d1)
        if has_d2 && has_d1; d_slice = d12_gridvals(:,:,d2_c); else; d_slice = d12_gridvals; end
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
