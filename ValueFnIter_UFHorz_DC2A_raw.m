function [V, Policy] = ValueFnIter_UFHorz_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;
has_d2 = Dims.has_d2; has_d1 = Dims.has_d1; N_semiz_raw = Dims.N_semiz_raw;

N_a1 = n_a(1); N_a2 = n_a(2:end);
a1_grid = a_grid(1:N_a1); a2_grid = a_grid(N_a1+1:end);

level1ii = round(linspace(1, N_a1, vfoptions.level1n));
level1iidiff = level1ii(2:end) - level1ii(1:end-1) - 1;

V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

% Indexers for refinement steps (Using row vectors cleanly shifted to state dims)
a2_idx    = reshape(gpuArray(0:1:N_a2-1), 1, 1, N_a2);
bothz_idx = reshape(gpuArray(0:1:N_bothz-1), 1, 1, 1, N_bothz);
semiz_idx = reshape(gpuArray(0:1:N_semiz-1), 1, 1, 1, N_semiz);

a2p_choice = reshape(gpuArray(0:1:N_a2-1), 1, 1, N_a2);

% --- 2. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]);
        else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period(j, V_next);
end

% =====================================================================
% HELPER FUNCTION: Solves a single period j
% =====================================================================
    function [V_j, Pol_j] = solve_period(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Policy_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        % LOW MEMORY 0
        if vfoptions.lowmemory == 0
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                if has_cont
                    EV_expected = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                    DiscountedEV_flat = df * reshape(EV_expected, [N_a1 * N_a2 * N_bothz, 1]);
                    DiscountedEV = reshape(DiscountedEV_flat, [1, N_a1, N_a2, 1, N_a2, N_bothz, 1]);
                end

                RM_ii = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 1);
                RM_ii = reshape(RM_ii, [N_d1, N_a1, N_a2, vfoptions.level1n, N_a2, N_bothz, N_e]);

                if has_cont; entireRHS_ii = RM_ii + DiscountedEV; else; entireRHS_ii = RM_ii; end

                [~, maxindex1] = max(entireRHS_ii, [], 2);
                [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n, N_a2, N_bothz, N_e]), [], 1);

                curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(squeeze(a2_idx)', vfoptions.level1n, 1);
                V_ford2_jj(curraindex,:,:,d2_c) = shiftdim(Vtempii, 1);
                Policy_ford2_jj(curraindex,:,:,d2_c) = shiftdim(maxindex2, 1);

                maxgap = squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));

                for ii = 1:(vfoptions.level1n-1)
                    curraidx_local = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)', N_a2, 1) + N_a1*repelem(squeeze(a2_idx)', level1iidiff(ii), 1);
                    if isempty(curraidx_local); continue; end
                    L = length(curraidx_local);

                    loweredge = min(min(maxindex1(:,1,:,ii,:,:,:), [], 1), N_a1-maxgap(ii));

                    % 1. Create the base indices for the current states
                    base_idx = bsxfun(@plus, loweredge, N_a1*a2_idx + N_a*bothz_idx);
                    base_idx = repmat(base_idx, [1, L, 1, 1, 1]);
                    n_states = numel(base_idx);

                    base_flat = reshape(base_idx, 1, n_states);
                    le_flat = reshape(repmat(loweredge, [1, L, 1, 1, 1]), 1, n_states);

                    % 2. Choice Dimension 1: a1prime
                    a1p = bsxfun(@plus, base_flat, (0:maxgap(ii))'); % [maxgap+1, n_states]

                    % 3. Choice Dimension 2: a2prime
                    EV_idx_3d = bsxfun(@plus, a1p, N_a1 * a2p_choice); % [maxgap+1, n_states, N_a2]
                    EV_idx_3d = permute(EV_idx_3d, [1, 3, 2]); % Align to [a1p, a2p, states]
                    EV_idx_flat = reshape(EV_idx_3d, [(maxgap(ii)+1)*N_a2, n_states]);

                    if N_d1 > 1
                        a1p_rep = kron(a1p, ones(N_d1, 1));
                        EV_idx_flat = kron(EV_idx_flat, ones(N_d1, 1));
                    else
                        a1p_rep = a1p;
                    end

                    RM = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1_grid(a1p_rep), a2_grid, a1_grid(curraidx_local), a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 2);
                    entireRHS = reshape(RM, [N_d1*(maxgap(ii)+1)*N_a2, n_states]);

                    if has_cont
                        entireRHS = entireRHS + DiscountedEV_flat(EV_idx_flat);
                    end

                    [Vtempii, maxindex] = max(entireRHS, [], 1);

                    dind = rem(maxindex-1, N_d1) + 1;
                    a1primeind = rem(ceil(maxindex/N_d1)-1, maxgap(ii)+1);
                    a2primeind = ceil(maxindex/(N_d1*(maxgap(ii)+1))) - 1;

                    V_ford2_jj(curraidx_local,:,:,d2_c) = reshape(Vtempii, [L, N_bothz, N_e]);
                    pol_flat = dind + N_d1 * (le_flat + a1primeind - 1) + N_d1 * N_a1 * a2primeind;
                    Policy_ford2_jj(curraidx_local,:,:,d2_c) = reshape(pol_flat, [L, N_bothz, N_e]);
                end
            end

            % LOW MEMORY 1
        elseif vfoptions.lowmemory == 1
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                if has_cont
                    EV_expected = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                    DiscountedEV_flat = df * reshape(EV_expected, [N_a1 * N_a2 * N_bothz, 1]);
                    DiscountedEV = reshape(DiscountedEV_flat, [1, N_a1, N_a2, 1, N_a2, N_bothz, 1]);
                end

                for e_c = 1:N_e
                    e_val = Grids.e_gridvals_J(e_c,:,j);
                    RM_ii = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 1);
                    RM_ii = reshape(RM_ii, [N_d1, N_a1, N_a2, vfoptions.level1n, N_a2, N_bothz, 1]);

                    if has_cont; entireRHS_ii = RM_ii + DiscountedEV; else; entireRHS_ii = RM_ii; end

                    [~, maxindex1] = max(entireRHS_ii, [], 2);
                    [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n, N_a2, N_bothz]), [], 1);

                    curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(squeeze(a2_idx)', vfoptions.level1n, 1);
                    V_ford2_jj(curraindex,:,e_c,d2_c) = shiftdim(Vtempii, 1);
                    Policy_ford2_jj(curraindex,:,e_c,d2_c) = shiftdim(maxindex2, 1);

                    maxgap = squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));

                    for ii = 1:(vfoptions.level1n-1)
                        curraidx_local = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)', N_a2, 1) + N_a1*repelem(squeeze(a2_idx)', level1iidiff(ii), 1);
                        if isempty(curraidx_local); continue; end
                        L = length(curraidx_local);

                        loweredge = min(min(maxindex1(:,1,:,ii,:,:), [], 1), N_a1-maxgap(ii));

                        base_idx = bsxfun(@plus, loweredge, N_a1*a2_idx + N_a*bothz_idx);
                        base_idx = repmat(base_idx, [1, L, 1, 1]);
                        n_states = numel(base_idx);

                        base_flat = reshape(base_idx, 1, n_states);
                        le_flat = reshape(repmat(loweredge, [1, L, 1, 1]), 1, n_states);

                        a1p = bsxfun(@plus, base_flat, (0:maxgap(ii))');

                        EV_idx_3d = bsxfun(@plus, a1p, N_a1 * a2p_choice);
                        EV_idx_3d = permute(EV_idx_3d, [1, 3, 2]);
                        EV_idx_flat = reshape(EV_idx_3d, [(maxgap(ii)+1)*N_a2, n_states]);

                        if N_d1 > 1
                            a1p_rep = kron(a1p, ones(N_d1, 1));
                            EV_idx_flat = kron(EV_idx_flat, ones(N_d1, 1));
                        else
                            a1p_rep = a1p;
                        end

                        RM = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1_grid(a1p_rep), a2_grid, a1_grid(curraidx_local), a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 2);
                        entireRHS = reshape(RM, [N_d1*(maxgap(ii)+1)*N_a2, n_states]);

                        if has_cont
                            entireRHS = entireRHS + DiscountedEV_flat(EV_idx_flat);
                        end

                        [Vtempii, maxindex] = max(entireRHS, [], 1);

                        dind = rem(maxindex-1, N_d1) + 1;
                        a1primeind = rem(ceil(maxindex/N_d1)-1, maxgap(ii)+1);
                        a2primeind = ceil(maxindex/(N_d1*(maxgap(ii)+1))) - 1;

                        V_ford2_jj(curraidx_local,:,e_c,d2_c) = reshape(Vtempii, [L, N_bothz, 1]);
                        pol_flat = dind + N_d1 * (le_flat + a1primeind - 1) + N_d1 * N_a1 * a2primeind;
                        Policy_ford2_jj(curraidx_local,:,e_c,d2_c) = reshape(pol_flat, [L, N_bothz, 1]);
                    end
                end
            end

            % LOW MEMORY 2
        elseif vfoptions.lowmemory == 2
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                for z_c = 1:N_z
                    semizblock = (z_c-1)*N_semiz + (1:N_semiz);

                    if has_cont
                        EV_d2z = compute_expected_value_cond(EV, pi_bothz(semizblock,:)', N_a);
                        DiscountedEV_flat = df * reshape(EV_d2z, [N_a1 * N_a2 * N_semiz, 1]);
                        DiscountedEV_z = reshape(DiscountedEV_flat, [1, N_a1, N_a2, 1, N_a2, N_semiz, 1]);
                    end

                    for e_c = 1:N_e
                        e_val = Grids.e_gridvals_J(e_c,:,j);
                        RM_ii = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 1);
                        RM_ii = reshape(RM_ii, [N_d1, N_a1, N_a2, vfoptions.level1n, N_a2, N_semiz, 1]);

                        if has_cont; entireRHS_ii = RM_ii + DiscountedEV_z; else; entireRHS_ii = RM_ii; end

                        [~, maxindex1] = max(entireRHS_ii, [], 2);
                        [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n, N_a2, N_semiz]), [], 1);

                        curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(squeeze(a2_idx)', vfoptions.level1n, 1);
                        V_ford2_jj(curraindex,semizblock,e_c,d2_c) = shiftdim(Vtempii, 1);
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c) = shiftdim(maxindex2, 1);

                        maxgap = squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));

                        for ii = 1:(vfoptions.level1n-1)
                            curraidx_local = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)', N_a2, 1) + N_a1*repelem(squeeze(a2_idx)', level1iidiff(ii), 1);
                            if isempty(curraidx_local); continue; end
                            L = length(curraidx_local);

                            loweredge = min(min(maxindex1(:,1,:,ii,:), [], 1), N_a1-maxgap(ii));

                            base_idx = bsxfun(@plus, loweredge, N_a1*a2_idx + N_a*semiz_idx);
                            base_idx = repmat(base_idx, [1, L, 1, 1]);
                            n_states = numel(base_idx);

                            base_flat = reshape(base_idx, 1, n_states);
                            le_flat = reshape(repmat(loweredge, [1, L, 1, 1]), 1, n_states);

                            a1p = bsxfun(@plus, base_flat, (0:maxgap(ii))');

                            EV_idx_3d = bsxfun(@plus, a1p, N_a1 * a2p_choice);
                            EV_idx_3d = permute(EV_idx_3d, [1, 3, 2]);
                            EV_idx_flat = reshape(EV_idx_3d, [(maxgap(ii)+1)*N_a2, n_states]);

                            if N_d1 > 1
                                a1p_rep = kron(a1p, ones(N_d1, 1));
                                EV_idx_flat = kron(EV_idx_flat, ones(N_d1, 1));
                            else
                                a1p_rep = a1p;
                            end

                            RM = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_grid(a1p_rep), a2_grid, a1_grid(curraidx_local), a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 2);
                            entireRHS = reshape(RM, [N_d1*(maxgap(ii)+1)*N_a2, n_states]);

                            if has_cont
                                entireRHS = entireRHS + DiscountedEV_flat(EV_idx_flat);
                            end

                            [Vtempii, maxindex] = max(entireRHS, [], 1);

                            dind = rem(maxindex-1, N_d1) + 1;
                            a1primeind = rem(ceil(maxindex/N_d1)-1, maxgap(ii)+1);
                            a2primeind = ceil(maxindex/(N_d1*(maxgap(ii)+1))) - 1;

                            V_ford2_jj(curraidx_local,semizblock,e_c,d2_c) = reshape(Vtempii, [L, N_semiz, 1]);
                            pol_flat = dind + N_d1 * (le_flat + a1primeind - 1) + N_d1 * N_a1 * a2primeind;
                            Policy_ford2_jj(curraidx_local,semizblock,e_c,d2_c) = reshape(pol_flat, [L, N_semiz, 1]);
                        end
                    end
                end
            end

            % LOW MEMORY 3
        elseif vfoptions.lowmemory == 3
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                for z_c = 1:N_bothz
                    if has_cont
                        EV_d2z = compute_expected_value_cond(EV, pi_bothz(z_c,:)', N_a);
                        DiscountedEV_flat = df * reshape(EV_d2z, [N_a1 * N_a2, 1]);
                        DiscountedEV_z = reshape(DiscountedEV_flat, [1, N_a1, N_a2, 1, N_a2, 1, 1]);
                    end

                    for e_c = 1:N_e
                        e_val = Grids.e_gridvals_J(e_c,:,j);
                        RM_ii = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 1);
                        RM_ii = reshape(RM_ii, [N_d1, N_a1, N_a2, vfoptions.level1n, N_a2, 1, 1]);

                        if has_cont; entireRHS_ii = RM_ii + DiscountedEV_z; else; entireRHS_ii = RM_ii; end

                        [~, maxindex1] = max(entireRHS_ii, [], 2);
                        [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n, N_a2]), [], 1);

                        curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(squeeze(a2_idx)', vfoptions.level1n, 1);
                        V_ford2_jj(curraindex,z_c,e_c,d2_c) = shiftdim(Vtempii, 1);
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c) = shiftdim(maxindex2, 1);

                        maxgap = squeeze(max(max(maxindex1(:,1,:,2:end)-maxindex1(:,1,:,1:end-1),[],3),[],1));

                        for ii = 1:(vfoptions.level1n-1)
                            curraidx_local = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)', N_a2, 1) + N_a1*repelem(squeeze(a2_idx)', level1iidiff(ii), 1);
                            if isempty(curraidx_local); continue; end
                            L = length(curraidx_local);

                            loweredge = min(min(maxindex1(:,1,:,ii), [], 1), N_a1-maxgap(ii));

                            base_idx = bsxfun(@plus, loweredge, N_a1*a2_idx);
                            base_idx = repmat(base_idx, [1, L, 1]);
                            n_states = numel(base_idx);

                            base_flat = reshape(base_idx, 1, n_states);
                            le_flat = reshape(repmat(loweredge, [1, L, 1]), 1, n_states);

                            a1p = bsxfun(@plus, base_flat, (0:maxgap(ii))');

                            EV_idx_3d = bsxfun(@plus, a1p, N_a1 * a2p_choice);
                            EV_idx_3d = permute(EV_idx_3d, [1, 3, 2]);
                            EV_idx_flat = reshape(EV_idx_3d, [(maxgap(ii)+1)*N_a2, n_states]);

                            if N_d1 > 1
                                a1p_rep = kron(a1p, ones(N_d1, 1));
                                EV_idx_flat = kron(EV_idx_flat, ones(N_d1, 1));
                            else
                                a1p_rep = a1p;
                            end

                            RM = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_grid(a1p_rep), a2_grid, a1_grid(curraidx_local), a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 2);
                            entireRHS = reshape(RM, [N_d1*(maxgap(ii)+1)*N_a2, n_states]);

                            if has_cont
                                entireRHS = entireRHS + DiscountedEV_flat(EV_idx_flat);
                            end

                            [Vtempii, maxindex] = max(entireRHS, [], 1);

                            dind = rem(maxindex-1, N_d1) + 1;
                            a1primeind = rem(ceil(maxindex/N_d1)-1, maxgap(ii)+1);
                            a2primeind = ceil(maxindex/(N_d1*(maxgap(ii)+1))) - 1;

                            V_ford2_jj(curraidx_local,z_c,e_c,d2_c) = reshape(Vtempii, [L, 1, 1]);
                            pol_flat = dind + N_d1 * (le_flat + a1primeind - 1) + N_d1 * N_a1 * a2primeind;
                            Policy_ford2_jj(curraidx_local,z_c,e_c,d2_c) = reshape(pol_flat, [L, 1, 1]);
                        end
                    end
                end
            end
        end

        % Resolve max over d2 into a single unified Kronecker Policy index
        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);

        linear_d2 = reshape(maxindex_d2, [N_a * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_d2 - 1);
        d1a1a2_ind = reshape(Policy_ford2_jj(lin_idx), [1, N_a, N_bothz, N_e]);

        d1_idx = rem(d1a1a2_ind - 1, N_d1) + 1;
        a1a2_idx = ceil(d1a1a2_ind / N_d1);
        d2_idx_reshaped = reshape(maxindex_d2, [1, N_a, N_bothz, N_e]);

        Pol_j = d1_idx + (d2_idx_reshaped - 1)*N_d1 + (a1a2_idx - 1)*(N_d1*N_d2);
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
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