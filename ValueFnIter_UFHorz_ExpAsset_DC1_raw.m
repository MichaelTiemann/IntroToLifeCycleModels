function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_semiz_raw = Dims.N_semiz_raw > 0;

n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_e = Dims.n_e;

n_d1 = vfoptions.n_d1;
n_d2 = vfoptions.n_d2;
N_d1 = max(prod(n_d1), 1);
N_d2 = max(prod(n_d2), 1);

if isfield(vfoptions, 'n_d3') && prod(vfoptions.n_d3) > 0
    n_d3 = vfoptions.n_d3;
    N_d3 = max(prod(n_d3), 1);
else
    n_d3 = 0;
    N_d3 = 1;
end
N_d_inner = N_d1 * N_d2;
N_d_outer = N_d3;

% --- Divide-and-Conquer Initialization ---
if N_a1 == 0
    error('Cannot use vfoptions.divideandconquer with experience asset if there is no standard endogenous state (N_a1==0)')
end
if ~isfield(vfoptions, 'level1n')
    vfoptions.level1n = floor(sqrt(n_a1(1)));
    if n_a1(1) < 5
        error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable')
    end
end
vfoptions.level1n = min(vfoptions.level1n, n_a1(1));
level1ii = round(linspace(1, n_a1(1), vfoptions.level1n));

% Grid Extracts
a1_gridvals = CreateGridvals(n_a1, vfoptions.a1_grid, 1);
a2_gridvals = CreateGridvals(n_a2, vfoptions.a2_grid, 1);

% --- 2. Introspect ExpAsset Parameters ---
if Dims.exp_flag > 0
    l_d2_len = length(n_d2); if prod(n_d2) == 0; l_d2_len = 0; end
    l_a2_len = length(n_a2); if prod(n_a2) == 0; l_a2_len = 0; end
    l_z_len  = length(n_z);  if prod(n_z) == 0;  l_z_len = 0; end
    l_e_len  = length(n_e);  if prod(n_e) == 0;  l_e_len = 0; end
    l_bothz_len = length(Dims.n_bothz); if prod(Dims.n_bothz) == 0; l_bothz_len = 0; end
    if isfield(vfoptions, 'n_u')
        l_u_len = length(vfoptions.n_u); if prod(vfoptions.n_u) == 0; l_u_len = 0; end
    else; l_u_len = 0; end

    if Dims.exp_flag == 3 || Dims.exp_flag == 4 % e or ze
        num_state_args = l_d2_len + l_a2_len + l_z_len + l_e_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 2 || Dims.exp_flag == 5 % z or semiz
        num_state_args = l_d2_len + l_a2_len + l_bothz_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 6 % experienceassetu
        num_state_args = l_d2_len + l_a2_len + l_u_len + (l_a2_len >= 2);
    else % 1 (experienceasset)
        num_state_args = l_d2_len + l_a2_len + (l_a2_len >= 2);
    end

    temp_args = getAnonymousFnInputNames(vfoptions.aprimeFn);
    if length(temp_args) > num_state_args
        aprimeFnParamNames = temp_args(num_state_args + 1 : end);
    else
        aprimeFnParamNames = {};
    end
else
    aprimeFnParamNames = {};
end

% --- 3. Preallocate Outputs ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

% --- 4. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]);
        else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    has_cont = ~isempty(V_next);
    if has_cont
        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, j);
        d2_gridvals = CreateGridvals(n_d2, vfoptions.d2_grid, 1);

        if Dims.exp_flag == 3 || Dims.exp_flag == 4
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, Grids.z_gridvals_J(:,:,j), n_e, Grids.e_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 2 || Dims.exp_flag == 5
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, Dims.n_bothz, Grids.bothz_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 6
            u_gridvals = CreateGridvals(vfoptions.n_u, vfoptions.u_grid, 1);
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, vfoptions.n_u, u_gridvals);
        else
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2);
        end
        ExpAsset_J.a2primeIndex = a2primeIndex;
        ExpAsset_J.a2primeProbs = a2primeProbs;
    else
        ExpAsset_J = [];
    end

    [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period_full(j, V_next, ExpAsset_J);
end

% =====================================================================
% SOLVE FUNCTION (DC1)
% =====================================================================
    function [V_j, Pol_j] = solve_period_full(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);

            if Dims.exp_flag == 6
                N_u = prod(vfoptions.n_u);
                EV_interp_sum = zeros(N_d2 * N_a1 * N_a2, N_bothz, 'gpuArray');
                for u_c = 1:N_u
                    idx_u = ExpAsset_J.a2primeIndex(:,:,u_c);
                    prob_u = ExpAsset_J.a2primeProbs(:,:,u_c);
                    EV_interp_u = InterpolateExpAssetEV(EVpre, n_a2, N_d2, n_a1, n_a2, N_bothz, idx_u, prob_u);
                    EV_interp_sum = EV_interp_sum + vfoptions.pi_u(u_c) * reshape(EV_interp_u, [N_d2 * N_a1 * N_a2, N_bothz]);
                end
                EV_interp = EV_interp_sum;
            else
                EV_interp = InterpolateExpAssetEV(EVpre, n_a2, N_d2, n_a1, n_a2, N_bothz, ExpAsset_J.a2primeIndex, ExpAsset_J.a2primeProbs);
                EV_interp = reshape(EV_interp, [N_d2 * N_a1 * N_a2, N_bothz]);
            end
        end

        V_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');
        Policy_ford23 = zeros(N_a1, N_a2, N_bothz, N_e, N_d2 * N_d_outer, 'gpuArray');

        % Input Binders natively matching N-D shapes
        a2_idx = reshape(gpuArray(0:1:N_a2-1), 1, 1, 1, N_a2);
        bothz_idx = reshape(gpuArray(0:1:N_bothz-1), 1, 1, 1, 1, N_bothz);
        semiz_idx = reshape(gpuArray(0:1:N_semiz-1), 1, 1, 1, 1, N_semiz);

        % LOW MEMORY 0
        if vfoptions.lowmemory == 0
            for d_outer_c = 1:N_d_outer
                if has_cont
                    if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                    else; pi_bz = Probs.pi_z_J(:,:,j); end
                end

                for d2_c = 1:N_d2
                    d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                    d_val = d_gridvals( (d_outer_c-1)*N_d1*N_d2 + (d2_c-1)*N_d1 + (1:N_d1), : );

                    if has_cont
                        EV_interp_d2 = EV_interp( (d2_c-1)*N_a1*N_a2 + (1:N_a1*N_a2), : );
                        EV_expected = compute_expected_value(EV_interp_d2, pi_bz);
                        DiscountedEV_flat = df * reshape(EV_expected, [N_a1 * N_a2 * N_bothz, 1]);
                        DiscountedEV = reshape(DiscountedEV_flat, [1, N_a1, 1, N_a2, N_bothz, 1]);
                    end

                    RM_ii = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 1);
                    RM_ii = reshape(RM_ii, [N_d1, N_a1, vfoptions.level1n, N_a2, N_bothz, N_e]);

                    if has_cont; entireRHS_ii = RM_ii + DiscountedEV; else; entireRHS_ii = RM_ii; end

                    [~, maxindex1] = max(entireRHS_ii, [], 2);
                    [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1, vfoptions.level1n, N_a2, N_bothz, N_e]), [], 1);

                    V_ford23(level1ii,:,:,:,d23_idx) = shiftdim(Vtempii, 1);
                    Policy_ford23(level1ii,:,:,:,d23_idx) = shiftdim(maxindex2, 1);

                    maxgap = squeeze(max(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],2),[],1));

                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        if isempty(curraindex); continue; end
                        L = length(curraindex);

                        loweredge = min(min(maxindex1(:,1,ii,:,:,:), [], 1), N_a1-maxgap(ii));

                        % Cleanly Flatten N-D matrices to 2D exactly for RefineSearch routing
                        base_idx = bsxfun(@plus, loweredge, N_a1*a2_idx + N_a1*N_a2*bothz_idx);
                        base_idx = repmat(base_idx, [1, 1, L, 1, 1, 1]);
                        n_states = numel(base_idx);

                        base_flat = reshape(base_idx, 1, n_states);
                        le_flat = reshape(repmat(loweredge, [1, 1, L, 1, 1, 1]), 1, n_states);

                        aprimez = bsxfun(@plus, base_flat, (0:maxgap(ii))'); % [maxgap+1, n_states]
                        reshape_size = [N_d1*(maxgap(ii)+1), 1, L, N_a2, N_bothz, N_e];

                        ReturnFnHandle = @(ap) reshape( ...
                            CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                            reshape(a1_gridvals(ap,:), reshape_size), a1_gridvals(curraindex,:), a2_gridvals, ...
                            Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 2), ...
                            [N_d1*(maxgap(ii)+1), n_states] ...
                            );

                        if has_cont
                            EV_slice = DiscountedEV_flat(aprimez);
                            [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, le_flat, maxgap(ii), N_d1, reshape_size, EV_slice);
                        else
                            [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, le_flat, maxgap(ii), N_d1, reshape_size);
                        end

                        V_ford23(curraindex,:,:,:,d23_idx) = reshape(Vtempii, [L, N_a2, N_bothz, N_e]);
                        pol_flat = dind + (le_flat + maxindex - 2) * N_d1;
                        Policy_ford23(curraindex,:,:,:,d23_idx) = reshape(pol_flat, [L, N_a2, N_bothz, N_e]);
                    end
                end
            end

            % LOW MEMORY 1
        elseif vfoptions.lowmemory == 1
            for d_outer_c = 1:N_d_outer
                if has_cont
                    if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                    else; pi_bz = Probs.pi_z_J(:,:,j); end
                end

                for d2_c = 1:N_d2
                    d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                    d_val = d_gridvals( (d_outer_c-1)*N_d1*N_d2 + (d2_c-1)*N_d1 + (1:N_d1), : );

                    if has_cont
                        EV_interp_d2 = EV_interp( (d2_c-1)*N_a1*N_a2 + (1:N_a1*N_a2), : );
                        EV_expected = compute_expected_value(EV_interp_d2, pi_bz);
                        DiscountedEV_flat = df * reshape(EV_expected, [N_a1 * N_a2 * N_bothz, 1]);
                        DiscountedEV = reshape(DiscountedEV_flat, [1, N_a1, 1, N_a2, N_bothz, 1]);
                    end

                    for e_c = 1:N_e
                        RM_ii = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 1);
                        RM_ii = reshape(RM_ii, [N_d1, N_a1, vfoptions.level1n, N_a2, N_bothz, 1]);

                        if has_cont; entireRHS_ii = RM_ii + DiscountedEV; else; entireRHS_ii = RM_ii; end

                        [~, maxindex1] = max(entireRHS_ii, [], 2);
                        [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1, vfoptions.level1n, N_a2, N_bothz]), [], 1);

                        V_ford23(level1ii,:,:,e_c,d23_idx) = shiftdim(Vtempii, 1);
                        Policy_ford23(level1ii,:,:,e_c,d23_idx) = shiftdim(maxindex2, 1);

                        maxgap = squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],5),[],4),[],2),[],1));

                        for ii = 1:(vfoptions.level1n-1)
                            curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                            if isempty(curraindex); continue; end
                            L = length(curraindex);

                            loweredge = min(min(maxindex1(:,1,ii,:,:,:), [], 1), N_a1-maxgap(ii));

                            base_idx = bsxfun(@plus, loweredge, N_a1*a2_idx + N_a1*N_a2*bothz_idx);
                            base_idx = repmat(base_idx, [1, 1, L, 1, 1, 1]);
                            n_states = numel(base_idx);

                            base_flat = reshape(base_idx, 1, n_states);
                            le_flat = reshape(repmat(loweredge, [1, 1, L, 1, 1, 1]), 1, n_states);

                            aprimez = bsxfun(@plus, base_flat, (0:maxgap(ii))');
                            reshape_size = [N_d1*(maxgap(ii)+1), 1, L, N_a2, N_bothz, 1];

                            ReturnFnHandle = @(ap) reshape( ...
                                CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                                reshape(a1_gridvals(ap,:), reshape_size), a1_gridvals(curraindex,:), a2_gridvals, ...
                                Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 2), ...
                                [N_d1*(maxgap(ii)+1), n_states] ...
                                );

                            if has_cont
                                EV_slice = DiscountedEV_flat(aprimez);
                                [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, le_flat, maxgap(ii), N_d1, reshape_size, EV_slice);
                            else
                                [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, le_flat, maxgap(ii), N_d1, reshape_size);
                            end

                            V_ford23(curraindex,:,:,e_c,d23_idx) = reshape(Vtempii, [L, N_a2, N_bothz, 1]);
                            pol_flat = dind + (le_flat + maxindex - 2) * N_d1;
                            Policy_ford23(curraindex,:,:,e_c,d23_idx) = reshape(pol_flat, [L, N_a2, N_bothz, 1]);
                        end
                    end
                end
            end

            % LOW MEMORY 2
        elseif vfoptions.lowmemory == 2
            for d_outer_c = 1:N_d_outer
                if has_cont
                    if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                    else; pi_bz = Probs.pi_z_J(:,:,j); end
                end

                for d2_c = 1:N_d2
                    d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                    d_val = d_gridvals( (d_outer_c-1)*N_d1*N_d2 + (d2_c-1)*N_d1 + (1:N_d1), : );

                    for z_c = 1:N_z
                        semizblock = (z_c-1)*N_semiz + (1:N_semiz);

                        if has_cont
                            EV_interp_d2 = EV_interp( (d2_c-1)*N_a1*N_a2 + (1:N_a1*N_a2), : );
                            EV_expected = compute_expected_value_cond(EV_interp_d2, pi_bz(semizblock,:)', N_a1*N_a2);
                            DiscountedEV_flat = df * reshape(EV_expected, [N_a1 * N_a2 * N_semiz, 1]);
                            DiscountedEV_z = reshape(DiscountedEV_flat, [1, N_a1, 1, N_a2, N_semiz, 1]);
                        end

                        for e_c = 1:N_e
                            RM_ii = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, Grids.bothz_gridvals_J(semizblock,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 1);
                            RM_ii = reshape(RM_ii, [N_d1, N_a1, vfoptions.level1n, N_a2, N_semiz, 1]);

                            if has_cont; entireRHS_ii = RM_ii + DiscountedEV_z; else; entireRHS_ii = RM_ii; end

                            [~, maxindex1] = max(entireRHS_ii, [], 2);
                            [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1, vfoptions.level1n, N_a2, N_semiz]), [], 1);

                            V_ford23(level1ii,:,semizblock,e_c,d23_idx) = shiftdim(Vtempii, 1);
                            Policy_ford23(level1ii,:,semizblock,e_c,d23_idx) = shiftdim(maxindex2, 1);

                            maxgap = squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],5),[],4),[],2),[],1));

                            for ii = 1:(vfoptions.level1n-1)
                                curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                                if isempty(curraindex); continue; end
                                L = length(curraindex);

                                loweredge = min(min(maxindex1(:,1,ii,:,:,:), [], 1), N_a1-maxgap(ii));

                                base_idx = bsxfun(@plus, loweredge, N_a1*a2_idx + N_a1*N_a2*semiz_idx);
                                base_idx = repmat(base_idx, [1, 1, L, 1, 1, 1]);
                                n_states = numel(base_idx);

                                base_flat = reshape(base_idx, 1, n_states);
                                le_flat = reshape(repmat(loweredge, [1, 1, L, 1, 1, 1]), 1, n_states);

                                aprimez = bsxfun(@plus, base_flat, (0:maxgap(ii))');
                                reshape_size = [N_d1*(maxgap(ii)+1), 1, L, N_a2, N_semiz, 1];

                                ReturnFnHandle = @(ap) reshape( ...
                                    CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, ...
                                    reshape(a1_gridvals(ap,:), reshape_size), a1_gridvals(curraindex,:), a2_gridvals, ...
                                    Grids.bothz_gridvals_J(semizblock,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 2), ...
                                    [N_d1*(maxgap(ii)+1), n_states] ...
                                    );

                                if has_cont
                                    EV_slice = DiscountedEV_flat(aprimez);
                                    [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, le_flat, maxgap(ii), N_d1, reshape_size, EV_slice);
                                else
                                    [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, le_flat, maxgap(ii), N_d1, reshape_size);
                                end

                                V_ford23(curraindex,:,semizblock,e_c,d23_idx) = reshape(Vtempii, [L, N_a2, N_semiz, 1]);
                                pol_flat = dind + (le_flat + maxindex - 2) * N_d1;
                                Policy_ford23(curraindex,:,semizblock,e_c,d23_idx) = reshape(pol_flat, [L, N_a2, N_semiz, 1]);
                            end
                        end
                    end
                end
            end

            % LOW MEMORY 3
        elseif vfoptions.lowmemory == 3
            for d_outer_c = 1:N_d_outer
                if has_cont
                    if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                    else; pi_bz = Probs.pi_z_J(:,:,j); end
                end

                for d2_c = 1:N_d2
                    d23_idx = (d_outer_c - 1) * N_d2 + d2_c;
                    d_val = d_gridvals( (d_outer_c-1)*N_d1*N_d2 + (d2_c-1)*N_d1 + (1:N_d1), : );

                    for z_c = 1:N_bothz
                        if has_cont
                            EV_interp_d2 = EV_interp( (d2_c-1)*N_a1*N_a2 + (1:N_a1*N_a2), : );
                            EV_expected = compute_expected_value_cond(EV_interp_d2, pi_bz(z_c,:)', N_a1*N_a2);
                            DiscountedEV_flat = df * reshape(EV_expected, [N_a1 * N_a2, 1]);
                            DiscountedEV_z = reshape(DiscountedEV_flat, [1, N_a1, 1, N_a2, 1, 1]);
                        end

                        for e_c = 1:N_e
                            RM_ii = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii,:), a2_gridvals, Grids.bothz_gridvals_J(z_c,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 1);
                            RM_ii = reshape(RM_ii, [N_d1, N_a1, vfoptions.level1n, N_a2, 1, 1]);

                            if has_cont; entireRHS_ii = RM_ii + DiscountedEV_z; else; entireRHS_ii = RM_ii; end

                            [~, maxindex1] = max(entireRHS_ii, [], 2);
                            [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1, vfoptions.level1n, N_a2]), [], 1);

                            V_ford23(level1ii,:,z_c,e_c,d23_idx) = shiftdim(Vtempii, 1);
                            Policy_ford23(level1ii,:,z_c,e_c,d23_idx) = shiftdim(maxindex2, 1);

                            maxgap = squeeze(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],4),[],2),[],1));

                            for ii = 1:(vfoptions.level1n-1)
                                curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                                if isempty(curraindex); continue; end
                                L = length(curraindex);

                                loweredge = min(min(maxindex1(:,1,ii,:,:,:), [], 1), N_a1-maxgap(ii));

                                base_idx = bsxfun(@plus, loweredge, N_a1*a2_idx);
                                base_idx = repmat(base_idx, [1, 1, L, 1, 1, 1]);
                                n_states = numel(base_idx);

                                base_flat = reshape(base_idx, 1, n_states);
                                le_flat = reshape(repmat(loweredge, [1, 1, L, 1, 1, 1]), 1, n_states);

                                aprimez = bsxfun(@plus, base_flat, (0:maxgap(ii))');
                                reshape_size = [N_d1*(maxgap(ii)+1), 1, L, N_a2, 1, 1];

                                ReturnFnHandle = @(ap) reshape( ...
                                    CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, ...
                                    reshape(a1_gridvals(ap,:), reshape_size), a1_gridvals(curraindex,:), a2_gridvals, ...
                                    Grids.bothz_gridvals_J(z_c,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 2), ...
                                    [N_d1*(maxgap(ii)+1), n_states] ...
                                    );

                                if has_cont
                                    EV_slice = DiscountedEV_flat(aprimez);
                                    [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, le_flat, maxgap(ii), N_d1, reshape_size, EV_slice);
                                else
                                    [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, le_flat, maxgap(ii), N_d1, reshape_size);
                                end

                                V_ford23(curraindex,:,z_c,e_c,d23_idx) = reshape(Vtempii, [L, N_a2, 1, 1]);
                                pol_flat = dind + (le_flat + maxindex - 2) * N_d1;
                                Policy_ford23(curraindex,:,z_c,e_c,d23_idx) = reshape(pol_flat, [L, N_a2, 1, 1]);
                            end
                        end
                    end
                end
            end
        end

        % Max over (d2, d3) combinations
        [V_j, maxindex_d23] = max(reshape(V_ford23, [N_a1 * N_a2, N_bothz, N_e, N_d2 * N_d_outer]), [], 4);
        V_j = reshape(V_j, [N_a1 * N_a2, N_bothz, N_e]);

        % Reconstruct Kronecker Index
        linear_d23 = reshape(maxindex_d23, [N_a1 * N_a2 * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a1 * N_a2 * N_bothz * N_e))' + (N_a1 * N_a2 * N_bothz * N_e) * (linear_d23 - 1);
        inner_pol = reshape(Policy_ford23(lin_idx), [1, N_a1 * N_a2, N_bothz, N_e]);

        d1_idx = rem(inner_pol - 1, N_d1) + 1;
        a1prime_idx = ceil(inner_pol / N_d1);
        d23_idx_reshaped = reshape(maxindex_d23, [1, N_a1 * N_a2, N_bothz, N_e]);

        Pol_j = d1_idx + (d23_idx_reshaped - 1)*N_d1 + (a1prime_idx - 1)*(N_d1*N_d2*N_d_outer);
    end

    function EV_out = compute_expected_value(EV_in, pi_slice)
        EVinf = (EV_in == -Inf);
        EV_val = EV_in;
        EV_val(EVinf) = -1e250;
        EV_out = EV_val * pi_slice';
        EV_out(EVinf * (pi_slice' > 0) > 0) = -Inf;
    end

    function EV_out = compute_expected_value_cond(EV, pi_slice, N_state)
        EV_val = EV .* shiftdim(pi_slice, -1);
        EV_val(isnan(EV_val)) = 0;
        EV_out = sum(EV_val, 2);
    end


end
