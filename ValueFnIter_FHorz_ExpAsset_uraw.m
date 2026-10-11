function [V, Policy] = ValueFnIter_FHorz_ExpAsset_uraw(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_uraw_FHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_z_raw = Dims.special_n_z(1) > 0;
N_semiz_raw = Dims.N_semiz_raw > 0;
N_e_raw = Dims.N_e_raw > 0;

n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_e = Dims.n_e;

% ExpAsset hierarchy defined by SetupNonStandardEndoStates_FHorz
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

% Grid Extracts
a1_gridvals = CreateGridvals(n_a1, vfoptions.a1_grid, 1);
a2_gridvals = CreateGridvals(n_a2, vfoptions.a2_grid, 1);

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

% --- 3. Introspect ExpAsset Parameters ---
if Dims.exp_flag > 0
    l_d2_len = length(n_d2); if prod(n_d2) == 0; l_d2_len = 0; end
    l_a2_len = length(n_a2); if prod(n_a2) == 0; l_a2_len = 0; end
    l_z_len  = length(n_z);  if prod(n_z) == 0;  l_z_len = 0; end
    l_e_len  = length(n_e);  if prod(n_e) == 0;  l_e_len = 0; end
    l_bothz_len = length(Dims.n_bothz); if prod(Dims.n_bothz) == 0; l_bothz_len = 0; end

    % Calculate expected state arguments based on the active variant
    if Dims.exp_flag == 3 || Dims.exp_flag == 4 % e or ze
        num_state_args = l_d2_len + l_a2_len + l_z_len + l_e_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 2 || Dims.exp_flag == 5 % z or semiz
        num_state_args = l_d2_len + l_a2_len + l_bothz_len + (l_a2_len >= 2);
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

% --- 4. Preallocate Outputs ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

% --- 5. Backward Induction Loop ---
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

        if Dims.exp_flag == 3 || Dims.exp_flag == 4 % experienceassete or experienceassetze
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, Grids.z_gridvals_J(:,:,j), n_e, Grids.e_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 2 || Dims.exp_flag == 5 % experienceassetz or experienceassetsemiz
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, Dims.n_bothz, Grids.bothz_gridvals_J(:,:,j));
        else
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2);
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

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EV_interp = InterpolateExpAssetEV(EVpre, n_a2, N_d2, n_a1, n_a2, N_bothz, ExpAsset_J.a2primeIndex, ExpAsset_J.a2primeProbs);
            EV_interp = reshape(EV_interp, [N_d2 * N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Pol_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
            else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                else; pi_bz = Probs.pi_z_J(:,:,j); end

                EV_expected = compute_expected_value(EV_interp, pi_bz);
                DiscountedEV = df * reshape(EV_expected, [1, N_d2*N_a1, 1, N_a2, N_bothz, 1]);
            end

            RM = CreateReturnFnMatrix_ExpAsset_uraw(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec);
            RM = reshape(RM, [N_d1, N_d2*N_a1, N_a1, N_a2, N_bothz, N_e]);

            if has_cont; RHS = RM + DiscountedEV; else; RHS = RM; end
            [v_tmp, p_tmp] = max(reshape(RHS, [N_d_inner * N_a1, N_a, N_bothz, N_e]), [], 1);

            V_fordouter(:,:,:,d_outer_c) = shiftdim(v_tmp, 1);
            Pol_fordouter(:,:,:,d_outer_c) = shiftdim(p_tmp, 1);
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Pol_fordouter);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EV_interp = reshape(InterpolateExpAssetEV(EVpre, n_a2, N_d2, n_a1, n_a2, N_bothz, ExpAsset_J.a2primeIndex, ExpAsset_J.a2primeProbs), [N_d2 * N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Pol_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
            else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                else; pi_bz = Probs.pi_z_J(:,:,j); end

                EV_expected = compute_expected_value(EV_interp, pi_bz);
                DiscountedEV = df * reshape(EV_expected, [1, N_d2*N_a1, 1, N_a2, N_bothz, 1]);
            end

            for e_c = 1:N_e
                RM = CreateReturnFnMatrix_ExpAsset_uraw(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec);
                RM = reshape(RM, [N_d1, N_d2*N_a1, N_a1, N_a2, N_bothz, 1]);

                if has_cont; RHS = RM + DiscountedEV; else; RHS = RM; end
                [v_tmp, p_tmp] = max(reshape(RHS, [N_d_inner * N_a1, N_a, N_bothz, 1]), [], 1);

                V_fordouter(:,:,e_c,d_outer_c) = shiftdim(v_tmp, 1);
                Pol_fordouter(:,:,e_c,d_outer_c) = shiftdim(p_tmp, 1);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Pol_fordouter);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EV_interp = reshape(InterpolateExpAssetEV(EVpre, n_a2, N_d2, n_a1, n_a2, N_bothz, ExpAsset_J.a2primeIndex, ExpAsset_J.a2primeProbs), [N_d2 * N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Pol_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
            else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                else; pi_bz = Probs.pi_z_J(:,:,j); end
            end

            for z_c = 1:N_bothz
                if has_cont
                    EV_expected = compute_expected_value(EV_interp, pi_bz(z_c,:));
                    DiscountedEV = df * reshape(EV_expected, [1, N_d2*N_a1, 1, N_a2, 1, 1]);
                end

                RM = CreateReturnFnMatrix_ExpAsset_uraw(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.n_bothz_lm2, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, Grids.bothz_gridvals_J(z_c,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec);
                RM = reshape(RM, [N_d1, N_d2*N_a1, N_a1, N_a2, 1, N_e]);

                if has_cont; RHS = RM + DiscountedEV; else; RHS = RM; end
                [v_tmp, p_tmp] = max(reshape(RHS, [N_d_inner * N_a1, N_a, 1, N_e]), [], 1);

                V_fordouter(:,z_c,:,d_outer_c) = shiftdim(v_tmp, 1);
                Pol_fordouter(:,z_c,:,d_outer_c) = shiftdim(p_tmp, 1);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Pol_fordouter);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EV_interp = reshape(InterpolateExpAssetEV(EVpre, n_a2, N_d2, n_a1, n_a2, N_bothz, ExpAsset_J.a2primeIndex, ExpAsset_J.a2primeProbs), [N_d2 * N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Pol_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
            else; d_val = Grids.d12_gridvals; end
            if has_cont; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                if has_cont
                    EV_expected = compute_expected_value(EV_interp, pi_bz(sb,:));
                    DiscountedEV = df * reshape(EV_expected, [1, N_d2*N_a1, 1, N_a2, N_semiz, 1]);
                end

                RM = CreateReturnFnMatrix_ExpAsset_uraw(ReturnFn, Dims.special_n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, Grids.bothz_gridvals_J(sb,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec);
                RM = reshape(RM, [N_d1, N_d2*N_a1, N_a1, N_a2, N_semiz, N_e]);

                if has_cont; RHS = RM + DiscountedEV; else; RHS = RM; end
                [v_tmp, p_tmp] = max(reshape(RHS, [N_d_inner * N_a1, N_a, N_semiz, N_e]), [], 1);

                V_fordouter(:,sb,:,d_outer_c) = shiftdim(v_tmp, 1);
                Pol_fordouter(:,sb,:,d_outer_c) = shiftdim(p_tmp, 1);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Pol_fordouter);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EV_interp = reshape(InterpolateExpAssetEV(EVpre, n_a2, N_d2, n_a1, n_a2, N_bothz, ExpAsset_J.a2primeIndex, ExpAsset_J.a2primeProbs), [N_d2 * N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Pol_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
            else; d_val = Grids.d12_gridvals; end

            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                else; pi_bz = Probs.pi_z_J(:,:,j); end
            end

            for z_c = 1:N_bothz
                if has_cont
                    EV_expected = compute_expected_value(EV_interp, pi_bz(z_c,:));
                    DiscountedEV = df * reshape(EV_expected, [1, N_d2*N_a1, 1, N_a2, 1, 1]);
                end
                for e_c = 1:N_e
                    RM = CreateReturnFnMatrix_ExpAsset_uraw(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.n_bothz_lm2, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, Grids.bothz_gridvals_J(z_c,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec);
                    RM = reshape(RM, [N_d1, N_d2*N_a1, N_a1, N_a2, 1, 1]);

                    if has_cont; RHS = RM + DiscountedEV; else; RHS = RM; end
                    [v_tmp, p_tmp] = max(reshape(RHS, [N_d_inner * N_a1, N_a, 1, 1]), [], 1);

                    V_fordouter(:,z_c,e_c,d_outer_c) = shiftdim(v_tmp, 1);
                    Pol_fordouter(:,z_c,e_c,d_outer_c) = shiftdim(p_tmp, 1);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Pol_fordouter);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EVpre = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
            EV_interp = reshape(InterpolateExpAssetEV(EVpre, n_a2, N_d2, n_a1, n_a2, N_bothz, ExpAsset_J.a2primeIndex, ExpAsset_J.a2primeProbs), [N_d2 * N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        Pol_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
            else; d_val = Grids.d12_gridvals; end
            if has_cont; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                if has_cont
                    EV_expected = compute_expected_value(EV_interp, pi_bz(sb,:));
                    DiscountedEV = df * reshape(EV_expected, [1, N_d2*N_a1, 1, N_a2, N_semiz, 1]);
                end
                for e_c = 1:N_e
                    RM = CreateReturnFnMatrix_ExpAsset_uraw(ReturnFn, Dims.special_n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, Grids.bothz_gridvals_J(sb,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec);
                    RM = reshape(RM, [N_d1, N_d2*N_a1, N_a1, N_a2, N_semiz, 1]);

                    if has_cont; RHS = RM + DiscountedEV; else; RHS = RM; end
                    [v_tmp, p_tmp] = max(reshape(RHS, [N_d_inner * N_a1, N_a, N_semiz, 1]), [], 1);

                    V_fordouter(:,sb,e_c,d_outer_c) = shiftdim(v_tmp, 1);
                    Pol_fordouter(:,sb,e_c,d_outer_c) = shiftdim(p_tmp, 1);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, Pol_fordouter);
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function [V_out, Pol_out] = extract_pol(V_douter, Pol_douter)
        [V_out, maxindex_douter] = max(V_douter, [], 4);
        linear_douter = reshape(maxindex_douter, [N_a * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_douter - 1);
        inner_pol = reshape(Pol_douter(lin_idx), [1, N_a, N_bothz, N_e]);

        % inner_pol is index out of [N_d_inner * N_a1].
        % We want Pol_out to be the full index across [N_d_inner * N_d_outer * N_a1]
        d12_idx = rem(inner_pol - 1, N_d_inner) + 1;
        a1prime_idx = ceil(inner_pol / N_d_inner);

        Pol_out = d12_idx + reshape(maxindex_douter - 1, [1, N_a, N_bothz, N_e]) * N_d_inner + (a1prime_idx - 1) * (N_d_inner * N_d_outer);
    end

    function EV_out = compute_expected_value(EV_in, pi_slice)
        EVinf = (EV_in == -Inf);
        EV_val = EV_in;
        EV_val(EVinf) = -1e250;
        EV_out = EV_val * pi_slice';
        EV_out(EVinf * (pi_slice' > 0) > 0) = -Inf;
    end


end
