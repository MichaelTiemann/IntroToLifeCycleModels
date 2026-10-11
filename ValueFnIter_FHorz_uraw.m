function [V, Policy] = ValueFnIter_FHorz_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_uraw_FHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_d = Dims.N_d; N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;
has_d2 = Dims.has_d2; has_d1 = Dims.has_d1;

N_z_raw = Dims.special_n_z(1) > 0;
N_semiz_raw = Dims.N_semiz_raw > 0;
N_e_raw = Dims.N_e_raw > 0;

% --- 2. Determine Iteration Strategy ---
if vfoptions.lowmemory == 0
    IterStrategy = 'Full';
elseif vfoptions.lowmemory == 1
    if N_e_raw
        IterStrategy = 'Loop_e';
    elseif N_semiz_raw && N_z_raw
        IterStrategy = 'Loop_z_parallel_semiz';
    else
        IterStrategy = 'Loop_bothz'; % Covers z-only or semiz-only
    end
elseif vfoptions.lowmemory == 2
    if N_e_raw && N_semiz_raw && N_z_raw
        IterStrategy = 'Loop_z_parallel_semiz_e';
    elseif N_e_raw
        IterStrategy = 'Loop_bothz_e'; % Covers z+e or semiz+e
    elseif N_semiz_raw && N_z_raw
        IterStrategy = 'Loop_bothz';
    else
        error('Invalid lowmemory=2 for this shock combination.');
    end
elseif vfoptions.lowmemory == 3
    if N_e_raw && N_semiz_raw && N_z_raw
        IterStrategy = 'Loop_bothz_e';
    else
        error('Invalid lowmemory=3 for this shock combination.');
    end
end

% --- 3. Preallocate Outputs ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

% --- 4. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]); else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    switch IterStrategy
        case 'Full'
            [V(:,:,:,j), Pol_j] = solve_period_full(j, V_next);
        case 'Loop_e'
            [V(:,:,:,j), Pol_j] = solve_period_loop_e(j, V_next);
        case 'Loop_z_parallel_semiz'
            [V(:,:,:,j), Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next);
        case 'Loop_bothz'
            [V(:,:,:,j), Pol_j] = solve_period_loop_bothz(j, V_next);
        case 'Loop_z_parallel_semiz_e'
            [V(:,:,:,j), Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next);
        case 'Loop_bothz_e'
            [V(:,:,:,j), Pol_j] = solve_period_loop_bothz_e(j, V_next);
    end

    Policy(:,:,:,:,j) = Pol_j;
end

% =====================================================================
% SPECIALIZED SOLVE FUNCTIONS
% =====================================================================

    function [V_j, Pol_j] = solve_period_full(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont; df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j)); EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3); end

        V_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Pol_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_d2 = compute_expected_value(EV, pi_bz, N_a, N_bothz);
            end

            RM = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.n_bothz, Dims.n_e, d_val, a_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 0);
            RM = reshape(RM, [N_d1, N_a, N_a, N_bothz, N_e]);
            if has_cont; RHS = RM + df * reshape(EV_d2, [1, N_a, 1, N_bothz, 1]); else; RHS = RM; end

            [v_tmp, p_tmp] = max(reshape(RHS, [N_d1 * N_a, N_a, N_bothz, N_e]), [], 1);
            V_ford2(:,:,:,d2_c) = shiftdim(v_tmp, 1); Pol_ford2(:,:,:,d2_c) = shiftdim(p_tmp, 1);
        end
        [V_j, Pol_j] = extract_pol(V_ford2, Pol_ford2);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont; df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j)); EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3); end

        V_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Pol_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
            if has_cont
                if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_d2 = compute_expected_value(EV, pi_bz, N_a, N_bothz);
            end

            for e_c = 1:N_e
                RM = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.n_bothz, Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 0);
                RM = reshape(RM, [N_d1, N_a, N_a, N_bothz, 1]);
                if has_cont; RHS = RM + df * reshape(EV_d2, [1, N_a, 1, N_bothz, 1]); else; RHS = RM; end

                [v_tmp, p_tmp] = max(reshape(RHS, [N_d1 * N_a, N_a, N_bothz, 1]), [], 1);
                V_ford2(:,:,e_c,d2_c) = shiftdim(v_tmp, 1); Pol_ford2(:,:,e_c,d2_c) = shiftdim(p_tmp, 1);
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2, Pol_ford2);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont; df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j)); EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3); end

        V_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Pol_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
            if has_cont; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; end

            for z_c = 1:N_bothz
                if has_cont; EV_d2z = compute_expected_value_cond(EV, pi_bz(z_c,:)', N_a); end
                RM = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.n_e, d_val, a_grid, Grids.bothz_gridvals_J(z_c,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 0);
                RM = reshape(RM, [N_d1, N_a, N_a, 1, N_e]);
                if has_cont; RHS = RM + df * reshape(EV_d2z, [1, N_a, 1, 1, 1]); else; RHS = RM; end

                [v_tmp, p_tmp] = max(reshape(RHS, [N_d1 * N_a, N_a, 1, N_e]), [], 1);
                V_ford2(:,z_c,:,d2_c) = shiftdim(v_tmp, 1); Pol_ford2(:,z_c,:,d2_c) = shiftdim(p_tmp, 1);
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2, Pol_ford2);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont; df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j)); EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3); end

        V_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Pol_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
            if has_cont; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                if has_cont; EV_d2z = compute_expected_value_cond(EV, pi_bz(sb,:)', N_a); end
                RM = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.n_bothz_lm2, Dims.n_e, d_val, a_grid, Grids.bothz_gridvals_J(sb,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 0);
                RM = reshape(RM, [N_d1, N_a, N_a, N_semiz, N_e]);
                if has_cont; RHS = RM + df * reshape(EV_d2z, [1, N_a, 1, N_semiz, 1]); else; RHS = RM; end

                [v_tmp, p_tmp] = max(reshape(RHS, [N_d1 * N_a, N_a, N_semiz, N_e]), [], 1);
                V_ford2(:,sb,:,d2_c) = shiftdim(v_tmp, 1); Pol_ford2(:,sb,:,d2_c) = shiftdim(p_tmp, 1);
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2, Pol_ford2);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont; df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j)); EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3); end

        V_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Pol_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
            if has_cont; if N_semiz_raw; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end; end

            for z_c = 1:N_bothz
                if has_cont; EV_d2z = compute_expected_value_cond(EV, pi_bz(z_c,:)', N_a); end
                for e_c = 1:N_e
                    RM = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(z_c,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 0);
                    RM = reshape(RM, [N_d1, N_a, N_a, 1, 1]);
                    if has_cont; RHS = RM + df * reshape(EV_d2z, [1, N_a, 1, 1, 1]); else; RHS = RM; end

                    [v_tmp, p_tmp] = max(reshape(RHS, [N_d1 * N_a, N_a, 1, 1]), [], 1);
                    V_ford2(:,z_c,e_c,d2_c) = shiftdim(v_tmp, 1); Pol_ford2(:,z_c,e_c,d2_c) = shiftdim(p_tmp, 1);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2, Pol_ford2);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_cont = ~isempty(V_next);
        if has_cont; df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j)); EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3); end

        V_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Pol_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        for d2_c = 1:N_d2
            d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
            if has_cont; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                if has_cont; EV_d2z = compute_expected_value_cond(EV, pi_bz(sb,:)', N_a); end
                for e_c = 1:N_e
                    RM = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.n_bothz_lm2, Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(sb,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 0);
                    RM = reshape(RM, [N_d1, N_a, N_a, N_semiz, 1]);
                    if has_cont; RHS = RM + df * reshape(EV_d2z, [1, N_a, 1, N_semiz, 1]); else; RHS = RM; end

                    [v_tmp, p_tmp] = max(reshape(RHS, [N_d1 * N_a, N_a, N_semiz, 1]), [], 1);
                    V_ford2(:,sb,e_c,d2_c) = shiftdim(v_tmp, 1); Pol_ford2(:,sb,e_c,d2_c) = shiftdim(p_tmp, 1);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_ford2, Pol_ford2);
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function [V_out, Pol_out] = extract_pol(V_d2, Pol_d2)
        [V_out, maxindex_d2] = max(V_d2, [], 4);
        if has_d2 && N_semiz_raw
            linear_d2 = reshape(maxindex_d2, [N_a * N_bothz * N_e, 1]);
            lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_d2 - 1);
            inner_pol = reshape(Pol_d2(lin_idx), [1, N_a, N_bothz, N_e]);
            
            % Unpack inner_pol to d1 and aprime
            d1_idx = rem(inner_pol - 1, N_d1) + 1;
            aprime_idx = ceil(inner_pol / N_d1);
            
            % Re-pack into strict d1 x d2 x aprime order for standard UnKron
            Pol_out = d1_idx + reshape(maxindex_d2 - 1, [1, N_a, N_bothz, N_e]) * N_d1 + (aprime_idx - 1) * (N_d1 * N_d2);
        else
            Pol_out = shiftdim(Pol_d2(:,:,:,1), -1);
        end
    end

    function d_slice = get_d_slice(d12_gridvals, d2_c, has_d2, has_d1)
        if has_d2 && has_d1; d_slice = d12_gridvals(:,:,d2_c); else; d_slice = d12_gridvals; end
    end

    function EV_out = compute_expected_value(EV, pi_bothz, N_a, N_bothz)
        EVinf = (EV == -Inf); EV_val = EV; EV_val(EVinf) = -1e250;
        EV_val = EV_val * pi_bothz'; EV_val(EVinf * (pi_bothz' > 0) > 0) = -Inf;
        EV_out = reshape(EV_val, [N_a, 1, N_bothz]);
    end

    function EV_out = compute_expected_value_cond(EV, pi_slice, N_a)
        EV_val = EV .* shiftdim(pi_slice, -1); EV_val(isnan(EV_val)) = 0;
        EV_out = sum(EV_val, 2);
    end


end