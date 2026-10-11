function [V, Policy] = ValueFnIter_FHorz_SemiExo_e_raw(n_d1, n_d2, n_a, n_z, n_semiz, n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Setup Dimensions & Grids ---
if n_d1==0
    n_d = n_d2;
elseif n_d2==0
    n_d = n_d1;
else
    n_d = [n_d1, n_d2];
end

if n_semiz==0
    n_bothz = n_z;
elseif n_z==0
    n_bothz = n_semiz;
else
    n_bothz = [n_semiz, n_z];
end

% Raw dimensions
N_d1_raw = prod(n_d1);
N_d2_raw = prod(n_d2);
N_a = prod(n_a);
N_semiz_raw = prod(n_semiz);
N_z_raw = prod(n_z);
N_e_raw = prod(n_e);

% Protections and dummy variables for missing dimensions
has_d1 = (N_d1_raw > 0);
l_d1 = has_d1 * length(n_d1);
N_d1 = max(N_d1_raw, 1);

has_d2 = (N_d2_raw > 0);
N_d2 = max(N_d2_raw, 1);
l_d2 = has_d2 * length(n_d2);
N_d = N_d1 * N_d2;

if N_z_raw == 0
    pi_z_J = ones(1, 1, N_j);
    z_gridvals_J = zeros(1, 1, N_j);
    if vfoptions.lowmemory > 0
        special_n_z = 0;
    end
else
    if vfoptions.lowmemory > 0
        special_n_z = ones(1, length(n_z));
    end
end
N_z = max(N_z_raw, 1);

if N_semiz_raw == 0
    pi_semiz_J = ones(1, 1, N_d2, N_j);
    semiz_gridvals_J = zeros(1, 1, N_j);
end
N_semiz = max(N_semiz_raw, 1);
N_bothz = N_semiz * N_z;

if N_e_raw == 0
    pi_e_J = ones(1, 1, 1, N_j); % 3D array of ones to allow EV integration sum(..., 3)
    e_gridvals_J = zeros(1, 1, N_j);
    if vfoptions.lowmemory > 0
        special_n_e = 0;
    end
else
    pi_e_J = shiftdim(pi_e_J, -2); % Shift to 3rd dimension as expected by the math
    if vfoptions.lowmemory > 0
        special_n_e = ones(1, length(n_e));
    end
end
N_e = max(N_e_raw, 1);


% Preallocate outputs
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');

% Dynamically size Policy based on whether d2 exists
if has_d2
    Policy = zeros(3, N_a, N_bothz, N_e, N_j, 'gpuArray');
else
    Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');
end

% Setup concatenated grids
if N_d2_raw > 0
    d_gridvals = [repmat(d1_gridvals, N_d2, 1), repelem(d2_gridvals, N_d1, 1)];
else
    d_gridvals = d1_gridvals;
end

bothz_gridvals_J = [repmat(semiz_gridvals_J, N_z, 1, 1), repelem(z_gridvals_J, N_semiz, 1, 1)];

% Memory-specific setups
special_n_d = [n_d1, ones(1, l_d2)];
d12_gridvals = permute(reshape(d_gridvals, [N_d1, N_d2, l_d1 + l_d2]), [1, 3, 2]);

if vfoptions.lowmemory == 3
    special_n_bothz = ones(1, length(n_semiz) + length(n_z));
end

% --- 2. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', j, N_j)
    end

    % Determine continuation value (EV)
    if j == N_j
        if isfield(vfoptions, 'V_Jplus1')
            V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]);
        else
            V_next = []; % Terminal period with no continuation
        end
    else
        V_next = V(:, :, :, j+1);
    end

    % Solve for this period using our centralized helper
    [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period(j, V_next);
end

% =====================================================================
% HELPER FUNCTION: Solves a single period j
% =====================================================================
    function [V_j, Pol_j] = solve_period(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_continuation = ~isempty(V_next);

        if has_continuation
            DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            % Pre-integrate continuation value over e'
            EV = sum(V_next .* pi_e_J(1, 1, :, j+1), 3); % Leaves [N_a, N_bothz]
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Policy_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        % LOW MEMORY 0
        if vfoptions.lowmemory == 0
            if ~has_continuation
                ReturnMatrix = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_bothz, n_e, d_gridvals, a_grid, bothz_gridvals_J(:,:,j), e_gridvals_J(:,:,j), ReturnFnParamsVec, 0);
                [V_j, maxindex] = max(ReturnMatrix, [], 1);

                if has_d2
                    Pol_j = extract_policy_no_d2(maxindex, N_d, N_d1);
                else
                    Pol_j = shiftdim(maxindex, -1);
                end
                return; % Exit early for no-continuation case
            end

            for d2_c = 1:N_d2
                pi_bothz = kron(pi_z_J(:,:,j), pi_semiz_J(:,:,d2_c,j));
                ReturnMatrix_d2 = CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, n_e, d12_gridvals(:,:,d2_c), a_grid, bothz_gridvals_J(:,:,j), e_gridvals_J(:,:,j), ReturnFnParamsVec, 0);

                EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                entireRHS = ReturnMatrix_d2 + DiscountFactorParamsVec * repelem(EV_d2, N_d1, 1);

                [Vtemp, maxindex] = max(entireRHS, [], 1);
                V_ford2_jj(:,:,:,d2_c) = shiftdim(Vtemp, 1);
                Policy_ford2_jj(:,:,:,d2_c) = shiftdim(maxindex, 1);
            end

            % LOW MEMORY 1 (Parallel over bothz, loop over e)
        elseif vfoptions.lowmemory == 1
            for d2_c = 1:N_d2
                pi_bothz = kron(pi_z_J(:,:,j), pi_semiz_J(:,:,d2_c,j));
                if has_continuation
                    EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                end

                for e_c = 1:N_e
                    e_val = e_gridvals_J(e_c,:,j);
                    ReturnMatrix_d2e = CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, n_bothz, special_n_e, d12_gridvals(:,:,d2_c), a_grid, bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 0);

                    if has_continuation
                        entireRHS = ReturnMatrix_d2e + DiscountFactorParamsVec * repelem(EV_d2, N_d1, 1);
                    else
                        entireRHS = ReturnMatrix_d2e;
                    end

                    [Vtemp, maxindex] = max(entireRHS, [], 1);
                    V_ford2_jj(:,:,e_c,d2_c) = shiftdim(Vtemp, 1);
                    Policy_ford2_jj(:,:,e_c,d2_c) = shiftdim(maxindex, 1);
                end
            end

            % LOW MEMORY 2 (Outer z / inner e, vectorize semiz)
        elseif vfoptions.lowmemory == 2
            for d2_c = 1:N_d2
                pi_bothz = kron(pi_z_J(:,:,j), pi_semiz_J(:,:,d2_c,j));

                for z_c = 1:N_z
                    semizblock = (z_c-1)*N_semiz + (1:N_semiz);
                    z_valblock = bothz_gridvals_J(semizblock,:,j);

                    if has_continuation
                        EV_d2z = compute_expected_value_cond(EV, pi_bothz(semizblock,:)', N_a);
                    end

                    for e_c = 1:N_e
                        e_val = e_gridvals_J(e_c,:,j);
                        ReturnMatrix_d2ze = CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, [n_semiz, special_n_z], special_n_e, d12_gridvals(:,:,d2_c), a_grid, z_valblock, e_val, ReturnFnParamsVec, 0);

                        if has_continuation
                            entireRHS = ReturnMatrix_d2ze + DiscountFactorParamsVec * repelem(EV_d2z, N_d1, 1);
                        else
                            entireRHS = ReturnMatrix_d2ze;
                        end

                        [Vtemp, maxindex] = max(entireRHS, [], 1);
                        V_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(Vtemp, 1);
                        Policy_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(maxindex, 1);
                    end
                end
            end

            % LOW MEMORY 3 (Joint bothz, inner e)
        elseif vfoptions.lowmemory == 3
            for d2_c = 1:N_d2
                pi_bothz = kron(pi_z_J(:,:,j), pi_semiz_J(:,:,d2_c,j));

                for z_c = 1:N_bothz
                    z_val = bothz_gridvals_J(z_c,:,j);

                    if has_continuation
                        EV_d2z = compute_expected_value_cond(EV, pi_bothz(z_c,:)', N_a);
                    end

                    for e_c = 1:N_e
                        e_val = e_gridvals_J(e_c,:,j);
                        ReturnMatrix_d2ze = CreateReturnFnMatrix_Disc_e(ReturnFn, special_n_d, n_a, special_n_bothz, special_n_e, d12_gridvals(:,:,d2_c), a_grid, z_val, e_val, ReturnFnParamsVec, 0);

                        if has_continuation
                            entireRHS = ReturnMatrix_d2ze + DiscountFactorParamsVec * repelem(EV_d2z, N_d1, 1);
                        else
                            entireRHS = ReturnMatrix_d2ze;
                        end

                        [Vtemp, maxindex] = max(entireRHS, [], 1);
                        V_ford2_jj(:,z_c,e_c,d2_c) = Vtemp;
                        Policy_ford2_jj(:,z_c,e_c,d2_c) = maxindex;
                    end
                end
            end
        end

        % Resolve max over d2 and extract policies (Unified across all modes)
        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);

        if has_d2
            Pol_j = extract_policy_d2(Policy_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = shiftdim(Policy_ford2_jj, -1);
        end
    end

% =====================================================================
% SUB-HELPERS for cleanly extracting policies and EV matrix math
% =====================================================================
    function Pol3 = extract_policy_no_d2(maxindex, N_d, N_d1)
        Pol3 = zeros(3, size(maxindex, 2), size(maxindex, 3), size(maxindex, 4), 'gpuArray');
        d_ind = shiftdim(rem(maxindex - 1, N_d) + 1, -1);
        Pol3(1,:,:,:) = shiftdim(rem(d_ind - 1, N_d1) + 1, -1);
        Pol3(2,:,:,:) = shiftdim(ceil(d_ind / N_d1), -1);
        Pol3(3,:,:,:) = shiftdim(ceil(maxindex / N_d), -1);
    end

    function Pol3 = extract_policy_d2(Policy_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1)
        Pol3 = zeros(3, N_a, N_bothz, N_e, 'gpuArray');
        Pol3(2,:,:,:) = shiftdim(maxindex_d2, -1);

        linear_d2 = reshape(maxindex_d2, [N_a * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_d2 - 1);

        d1aprime_ind = reshape(Policy_ford2_jj(lin_idx), [1, N_a, N_bothz, N_e]);
        Pol3(1,:,:,:) = shiftdim(rem(d1aprime_ind - 1, N_d1) + 1, -1);
        Pol3(3,:,:,:) = shiftdim(ceil(d1aprime_ind / N_d1), -1);
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
        EV_val(isnan(EV_val)) = 0; % Handle -Inf * 0
        EV_out = sum(EV_val, 2);
    end


end
