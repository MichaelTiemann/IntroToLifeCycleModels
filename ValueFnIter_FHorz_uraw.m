function [V, Policy] = ValueFnIter_FHorz_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_uraw_FHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

% Unpack workspace strictly for JIT efficiency
N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_d = Dims.N_d; N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
has_d2 = Dims.has_d2; has_d1 = Dims.has_d1; N_semiz_raw = Dims.N_semiz_raw;

% Preallocate outputs
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
if has_d2 && N_semiz_raw > 0
    Policy = zeros(3, N_a, N_bothz, N_e, N_j, 'gpuArray');
else
    Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');
end

% --- 2. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1')
            V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]);
        else
            V_next = [];
        end
    else
        V_next = V(:, :, :, j+1);
    end

    if has_d2 && N_semiz_raw > 0
        [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period(j, V_next);
    else
        [V(:,:,:,j), Policy(1,:,:,:,j)] = solve_period(j, V_next);
    end
end

% =====================================================================
% HELPER FUNCTION: Solves a single period j
% =====================================================================
    function [V_j, Pol_j] = solve_period(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_continuation = ~isempty(V_next);

        if has_continuation
            DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Policy_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        % LOW MEMORY 0
        if vfoptions.lowmemory == 0
            if ~has_continuation
                ReturnMatrix = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.n_d_comb, n_a, Dims.n_bothz, Dims.n_e, d_gridvals, a_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 0);
                [V_j, maxindex] = max(ReturnMatrix, [], 1);
                if has_d2 && N_semiz_raw > 0
                    Pol_j = extract_policy_no_d2(maxindex, N_d, N_d1);
                else
                    Pol_j = shiftdim(maxindex, -1);
                end
                return;
            end

            for d2_c = 1:N_d2
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                if has_continuation
                    if N_semiz_raw > 0
                        pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                    else
                        pi_bothz = Probs.pi_z_J(:,:,j);
                    end
                    EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                end

                ReturnMatrix_d2 = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.n_bothz, Dims.n_e, d_val, a_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 0);

                if has_continuation
                    entireRHS = ReturnMatrix_d2 + DiscountFactorParamsVec * repelem(EV_d2, N_d1, 1);
                else
                    entireRHS = ReturnMatrix_d2;
                end

                [Vtemp, maxindex] = max(entireRHS, [], 1);
                V_ford2_jj(:,:,:,d2_c) = shiftdim(Vtemp, 1);
                Policy_ford2_jj(:,:,:,d2_c) = shiftdim(maxindex, 1);
            end

            % LOW MEMORY 1
        elseif vfoptions.lowmemory == 1
            for d2_c = 1:N_d2
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                if has_continuation
                    if N_semiz_raw > 0
                        pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                    else
                        pi_bothz = Probs.pi_z_J(:,:,j);
                    end
                    EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                end

                for e_c = 1:N_e
                    ReturnMatrix_d2e = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.n_bothz, Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 0);
                    if has_continuation; entireRHS = ReturnMatrix_d2e + DiscountFactorParamsVec * repelem(EV_d2, N_d1, 1); else; entireRHS = ReturnMatrix_d2e; end

                    [Vtemp, maxindex] = max(entireRHS, [], 1);
                    V_ford2_jj(:,:,e_c,d2_c) = shiftdim(Vtemp, 1);
                    Policy_ford2_jj(:,:,e_c,d2_c) = shiftdim(maxindex, 1);
                end
            end

            % LOW MEMORY 2
        elseif vfoptions.lowmemory == 2
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                for z_c = 1:N_z
                    semizblock = (z_c-1)*N_semiz + (1:N_semiz);
                    if has_continuation; EV_d2z = compute_expected_value_cond(EV, pi_bothz(semizblock,:)', N_a); end

                    for e_c = 1:N_e
                        ReturnMatrix_d2ze = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(semizblock,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 0);
                        if has_continuation; entireRHS = ReturnMatrix_d2ze + DiscountFactorParamsVec * repelem(EV_d2z, N_d1, 1); else; entireRHS = ReturnMatrix_d2ze; end

                        [Vtemp, maxindex] = max(entireRHS, [], 1);
                        V_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(Vtemp, 1);
                        Policy_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(maxindex, 1);
                    end
                end
            end

            % LOW MEMORY 3
        elseif vfoptions.lowmemory == 3
            for d2_c = 1:N_d2
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                for z_c = 1:N_bothz
                    if has_continuation
                        if N_semiz_raw > 0
                            pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                        else
                            pi_bothz = Probs.pi_z_J(:,:,j);
                        end
                        EV_d2z = compute_expected_value_cond(EV, pi_bothz(z_c,:)', N_a);
                    end

                    for e_c = 1:N_e
                        ReturnMatrix_d2ze = CreateReturnFnMatrix_Disc_uraw(ReturnFn, Dims.special_n_d, n_a, Dims.special_n_bothz, Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(z_c,:,j), Grids.e_gridvals_J(e_c,:,j), ReturnFnParamsVec, 0);
                        if has_continuation; entireRHS = ReturnMatrix_d2ze + DiscountFactorParamsVec * repelem(EV_d2z, N_d1, 1); else; entireRHS = ReturnMatrix_d2ze; end

                        [Vtemp, maxindex] = max(entireRHS, [], 1);
                        V_ford2_jj(:,z_c,e_c,d2_c) = Vtemp;
                        Policy_ford2_jj(:,z_c,e_c,d2_c) = maxindex;
                    end
                end
            end
        end

        % Resolve max over d2
        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_d2(Policy_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = shiftdim(Policy_ford2_jj, -1);
        end
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function d_slice = get_d_slice(d12_gridvals, d2_c, has_d2, has_d1)
        if has_d2 && has_d1; d_slice = d12_gridvals(:,:,d2_c); else; d_slice = d12_gridvals; end
    end

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
        EVinf = (EV == -Inf); EV_val = EV; EV_val(EVinf) = -1e250;
        EV_val = EV_val * pi_bothz'; EV_val(EVinf * (pi_bothz' > 0) > 0) = -Inf;
        EV_out = reshape(EV_val, [N_a, 1, N_bothz]);
    end

    function EV_out = compute_expected_value_cond(EV, pi_slice, N_a)
        EV_val = EV .* shiftdim(pi_slice, -1); EV_val(isnan(EV_val)) = 0;
        EV_out = sum(EV_val, 2);
    end


end
