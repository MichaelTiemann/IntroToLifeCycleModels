function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, vfoptions);

N_a       = Dims.N_a;
N_bothz   = Dims.N_bothz;
N_e       = Dims.N_e;
N_semiz   = Dims.N_semiz;
N_d_inner = Dims.N_d_inner;
N_d_outer = Dims.N_d_outer;
has_semiz = Dims.has_semiz;
has_e     = Dims.has_e;
has_bothz = Dims.has_bothz;

N_z_iter = Dims.N_z_iter; N_z_slice = Dims.N_z_slice; z_mode = Dims.z_mode; spec_n_bz = Dims.spec_n_bz;
N_e_iter = Dims.N_e_iter; N_e_slice = Dims.N_e_slice; e_mode = Dims.e_mode; spec_n_e  = Dims.spec_n_e;

qh_mode = Dims.qh_mode; % 0 = Standard, 1 = Naive QH, 2 = Sophisticated QH
beta0   = Dims.beta0;

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 2. Preallocate Outputs & Persistent Buffers ---
V      = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

if qh_mode > 0
    Valt = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
else
    Valt = [];
end
if qh_mode == 1
    Policyalt = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');
else
    Policyalt = [];
end

if qh_mode == 2
    bz_off_soph = N_a * reshape(gpuArray(0:1:N_z_slice-1), [1, 1, N_z_slice, 1]);
end

if N_d_outer > 1
    V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    if qh_mode > 0
        Valt_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    end
    if qh_mode == 1
        Palt_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    end
end

% --- 3. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1')
            V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]);
        else
            V_next = [];
        end
    else
        if qh_mode > 0
            V_next = Valt(:, :, :, j+1);
        else
            V_next = V(:, :, :, j+1);
        end
    end
    has_cont = ~isempty(V_next);

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);

    if has_cont
        df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
        if has_e
            EVpre = sum(V_next .* reshape(vfoptions.pi_e_J(:, j+1), [1, 1, N_e]), 3);
        else
            EVpre = V_next;
        end
        EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
    end

    for d_outer_c = 1:N_d_outer
        if N_d_outer > 1
            d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
        else
            d_val = Grids.d12_gridvals;
        end

        if has_cont
            if has_bothz
                if has_semiz
                    pi_bz = kron(pi_z_J(:,:,j), vfoptions.pi_semiz_J(:,:,d_outer_c,j));
                else
                    pi_bz = pi_z_J(:,:,j);
                end
            else
                pi_bz = 1;
            end
            DiscountedEV = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz);
        end

        for z_c = 1:N_z_iter
            if z_mode == 0
                z_idx = 1:N_bothz;
                bz_val = get_grid_slice_local(Grids.bothz_gridvals_J, 'all', j, has_bothz);
            elseif z_mode == 1
                z_idx = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice_local(Grids.bothz_gridvals_J, z_idx, j, has_bothz);
            else
                z_idx = z_c;
                bz_val = get_grid_slice_local(Grids.bothz_gridvals_J, z_c, j, has_bothz);
            end

            for e_c = 1:N_e_iter
                if e_mode == 0
                    e_idx = 1:N_e;
                    e_val = get_grid_slice_local(vfoptions.e_gridvals_J, 'all', j, has_e);
                else
                    e_idx = e_c;
                    e_val = get_grid_slice_local(vfoptions.e_gridvals_J, e_c, j, has_e);
                end

                RM = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, spec_n_e, d_val, a_grid, bz_val, e_val, ReturnFnParamsVec, 0);
                RM = reshape(RM, [N_d_inner, N_a, N_a, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0
                        EV_slice = DiscountedEV;
                    else
                        EV_slice = DiscountedEV(:, :, :, z_idx, 1);
                    end

                    if qh_mode == 0
                        entireRHS = RM + EV_slice;
                    elseif qh_mode == 1
                        % Naive QH: solve exponential discounter (Valt, Policyalt) AND present-biased (V, Policy)
                        RHS_alt = RM + EV_slice;
                        [v_alt, p_alt] = max(reshape(RHS_alt, [N_d_inner * N_a, N_a, N_z_slice, N_e_slice]), [], 1);
                        entireRHS = RM + beta0 * EV_slice;
                    else
                        % Sophisticated QH: present self maximizes RM + beta0 * EV
                        entireRHS = RM + beta0 * EV_slice;
                        ev_offset_z = (z_c - 1) * (N_a * N_z_slice);
                    end
                else
                    entireRHS = RM;
                end

                [v_tmp, p_tmp] = max(reshape(entireRHS, [N_d_inner * N_a, N_a, N_z_slice, N_e_slice]), [], 1);

                % Compute Vunderbar (Valt) for Sophisticated QH using exact algebraic identity
                if qh_mode == 2
                    if has_cont
                        aprime_idx = ceil(p_tmp / N_d_inner);
                        v_alt = v_tmp + (1 - beta0) * DiscountedEV(aprime_idx + bz_off_soph + ev_offset_z);
                    else
                        v_alt = v_tmp;
                    end
                elseif qh_mode == 1 && ~has_cont
                    v_alt = v_tmp;
                    p_alt = p_tmp;
                end

                if N_d_outer == 1
                    V(:, z_idx, e_idx, j)         = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    Policy(1, :, z_idx, e_idx, j) = reshape(p_tmp, [1, N_a, N_z_slice, N_e_slice]);
                    if qh_mode > 0
                        Valt(:, z_idx, e_idx, j) = reshape(v_alt, [N_a, N_z_slice, N_e_slice]);
                    end
                    if qh_mode == 1
                        Policyalt(1, :, z_idx, e_idx, j) = reshape(p_alt, [1, N_a, N_z_slice, N_e_slice]);
                    end
                else
                    d_in_idx  = rem(p_tmp - 1, N_d_inner) + 1;
                    ap_idx    = ceil(p_tmp / N_d_inner);
                    linear_p  = d_in_idx + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (ap_idx - 1);

                    V_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    P_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(linear_p, [N_a, N_z_slice, N_e_slice]);

                    if qh_mode > 0
                        Valt_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(v_alt, [N_a, N_z_slice, N_e_slice]);
                    end
                    if qh_mode == 1
                        d_in_alt   = rem(p_alt - 1, N_d_inner) + 1;
                        ap_alt     = ceil(p_alt / N_d_inner);
                        linear_alt = d_in_alt + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (ap_alt - 1);
                        Palt_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(linear_alt, [N_a, N_z_slice, N_e_slice]);
                    end
                end
            end
        end
    end

    if N_d_outer > 1
        [V(:,:,:,j), Policy(:,:,:,:,j), max_d_out] = extract_pol_local(V_fordouter, P_fordouter, N_a, N_bothz, N_e);
        if qh_mode == 2
            % Sophisticated QH: Vunderbar is evaluated at the d_outer chosen by Vhat!
            lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);
            Valt(:,:,:,j) = reshape(Valt_fordouter(lin_idx), [N_a, N_bothz, N_e]);
        elseif qh_mode == 1
            % Naive QH: exponential discounter maximizes its own Valt over d_outer
            [Valt(:,:,:,j), Policyalt(:,:,:,:,j)] = extract_pol_local(Valt_fordouter, Palt_fordouter, N_a, N_bothz, N_e);
        end
    end
end

end
