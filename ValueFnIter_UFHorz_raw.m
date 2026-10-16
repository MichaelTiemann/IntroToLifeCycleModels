function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, vfoptions, Parameters);

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
ez_flag = Dims.ez_flag; % 0 = Standard/QH, 1 = Epstein-Zin

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

% --- 3. Backward Induction Loop (All 6 Strategies Inlined for Buffer Reuse) ---
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
    df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));

    if ez_flag
        if Dims.EZoneminusbeta == 1
            ezc1_j = 1 - df;
        elseif Dims.EZoneminusbeta == 2
            ezc1_j = 1 - Dims.sj(j) * df;
        else
            ezc1_j = Dims.ezc1;
        end
        if Dims.warmglow
            WGParamsVec = CreateVectorFromParams(Parameters, Dims.WarmGlowParamNames, j);
            WGraw = CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
            WG_j  = WGraw;
            WG_j(isfinite(WGraw)) = (Dims.ezc4 * WGraw(isfinite(WGraw))).^Dims.ezc5(j);
            WG_j(WGraw == 0) = 0; % [N_a, 1]
        else
            WG_j = [];
        end
        has_ev_term = has_cont || Dims.warmglow;
    else
        has_ev_term = has_cont;
    end

    if has_cont
        if ez_flag
            V_trans = V_next;
            V_trans(isfinite(V_next)) = (Dims.ezc4 * V_next(isfinite(V_next))).^Dims.ezc5(j);
            V_trans(V_next == 0) = 0;
        else
            V_trans = V_next;
        end
        if has_e
            EVpre = sum(V_trans .* reshape(vfoptions.pi_e_J(:, j+1), [1, 1, N_e]), 3);
        else
            EVpre = V_trans;
        end
        EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
    else
        EVpre_flat = [];
    end

    for d_outer_c = 1:N_d_outer
        if N_d_outer > 1
            d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
        else
            d_val = Grids.d12_gridvals;
        end

        if has_ev_term
            if has_bothz
                if has_semiz
                    pi_bz = kron(pi_z_J(:,:,j), vfoptions.pi_semiz_J(:,:,d_outer_c,j));
                else
                    pi_bz = pi_z_J(:,:,j);
                end
            else
                pi_bz = 1;
            end
            DiscountedEV = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz, j, Dims, WG_j);
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

                if ez_flag
                    % Epstein-Zin ReturnMatrix transformation & aggregation
                    becareful = isfinite(RM) & (RM ~= 0);
                    if has_ev_term
                        temp2 = RM;
                        temp2(becareful) = RM(becareful).^Dims.ezc2(j);
                        temp2(RM == 0) = -Inf;

                        if z_mode == 0
                            EV_slice = DiscountedEV;
                        else
                            EV_slice = DiscountedEV(:, :, :, z_idx, 1);
                        end

                        entireRHS = ezc1_j * temp2 + EV_slice;
                        % Preserve -Inf where temp2 or EV_slice is -Inf, raise positive finite terms to ezc7(j)
                        temp5 = isfinite(entireRHS) & (entireRHS ~= 0);
                        entireRHS(temp5) = entireRHS(temp5).^Dims.ezc7(j);
                        entireRHS(entireRHS == 0) = -Inf;
                    else
                        entireRHS = RM;
                        entireRHS(becareful) = (ezc1_j * RM(becareful).^Dims.ezc2(j)).^Dims.ezc7(j);
                        entireRHS(RM == 0) = -Inf;
                    end
                elseif has_cont
                    if z_mode == 0
                        EV_slice = DiscountedEV;
                    else
                        EV_slice = DiscountedEV(:, :, :, z_idx, 1);
                    end

                    if qh_mode == 0
                        entireRHS = RM + EV_slice;
                    elseif qh_mode == 1
                        RHS_alt = RM + EV_slice;
                        [v_alt, p_alt] = max(reshape(RHS_alt, [N_d_inner * N_a, N_a, N_z_slice, N_e_slice]), [], 1);
                        entireRHS = RM + beta0 * EV_slice;
                    else
                        entireRHS = RM + beta0 * EV_slice;
                        ev_offset_z = (z_c - 1) * (N_a * N_z_slice);
                    end
                else
                    entireRHS = RM;
                end

                [v_tmp, p_tmp] = max(reshape(entireRHS, [N_d_inner * N_a, N_a, N_z_slice, N_e_slice]), [], 1);

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
                    d_in_idx = rem(p_tmp - 1, N_d_inner) + 1;
                    ap_idx   = ceil(p_tmp / N_d_inner);
                    linear_p = d_in_idx + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (ap_idx - 1);

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
            lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);
            Valt(:,:,:,j) = reshape(Valt_fordouter(lin_idx), [N_a, N_bothz, N_e]);
        elseif qh_mode == 1
            [Valt(:,:,:,j), Policyalt(:,:,:,:,j)] = extract_pol_local(Valt_fordouter, Palt_fordouter, N_a, N_bothz, N_e);
        end
    end
end

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function DiscountedEV = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz, j, Dims, WG_j)
    if ~Dims.ez_flag
        EVinf = (EVpre_flat == -Inf);
        EV_val = EVpre_flat;
        EV_val(EVinf) = -1e250;
        EV_val = EV_val * pi_bz';
        EV_val(EVinf * (pi_bz' > 0) > 0) = -Inf;
        DiscountedEV = df * reshape(EV_val, [1, N_a, 1, N_bothz, 1]);
        return;
    end

    % Epstein-Zin certainty-equivalent + Warm-Glow bequest aggregation
    sj_j = Dims.sj(j);
    ezc6_j = Dims.ezc6(j);
    ezc8_j = Dims.ezc8(j);

    if ~isempty(EVpre_flat)
        EVinf = (EVpre_flat == -Inf);
        EV_val = EVpre_flat;
        EV_val(EVinf) = 0; % In EZ, 0*pi_bz replaces NaN from -Inf*0, while -Inf*positive_prob stays -Inf
        EV = EV_val * pi_bz';
        EV(EVinf * (pi_bz' > 0) > 0) = -Inf;

        temp4 = EV;
        if Dims.warmglow
            WG_rep = repmat(WG_j, 1, N_bothz);
            becareful = isfinite(temp4) & isfinite(WG_rep);
            temp4(becareful) = (sj_j * temp4(becareful).^ezc8_j + (1 - sj_j) * WG_rep(becareful).^ezc8_j).^ezc6_j;
            temp4((EV == 0) & (WG_rep == 0)) = 0;
        else
            fin_mask = isfinite(temp4);
            temp4(fin_mask) = (sj_j * temp4(fin_mask).^ezc8_j).^ezc6_j;
            temp4(EV == 0) = 0;
        end
    else
        % Terminal period j == N_j with Warm-Glow bequest only (no V_Jplus1)
        temp4 = WG_j;
        fin_mask = isfinite(temp4) & (temp4 ~= 0);
        temp4(fin_mask) = ((1 - sj_j) * temp4(fin_mask).^ezc8_j).^ezc6_j;
        temp4(WG_j == 0) = 0;
        temp4 = repmat(temp4, 1, N_bothz);
    end

    DiscountedEV = Dims.ezc3 * df * reshape(temp4, [1, N_a, 1, N_bothz, 1]);
end

function [V_j, Pol_j, max_d_out] = extract_pol_local(V_fordouter, P_fordouter, N_a, N_bothz, N_e)
    [V_j, max_d_out] = max(V_fordouter, [], 4);
    lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);
    Pol_j = reshape(P_fordouter(lin_idx), [1, N_a, N_bothz, N_e]);
end

function slice = get_grid_slice_local(grid_J, idx, j, has_var)
    if ~has_var || isempty(grid_J); slice = []; return; end
    if strcmp(idx, 'all')
        if ndims(grid_J) >= 3; slice = grid_J(:, :, j); else; slice = grid_J; end
    else
        if ndims(grid_J) >= 3; slice = grid_J(idx, :, j); else; slice = grid_J(idx, :); end
    end
end
