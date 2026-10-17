function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_RiskyAsset_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_grid, z_gridvals_J, vfoptions, Parameters);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d3 = Dims.N_d3;
N_d13 = Dims.N_d13; N_d_outer = Dims.N_d_outer; N_d = Dims.N_d;
has_z = Dims.has_z; has_semiz = Dims.has_semiz; has_e = Dims.has_e; has_bothz = Dims.has_bothz;
n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_u = vfoptions.n_u;

N_z_iter = Dims.N_z_iter; N_z_slice = Dims.N_z_slice; z_mode = Dims.z_mode; spec_n_bz = Dims.spec_n_bz;
N_e_iter = Dims.N_e_iter; N_e_slice = Dims.N_e_slice; e_mode = Dims.e_mode; spec_n_e  = Dims.spec_n_e;

qh_mode = Dims.qh_mode;
beta0   = Dims.beta0;
ez_flag = Dims.ez_flag;

a1_gridvals = Grids.a1_gridvals;
a2_gridvals = Grids.a2_gridvals;
u_gridvals  = Grids.u_gridvals;
pi_u        = vfoptions.pi_u;
aprimeFnParamNames = Dims.aprimeFnParamNames;

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

if qh_mode == 1
    vfopt_exp = vfoptions; vfopt_exp.exoticpreferences = 'None';
    [Valt, Policyalt] = ValueFnIter_UFHorz_RiskyAsset_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfopt_exp);
elseif qh_mode == 2
    Valt = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
    Policyalt = [];
else
    Valt = [];
    Policyalt = [];
end

% --- 2. Preallocate Outputs & Persistent Slice Offsets ---
V      = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

aind_SL  = reshape(gpuArray(0:1:N_a-1),       [1, N_a, 1, 1]);
bzind_SL = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, N_z_slice, 1]);
eind_SL  = reshape(gpuArray(0:1:N_e_slice-1), [1, 1, 1, N_e_slice]);

d1_state_off = (N_d3 * N_a1) * (aind_SL + N_a * bzind_SL + (N_a * N_z_slice) * eind_SL);
d2_bz_off    = (N_d3 * N_a1) * bzind_SL;

if N_d_outer > 1
    V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    if qh_mode == 2
        Valt_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
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
    df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));

    if ez_flag
        if Dims.EZoneminusbeta == 1;     ezc1_j = 1 - df;
        elseif Dims.EZoneminusbeta == 2; ezc1_j = 1 - Dims.sj(j) * df;
        else;                            ezc1_j = Dims.ezc1; end
        if Dims.warmglow
            WGParamsVec = CreateVectorFromParams(Parameters, Dims.WarmGlowParamNames, j);
            WGraw = CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, Grids.a2_grid, WGParamsVec);
            WG_j  = WGraw(:);
            WG_j(isfinite(WGraw)) = (Dims.ezc4 * WGraw(isfinite(WGraw))).^Dims.ezc5(j);
            WG_j(WGraw == 0) = 0;
        else
            WG_j = [];
        end
        has_ev_term = has_cont || Dims.warmglow;
    else
        WG_j = [];
        has_ev_term = has_cont;
    end

    if has_ev_term
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
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        else
            EVpre_flat = [];
        end

        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, j);
        [a2primeIndex, a2primeProbs] = CreateRiskyAssetFnMatrix(vfoptions.aprimeFn, Dims.n_d23, n_a2, n_u, Grids.d23_grid, Grids.a2_grid, u_gridvals, aprimeFnParamsVec, 2);
        Risky_J.a2primeIndex = a2primeIndex;
        Risky_J.a2primeProbs = a2primeProbs;
    end

    for d_outer_c = 1:N_d_outer
        if N_d_outer > 1
            d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d13, size(Grids.d12_gridvals, 3)]);
        else
            d_val = Grids.d12_gridvals;
        end

        if has_ev_term
            if has_cont && has_bothz
                if has_semiz
                    if Dims.has_z
                        pi_bz = kron(pi_z_J(:,:,j), vfoptions.pi_semiz_J(:,:,d_outer_c,j));
                    else
                        pi_bz = vfoptions.pi_semiz_J(:,:,d_outer_c,j);
                    end
                else
                    pi_bz = pi_z_J(:,:,j);
                end
            else
                pi_bz = 1;
            end
            [DiscountedEV_onlyd3, d2index] = precalc_ev_risky_local(EVpre_flat, df, pi_bz, pi_u, Risky_J, Dims, j, WG_j);
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

                % 1. Evaluate ReturnFn over (d1, d3, d4_c) and a1prime
                RM = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, spec_n_bz, spec_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);

                if ez_flag
                    becareful = isfinite(RM) & (RM ~= 0);
                    temp2 = RM;
                    temp2(becareful) = Dims.ezc9 * (RM(becareful).^Dims.ezc2(j));
                    temp2(~becareful) = -Inf;
                    if N_d1 > 1
                        [RM_onlyd3, d1index] = max(reshape(temp2, [N_d1, N_d3 * N_a1, N_a, N_z_slice, N_e_slice]), [], 1);
                        RM_onlyd3 = reshape(RM_onlyd3, [N_d3 * N_a1, N_a, N_z_slice, N_e_slice]);
                    else
                        RM_onlyd3 = reshape(temp2, [N_d3 * N_a1, N_a, N_z_slice, N_e_slice]);
                    end

                    if has_ev_term
                        if z_mode == 0
                            EV_slice = DiscountedEV_onlyd3;
                        else
                            EV_slice = DiscountedEV_onlyd3(:, :, z_idx, 1);
                        end
                        entireRHS = Dims.ezc9 * (ezc1_j * RM_onlyd3 + beta0 * EV_slice);
                        entireRHS(isinf(RM_onlyd3) | isinf(EV_slice)) = -Inf;
                        temp5 = isfinite(entireRHS) & (entireRHS ~= 0);
                        entireRHS(temp5) = entireRHS(temp5).^Dims.ezc7(j);
                        entireRHS(entireRHS == 0) = -Inf;
                        d2_offset_z = (z_c - 1) * (N_d3 * N_a1 * N_z_slice);
                    else
                        entireRHS = Dims.ezc9 * ezc1_j * RM_onlyd3;
                        entireRHS(isinf(RM_onlyd3)) = -Inf;
                        temp5 = isfinite(entireRHS) & (entireRHS ~= 0);
                        entireRHS(temp5) = entireRHS(temp5).^Dims.ezc7(j);
                        entireRHS(entireRHS == 0) = -Inf;
                    end
                else
                    if N_d1 > 1
                        [RM_onlyd3, d1index] = max(reshape(RM, [N_d1, N_d3 * N_a1, N_a, N_z_slice, N_e_slice]), [], 1);
                        RM_onlyd3 = reshape(RM_onlyd3, [N_d3 * N_a1, N_a, N_z_slice, N_e_slice]);
                    else
                        RM_onlyd3 = reshape(RM, [N_d3 * N_a1, N_a, N_z_slice, N_e_slice]);
                    end

                    if has_cont
                        if z_mode == 0
                            entireRHS = RM_onlyd3 + beta0 * DiscountedEV_onlyd3;
                        else
                            entireRHS = RM_onlyd3 + beta0 * DiscountedEV_onlyd3(:, :, z_idx, 1);
                        end
                        d2_offset_z = (z_c - 1) * (N_d3 * N_a1 * N_z_slice);
                    else
                        entireRHS = RM_onlyd3;
                    end
                end

                % Maximize over (d3, a1prime)
                [v_tmp, d3a1p_ind] = max(entireRHS, [], 1); % [1, N_a, N_z_slice, N_e_slice]

                d3_ind  = rem(d3a1p_ind - 1, N_d3) + 1;
                a1p_ind = ceil(d3a1p_ind / N_d3);

                if N_d1 > 1
                    d1_ind = d1index(d3a1p_ind + d1_state_off);
                else
                    d1_ind = 1;
                end

                if has_ev_term && N_d2 > 1
                    d2_ind = d2index(d3a1p_ind + d2_bz_off + d2_offset_z);
                else
                    d2_ind = 1;
                end

                if qh_mode == 2
                    if has_ev_term
                        ev_chosen = DiscountedEV_onlyd3(d3a1p_ind + d2_bz_off + d2_offset_z);
                        if ez_flag
                            v_alt = ((v_tmp).^(1/Dims.ezc7(j)) + Dims.ezc9 * (1 - beta0) * ev_chosen).^Dims.ezc7(j);
                            v_alt(v_tmp == -Inf) = -Inf;
                        else
                            v_alt = v_tmp + (1 - beta0) * ev_chosen;
                        end
                    else
                        v_alt = v_tmp;
                    end
                end

                linear_p = d1_ind + N_d1*(d2_ind - 1) + (N_d1*N_d2)*(d3_ind - 1) + (N_d1*N_d2*N_d3)*(d_outer_c - 1) + N_d*(a1p_ind - 1);

                if N_d_outer == 1
                    V(:, z_idx, e_idx, j)         = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    Policy(1, :, z_idx, e_idx, j) = reshape(linear_p, [1, N_a, N_z_slice, N_e_slice]);
                    if qh_mode == 2
                        Valt(:, z_idx, e_idx, j) = reshape(v_alt, [N_a, N_z_slice, N_e_slice]);
                    end
                else
                    V_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    P_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(linear_p, [N_a, N_z_slice, N_e_slice]);
                    if qh_mode == 2
                        Valt_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(v_alt, [N_a, N_z_slice, N_e_slice]);
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
        end
    end
end

end % <--- MAIN FUNCTION ENDS HERE!

function [DiscountedEV_onlyd3, d2index] = precalc_ev_risky_local(EVpre_flat, df, pi_bz, pi_u, Risky_J, Dims, j, WG_j)
N_a1 = Dims.N_a1; N_bothz = Dims.N_bothz;
N_d2 = Dims.N_d2; N_d3 = Dims.N_d3; N_d23 = Dims.N_d23; N_u = Dims.N_u;

a2p_idx   = Risky_J.a2primeIndex; % [N_d23, N_u]
a2p_probs = Risky_J.a2primeProbs; % [N_d23, N_u]

if ~isempty(EVpre_flat)
    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat;
    if Dims.ez_flag; EV_val(EVinf) = 0; else; EV_val(EVinf) = -1e250; end
    EV_z = EV_val * pi_bz';
    EV_z((EVinf * (pi_bz' > 0)) > 0) = -Inf; % [N_a, N_bothz]

    a1_base = repelem(gpuArray(1:1:N_a1)', N_d23, N_u);
    idx_lo  = a1_base + N_a1 * repmat(a2p_idx - 1, N_a1, 1);
    idx_hi  = a1_base + N_a1 * repmat(a2p_idx,     N_a1, 1);

    Vlower = reshape(EV_z(idx_lo(:), :), [N_d23 * N_a1, N_u, N_bothz]);
    Vupper = reshape(EV_z(idx_hi(:), :), [N_d23 * N_a1, N_u, N_bothz]);

    prob_u = repmat(reshape(a2p_probs, [N_d23, 1, N_u]), [1, N_a1, 1, N_bothz]);
    prob_u = reshape(prob_u, [N_d23 * N_a1, N_u, N_bothz]);
    prob_u(Vlower == Vupper) = 0;

    c_lo = prob_u .* Vlower;       c_lo(isnan(c_lo)) = 0;
    c_hi = (1 - prob_u) .* Vupper; c_hi(isnan(c_hi)) = 0;
    EV_integrated = reshape(sum((c_lo + c_hi) .* reshape(pi_u, [1, N_u, 1]), 2), [N_d23 * N_a1, N_bothz]);
else
    EV_integrated = [];
end

if ~Dims.ez_flag
    [EV_onlyd3, d2index] = max(reshape(df * EV_integrated, [N_d2, N_d3 * N_a1, N_bothz]), [], 1);
    DiscountedEV_onlyd3 = reshape(EV_onlyd3, [N_d3 * N_a1, 1, N_bothz, 1]);
    d2index             = reshape(d2index,   [N_d3 * N_a1, N_bothz]);
    return;
end

sj_j = Dims.sj(j); ezc6_j = Dims.ezc6(j); ezc8_j = Dims.ezc8(j);
if Dims.warmglow
    WG_lo  = reshape(WG_j(a2p_idx),     [N_d23, N_u]);
    WG_hi  = reshape(WG_j(a2p_idx + 1), [N_d23, N_u]);
    prb_wg = a2p_probs; prb_wg(WG_lo == WG_hi) = 0;
    w1 = prb_wg .* WG_lo;       w1(isnan(w1)) = 0;
    w2 = (1 - prb_wg) .* WG_hi; w2(isnan(w2)) = 0;
    WG_u = sum((w1 + w2) .* reshape(pi_u, [1, N_u]), 2); % [N_d23, 1]
    WG_rep = repmat(WG_u, N_a1, N_bothz);                % [N_d23 * N_a1, N_bothz]
end

if ~isempty(EV_integrated)
    temp4 = EV_integrated;
    if Dims.warmglow
        becareful = isfinite(temp4) & isfinite(WG_rep);
        temp4(becareful) = (sj_j * temp4(becareful).^ezc8_j + (1 - sj_j) * WG_rep(becareful).^ezc8_j).^ezc6_j;
        temp4((EV_integrated == 0) & (WG_rep == 0)) = 0;
    else
        fin_mask = isfinite(temp4);
        temp4(fin_mask) = (sj_j * temp4(fin_mask).^ezc8_j).^ezc6_j;
        temp4(EV_integrated == 0) = 0;
    end
else
    temp4 = WG_rep;
    fin_mask = isfinite(temp4) & (temp4 ~= 0);
    temp4(fin_mask) = ((1 - sj_j) * temp4(fin_mask).^ezc8_j).^ezc6_j;
    temp4(WG_rep == 0) = 0;
end

temp4_scaled = (Dims.ezc9 * Dims.ezc3 * df) * temp4;
temp4_scaled(isinf(temp4)) = -Inf;

[EV_onlyd3, d2index] = max(reshape(temp4_scaled, [N_d2, N_d3 * N_a1, N_bothz]), [], 1);
DiscountedEV_onlyd3 = reshape(EV_onlyd3, [N_d3 * N_a1, 1, N_bothz, 1]);
d2index             = reshape(d2index,   [N_d3 * N_a1, N_bothz]);
end

function [V_j, Pol_j, max_d_out] = extract_pol_local(V_fordouter, P1, N_a, N_bothz, N_e)
[V_j, max_d_out] = max(V_fordouter, [], 4);
lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);
Pol_j = reshape(P1(lin_idx), [1, N_a, N_bothz, N_e]);
end

function slice = get_grid_slice_local(grid_J, idx, j, has_var)
if ~has_var || isempty(grid_J); slice = []; return; end
if strcmp(idx, 'all')
    if ndims(grid_J) >= 3; slice = grid_J(:, :, j); else; slice = grid_J; end
else
    if ndims(grid_J) >= 3; slice = grid_J(idx, :, j); else; slice = grid_J(idx, :); end
end
end
