function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

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

qh_mode = Dims.qh_mode;
beta0   = Dims.beta0;
ez_flag = Dims.ez_flag;

a_gridvals = gpuArray(a_grid);
level1n    = Dims.level1n;
level1ii   = round(linspace(1, N_a, level1n));

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

if qh_mode == 1
    vfopt_exp = vfoptions; vfopt_exp.exoticpreferences = 'None';
    [Valt, Policyalt] = ValueFnIter_UFHorz_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfopt_exp);
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

a1p_offsets_base = gpuArray(0:1:N_a);
bz_off_L2 = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, 1, N_z_slice, 1]);

zind_SL = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, N_z_slice, 1]);
eind_SL = reshape(gpuArray(0:1:N_e_slice-1), [1, 1, 1, N_e_slice]);
ze_lin_off  = N_d_inner * zind_SL + (N_d_inner * N_z_slice) * eind_SL;
bz_off_soph = N_a * zind_SL;

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
            WGraw = CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
            WG_j  = WGraw;
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

                % --- Stage 1: Coarse Level 1 Divide-and-Conquer ---
                RM_L1 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, spec_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a, level1n, N_z_slice, N_e_slice]);

                if ez_flag
                    becareful1 = isfinite(RM_L1) & (RM_L1 ~= 0);
                    temp2_L1 = RM_L1;
                    temp2_L1(becareful1) = RM_L1(becareful1).^Dims.ezc2(j);
                    temp2_L1(RM_L1 == 0) = -Inf;
                    if has_ev_term
                        if z_mode == 0; EV_slice = DiscountedEV; else; EV_slice = DiscountedEV(:, :, :, z_idx, 1); end
                        RHS_L1 = ezc1_j * temp2_L1 + beta0 * EV_slice;
                        temp5 = isfinite(RHS_L1) & (RHS_L1 ~= 0);
                        RHS_L1(temp5) = RHS_L1(temp5).^Dims.ezc7(j);
                        RHS_L1(RHS_L1 == 0) = -Inf;
                        ev_flat_offset = (z_c - 1) * (N_a * N_z_slice);
                    else
                        RHS_L1 = RM_L1;
                        RHS_L1(becareful1) = (ezc1_j * temp2_L1(becareful1)).^Dims.ezc7(j);
                        RHS_L1(RM_L1 == 0) = -Inf;
                    end
                elseif has_cont
                    if z_mode == 0
                        RHS_L1 = RM_L1 + beta0 * DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + beta0 * DiscountedEV(:, :, :, z_idx, 1);
                    end
                    ev_flat_offset = (z_c - 1) * (N_a * N_z_slice);
                else
                    RHS_L1 = RM_L1;
                end

                [~, maxindex1] = max(RHS_L1, [], 2);
                [v_tmp_L1, maxindex2] = max(reshape(RHS_L1, [N_d_inner * N_a, level1n, N_z_slice, N_e_slice]), [], 1);

                a1p_L1 = ceil(maxindex2 / N_d_inner);
                if qh_mode == 2
                    if has_ev_term
                        ev_L1_chosen = DiscountedEV(a1p_L1 + bz_off_soph + ev_flat_offset);
                        if ez_flag
                            v_alt_L1 = ((v_tmp_L1).^(1/Dims.ezc7(j)) + (1 - beta0) * ev_L1_chosen).^Dims.ezc7(j);
                            v_alt_L1(v_tmp_L1 == -Inf) = -Inf;
                        else
                            v_alt_L1 = v_tmp_L1 + (1 - beta0) * ev_L1_chosen;
                        end
                    else
                        v_alt_L1 = v_tmp_L1;
                    end
                end

                if N_d_outer == 1
                    V(level1ii, z_idx, e_idx, j)         = reshape(v_tmp_L1, [level1n, N_z_slice, N_e_slice]);
                    Policy(1, level1ii, z_idx, e_idx, j) = reshape(maxindex2, [1, level1n, N_z_slice, N_e_slice]);
                    if qh_mode == 2
                        Valt(level1ii, z_idx, e_idx, j) = reshape(v_alt_L1, [level1n, N_z_slice, N_e_slice]);
                    end
                else
                    d_inner_L1 = rem(maxindex2 - 1, N_d_inner) + 1;
                    linear_L1  = d_inner_L1 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_L1 - 1);
                    V_fordouter(level1ii, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L1, [level1n, N_z_slice, N_e_slice]);
                    P_fordouter(level1ii, z_idx, e_idx, d_outer_c) = reshape(linear_L1, [level1n, N_z_slice, N_e_slice]);
                    if qh_mode == 2
                        Valt_fordouter(level1ii, z_idx, e_idx, d_outer_c) = reshape(v_alt_L1, [level1n, N_z_slice, N_e_slice]);
                    end
                end

                % --- Stage 2: Level 2 Bounded Monotonicity Refinement ---
                diff_a1p = max(0, maxindex1(:, 1, 2:end, :, :) - maxindex1(:, 1, 1:end-1, :, :));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 4, 5]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(level1n - 1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                    if mg > 0
                        loweredge = min(maxindex1(:, 1, ii, :, :), N_a - mg);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;
                    else
                        loweredge = maxindex1(:, 1, ii, :, :);
                        a1p_idx = loweredge;
                    end
                    a1p_grid_safe = reshape(a_gridvals(a1p_idx), [N_d_inner, mg+1, 1, N_z_slice, N_e_slice]);

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, spec_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner * (mg+1), L, N_z_slice, N_e_slice]);

                    if ez_flag
                        becareful2 = isfinite(RM_L2) & (RM_L2 ~= 0);
                        if has_ev_term
                            temp2_L2 = RM_L2;
                            temp2_L2(becareful2) = RM_L2(becareful2).^Dims.ezc2(j);
                            temp2_L2(RM_L2 == 0) = -Inf;
                            aprimez = a1p_idx + N_a * bz_off_L2 + ev_flat_offset;
                            EV_L2   = reshape(DiscountedEV(aprimez(:)), [N_d_inner * (mg+1), 1, N_z_slice, N_e_slice]);
                            RHS_L2  = ezc1_j * temp2_L2 + beta0 * EV_L2;
                            temp5 = isfinite(RHS_L2) & (RHS_L2 ~= 0);
                            RHS_L2(temp5) = RHS_L2(temp5).^Dims.ezc7(j);
                            RHS_L2(RHS_L2 == 0) = -Inf;
                        else
                            RHS_L2 = RM_L2;
                            RHS_L2(becareful2) = (ezc1_j * RM_L2(becareful2).^Dims.ezc2(j)).^Dims.ezc7(j);
                            RHS_L2(RM_L2 == 0) = -Inf;
                        end
                    elseif has_cont
                        aprimez = a1p_idx + N_a * bz_off_L2 + ev_flat_offset;
                        RHS_L2 = RM_L2 + beta0 * reshape(DiscountedEV(aprimez(:)), [N_d_inner * (mg+1), 1, N_z_slice, N_e_slice]);
                    else
                        RHS_L2 = RM_L2;
                    end

                    [v_tmp_L2, maxindex] = max(RHS_L2, [], 1);

                    dind    = rem(maxindex - 1, N_d_inner) + 1;
                    allind  = dind + ze_lin_off;
                    pol_L2  = maxindex + N_d_inner * (loweredge(allind) - 1);
                    a1p_abs = ceil(pol_L2 / N_d_inner);

                    if qh_mode == 2
                        if has_ev_term
                            ev_L2_chosen = DiscountedEV(a1p_abs + bz_off_soph + ev_flat_offset);
                            if ez_flag
                                v_alt_L2 = ((v_tmp_L2).^(1/Dims.ezc7(j)) + (1 - beta0) * ev_L2_chosen).^Dims.ezc7(j);
                                v_alt_L2(v_tmp_L2 == -Inf) = -Inf;
                            else
                                v_alt_L2 = v_tmp_L2 + (1 - beta0) * ev_L2_chosen;
                            end
                        else
                            v_alt_L2 = v_tmp_L2;
                        end
                    end

                    if N_d_outer == 1
                        V(curraindex, z_idx, e_idx, j)         = reshape(v_tmp_L2, [L, N_z_slice, N_e_slice]);
                        Policy(1, curraindex, z_idx, e_idx, j) = reshape(pol_L2, [1, L, N_z_slice, N_e_slice]);
                        if qh_mode == 2
                            Valt(curraindex, z_idx, e_idx, j) = reshape(v_alt_L2, [L, N_z_slice, N_e_slice]);
                        end
                    else
                        linear_L2 = dind + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);
                        V_fordouter(curraindex, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L2, [L, N_z_slice, N_e_slice]);
                        P_fordouter(curraindex, z_idx, e_idx, d_outer_c) = reshape(linear_L2, [L, N_z_slice, N_e_slice]);
                        if qh_mode == 2
                            Valt_fordouter(curraindex, z_idx, e_idx, d_outer_c) = reshape(v_alt_L2, [L, N_z_slice, N_e_slice]);
                        end
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
sj_j = Dims.sj(j); ezc6_j = Dims.ezc6(j); ezc8_j = Dims.ezc8(j);
if ~isempty(EVpre_flat)
    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat; EV_val(EVinf) = 0;
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
    temp4 = WG_j;
    fin_mask = isfinite(temp4) & (temp4 ~= 0);
    temp4(fin_mask) = ((1 - sj_j) * temp4(fin_mask).^ezc8_j).^ezc6_j;
    temp4(WG_j == 0) = 0;
    temp4 = repmat(temp4, 1, N_bothz);
end
DiscountedEV_2D = Dims.ezc3 * df * temp4;
DiscountedEV_2D(isinf(temp4)) = -Inf;
DiscountedEV = reshape(DiscountedEV_2D, [1, N_a, 1, N_bothz, 1]);
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
