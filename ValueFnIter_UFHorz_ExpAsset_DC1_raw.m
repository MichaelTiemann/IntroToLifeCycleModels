function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, vfoptions, Parameters);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;
has_z = Dims.has_z; has_semiz = Dims.has_semiz; has_e = Dims.has_e; has_bothz = Dims.has_bothz;
n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_e = Dims.n_e; n_semiz = Dims.n_semiz; n_d2 = Dims.n_d2;

N_z_iter = Dims.N_z_iter; N_z_slice = Dims.N_z_slice; z_mode = Dims.z_mode; spec_n_bz = Dims.spec_n_bz;
N_e_iter = Dims.N_e_iter; N_e_slice = Dims.N_e_slice; e_mode = Dims.e_mode; spec_n_e  = Dims.spec_n_e;

qh_mode = Dims.qh_mode;
beta0   = Dims.beta0;
ez_flag = Dims.ez_flag;

a1_gridvals = Grids.a1_gridvals;
a2_gridvals = Grids.a2_gridvals;
d2_gridvals = Grids.d2_gridvals;
aprimeFnParamNames = Dims.aprimeFnParamNames;
N_e_interp    = Dims.N_e_interp;
N_e_SL_interp = min(N_e_slice, N_e_interp);

level1n  = Dims.level1n;
level1ii = round(linspace(1, N_a1, level1n));

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

if qh_mode == 1
    vfopt_exp = vfoptions; vfopt_exp.exoticpreferences = 'None';
    [Valt, Policyalt] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfopt_exp);
elseif qh_mode == 2
    Valt = zeros(N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');
    Policyalt = [];
else
    Valt = [];
    Policyalt = [];
end

% --- 2. Preallocate Outputs & Persistent Slice Offsets ---
V      = zeros(N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:1:N_a1);

d2_off_L2 = reshape(repelem(gpuArray(1:1:N_d2)', N_d1, 1), [N_d_inner, 1, 1, 1, 1, 1]);
a2_off_L2 = reshape(gpuArray(0:1:N_a2-1),                  [1, 1, 1, N_a2, 1, 1]);
bz_off_L2 = reshape(gpuArray(0:1:N_z_slice-1),             [1, 1, 1, 1, N_z_slice, 1]);
e_off_L2  = reshape(gpuArray(0:1:N_e_SL_interp-1),         [1, 1, 1, 1, 1, N_e_SL_interp]);

le_state_off = N_d_inner * reshape(gpuArray(0:1:(N_a2 * N_z_slice * N_e_slice - 1)), [1, 1, N_a2, N_z_slice, N_e_slice]);

if qh_mode == 2
    a2_ind_soph = reshape(gpuArray(0:1:N_a2-1),          [1, 1, N_a2, 1, 1]);
    bz_ind_soph = reshape(gpuArray(0:1:N_z_slice-1),     [1, 1, 1, N_z_slice, 1]);
    e_ind_soph  = reshape(gpuArray(0:1:N_e_SL_interp-1), [1, 1, 1, 1, N_e_SL_interp]);
    ev_soph_off = (N_d2 * N_a1)*a2_ind_soph + (N_d2 * N_a1 * N_a2)*bz_ind_soph + (N_d2 * N_a1 * N_a2 * N_bothz)*e_ind_soph;
end

if N_d_outer > 1
    V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
    P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
    if qh_mode == 2
        Valt_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
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
        if qh_mode == 1
            V_next = Valt(:, :, :, j+1);
        elseif qh_mode == 2
            V_next = reshape(Valt(:, :, :, :, j+1), [N_a, N_bothz, N_e]);
        else
            V_next = reshape(V(:, :, :, :, j+1), [N_a, N_bothz, N_e]);
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
        if Dims.exp_flag == 3 || Dims.exp_flag == 4
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, z_gridvals_J(:,:,j), n_e, get_grid_slice_local(vfoptions.e_gridvals_J, 'all', j, has_e));
        elseif Dims.exp_flag == 2
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, z_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 5
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_semiz, get_grid_slice_local(vfoptions.semiz_gridvals_J, 'all', j, has_semiz));
        elseif Dims.exp_flag == 6
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, vfoptions.n_u, Grids.u_gridvals);
        else
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2);
        end
        ExpAsset_J.a2primeIndex = a2primeIndex;
        ExpAsset_J.a2primeProbs = a2primeProbs;
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
            [DiscountedEV, DiscountedEV_d2] = precalc_ev_local(EVpre_flat, df, pi_bz, N_e_interp, ExpAsset_J, Dims, vfoptions, j, WG_j);
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
                    e_ev_idx = 1:N_e_interp;
                else
                    e_idx = e_c;
                    e_val = get_grid_slice_local(vfoptions.e_gridvals_J, e_c, j, has_e);
                    e_ev_idx = min(e_c, N_e_interp);
                end

                % --- Stage 1: Coarse Level 1 Divide-and-Conquer ---
                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, spec_n_bz, spec_n_e, d_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, level1n, N_a2, N_z_slice, N_e_slice]);

                if ez_flag
                    becareful1 = isfinite(RM_L1) & (RM_L1 ~= 0);
                    temp2_L1 = RM_L1;
                    temp2_L1(becareful1) = RM_L1(becareful1).^Dims.ezc2(j);
                    temp2_L1(RM_L1 == 0) = -Inf;
                    if has_ev_term
                        if z_mode == 0 && (e_mode == 0 || N_e_interp == 1)
                            EV_slice = DiscountedEV;
                        else
                            EV_slice = DiscountedEV(:, :, :, :, z_idx, e_ev_idx);
                        end
                        RHS_L1 = ezc1_j * temp2_L1 + beta0 * EV_slice;
                        temp5 = isfinite(RHS_L1) & (RHS_L1 ~= 0);
                        RHS_L1(temp5) = RHS_L1(temp5).^Dims.ezc7(j);
                        RHS_L1(RHS_L1 == 0) = -Inf;
                        ev_flat_offset = (z_c - 1)*(N_d2*N_a1*N_a2*N_z_slice) + (e_ev_idx(1) - 1)*(N_d2*N_a1*N_a2*N_bothz);
                    else
                        RHS_L1 = RM_L1;
                        RHS_L1(becareful1) = (ezc1_j * temp2_L1(becareful1)).^Dims.ezc7(j);
                        RHS_L1(RM_L1 == 0) = -Inf;
                    end
                elseif has_cont
                    if z_mode == 0 && (e_mode == 0 || N_e_interp == 1)
                        RHS_L1 = RM_L1 + beta0 * DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + beta0 * DiscountedEV(:, :, :, :, z_idx, e_ev_idx);
                    end
                    ev_flat_offset = (z_c - 1)*(N_d2*N_a1*N_a2*N_z_slice) + (e_ev_idx(1) - 1)*(N_d2*N_a1*N_a2*N_bothz);
                else
                    RHS_L1 = RM_L1;
                end

                [~, a1p_idx_L1] = max(RHS_L1, [], 2);
                [v_tmp_L1, maxindex1] = max(reshape(RHS_L1, [N_d_inner * N_a1, level1n, N_a2, N_z_slice, N_e_slice]), [], 1);

                d_in_L1 = rem(maxindex1 - 1, N_d_inner) + 1;
                a1p_L1  = ceil(maxindex1 / N_d_inner);

                if qh_mode == 2
                    if has_ev_term
                        d2_L1 = ceil(d_in_L1 / N_d1);
                        ev_L1_chosen = DiscountedEV_d2(d2_L1 + N_d2*(a1p_L1 - 1) + ev_soph_off + ev_flat_offset);
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
                    V(level1ii, :, z_idx, e_idx, j)         = reshape(v_tmp_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    Policy(1, level1ii, :, z_idx, e_idx, j) = reshape(maxindex1, [1, level1n, N_a2, N_z_slice, N_e_slice]);
                    if qh_mode == 2
                        Valt(level1ii, :, z_idx, e_idx, j) = reshape(v_alt_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    end
                else
                    linear_L1 = d_in_L1 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_L1 - 1);
                    V_fordouter(level1ii, :, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    P_fordouter(level1ii, :, z_idx, e_idx, d_outer_c) = reshape(linear_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    if qh_mode == 2
                        Valt_fordouter(level1ii, :, z_idx, e_idx, d_outer_c) = reshape(v_alt_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    end
                end

                % --- Stage 2: Level 2 Bounded Monotonicity Refinement ---
                diff_a1p = max(0, a1p_idx_L1(:, 1, 2:end, :, :, :) - a1p_idx_L1(:, 1, 1:end-1, :, :, :));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 4, 5, 6]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(level1n - 1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                    if mg > 0
                        loweredge = min(a1p_idx_L1(:, 1, ii, :, :, :), N_a1 - mg);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets;
                    else
                        loweredge = a1p_idx_L1(:, 1, ii, :, :, :);
                        a1p_idx = loweredge;
                    end
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), [N_d_inner, mg+1, 1, N_a2, N_z_slice, N_e_slice]);

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, spec_n_bz, spec_n_e, d_val, a1p_grid_safe, a1_gridvals(curraindex), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, L, N_a2, N_z_slice, N_e_slice]);

                    if ez_flag
                        becareful2 = isfinite(RM_L2) & (RM_L2 ~= 0);
                        if has_ev_term
                            temp2_L2 = RM_L2;
                            temp2_L2(becareful2) = RM_L2(becareful2).^Dims.ezc2(j);
                            temp2_L2(RM_L2 == 0) = -Inf;
                            if N_e_SL_interp == 1 && N_e_slice > 1
                                d2a1pa2ze = d2_off_L2 + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*bz_off_L2 + ev_flat_offset;
                            else
                                d2a1pa2ze = d2_off_L2 + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*bz_off_L2 + N_d2*N_a1*N_a2*N_bothz*e_off_L2 + ev_flat_offset;
                            end
                            EV_L2 = reshape(DiscountedEV_d2(d2a1pa2ze(:)), [N_d_inner, mg+1, 1, N_a2, N_z_slice, N_e_slice]);
                            RHS_L2 = ezc1_j * temp2_L2 + beta0 * EV_L2;
                            temp5 = isfinite(RHS_L2) & (RHS_L2 ~= 0);
                            RHS_L2(temp5) = RHS_L2(temp5).^Dims.ezc7(j);
                            RHS_L2(RHS_L2 == 0) = -Inf;
                        else
                            RHS_L2 = RM_L2;
                            RHS_L2(becareful2) = (ezc1_j * RM_L2(becareful2).^Dims.ezc2(j)).^Dims.ezc7(j);
                            RHS_L2(RM_L2 == 0) = -Inf;
                        end
                    elseif has_cont
                        if N_e_SL_interp == 1 && N_e_slice > 1
                            d2a1pa2ze = d2_off_L2 + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*bz_off_L2 + ev_flat_offset;
                        else
                            d2a1pa2ze = d2_off_L2 + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*bz_off_L2 + N_d2*N_a1*N_a2*N_bothz*e_off_L2 + ev_flat_offset;
                        end
                        RHS_L2 = RM_L2 + beta0 * reshape(DiscountedEV_d2(d2a1pa2ze(:)), [N_d_inner, mg+1, 1, N_a2, N_z_slice, N_e_slice]);
                    else
                        RHS_L2 = RM_L2;
                    end

                    [v_tmp_L2, maxindex2] = max(reshape(RHS_L2, [N_d_inner * (mg+1), L, N_a2, N_z_slice, N_e_slice]), [], 1);

                    d_in_L2    = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    le_allind  = d_in_L2 + le_state_off;
                    a1p_abs    = loweredge(le_allind) + a1p_offset - 1;

                    if qh_mode == 2
                        if has_ev_term
                            d2_L2 = ceil(d_in_L2 / N_d1);
                            ev_L2_chosen = DiscountedEV_d2(d2_L2 + N_d2*(a1p_abs - 1) + ev_soph_off + ev_flat_offset);
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
                        pol_L2 = d_in_L2 + N_d_inner * (a1p_abs - 1);
                        V(curraindex, :, z_idx, e_idx, j)         = reshape(v_tmp_L2, [L, N_a2, N_z_slice, N_e_slice]);
                        Policy(1, curraindex, :, z_idx, e_idx, j) = reshape(pol_L2, [1, L, N_a2, N_z_slice, N_e_slice]);
                        if qh_mode == 2
                            Valt(curraindex, :, z_idx, e_idx, j) = reshape(v_alt_L2, [L, N_a2, N_z_slice, N_e_slice]);
                        end
                    else
                        linear_L2 = d_in_L2 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);
                        V_fordouter(curraindex, :, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L2, [L, N_a2, N_z_slice, N_e_slice]);
                        P_fordouter(curraindex, :, z_idx, e_idx, d_outer_c) = reshape(linear_L2, [L, N_a2, N_z_slice, N_e_slice]);
                        if qh_mode == 2
                            Valt_fordouter(curraindex, :, z_idx, e_idx, d_outer_c) = reshape(v_alt_L2, [L, N_a2, N_z_slice, N_e_slice]);
                        end
                    end
                end
            end
        end
    end

    if N_d_outer > 1
        [V(:,:,:,:,j), Policy(:,:,:,:,:,j), max_d_out] = extract_pol_local(V_fordouter, P_fordouter, N_a1, N_a2, N_bothz, N_e);
        if qh_mode == 2
            lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);
            Valt(:,:,:,:,j) = reshape(Valt_fordouter(lin_idx), [N_a1, N_a2, N_bothz, N_e]);
        end
    end
end

V = reshape(V, [N_a, N_bothz, N_e, N_j]);
Policy = reshape(Policy, [1, N_a, N_bothz, N_e, N_j]);
if qh_mode == 2
    Valt = reshape(Valt, [N_a, N_bothz, N_e, N_j]);
end

end % <--- MAIN FUNCTION ENDS HERE!

function [DiscountedEV, DiscountedEV_d2] = precalc_ev_local(EVpre_flat, df, pi_bz, N_e_interp, ExpAsset_J, Dims, vfoptions, j, WG_j)
    N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_bothz = Dims.N_bothz;
    N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d_inner = Dims.N_d_inner;

    a2p_idx_base    = ExpAsset_J.a2primeIndex;
    a2p_weight_base = ExpAsset_J.a2primeProbs;

    if ~isempty(EVpre_flat)
        EV_val = reshape(EVpre_flat, [N_a1*N_a2, N_bothz]);
        EV_raw = interp_expasset_local(EV_val, a2p_idx_base, a2p_weight_base, pi_bz, N_e_interp, Dims, vfoptions);
    else
        EV_raw = [];
    end

    if ~Dims.ez_flag
        EV_coarse = df * EV_raw;
    else
        sj_j = Dims.sj(j); ezc6_j = Dims.ezc6(j); ezc8_j = Dims.ezc8(j);
        if Dims.warmglow
            WG_val = repmat(WG_j, 1, N_bothz);
            WG_raw = interp_expasset_local(WG_val, a2p_idx_base, a2p_weight_base, pi_bz, N_e_interp, Dims, vfoptions);
        end
        if ~isempty(EV_raw)
            temp4 = EV_raw;
            if Dims.warmglow
                becareful = isfinite(temp4) & isfinite(WG_raw);
                temp4(becareful) = (sj_j * temp4(becareful).^ezc8_j + (1 - sj_j) * WG_raw(becareful).^ezc8_j).^ezc6_j;
                temp4((EV_raw == 0) & (WG_raw == 0)) = 0;
            else
                fin_mask = isfinite(temp4);
                temp4(fin_mask) = (sj_j * temp4(fin_mask).^ezc8_j).^ezc6_j;
                temp4(EV_raw == 0) = 0;
            end
        else
            temp4 = WG_raw;
            fin_mask = isfinite(temp4) & (temp4 ~= 0);
            temp4(fin_mask) = ((1 - sj_j) * temp4(fin_mask).^ezc8_j).^ezc6_j;
            temp4(WG_raw == 0) = 0;
        end
        EV_coarse = Dims.ezc3 * df * temp4;
        EV_coarse(isinf(temp4)) = -Inf;
    end

    N_d_exp  = size(EV_coarse, 1);
    N_a2_exp = size(EV_coarse, 4);
    if N_d_exp == 1 && N_d2 > 1;  EV_coarse = repmat(EV_coarse, [N_d2, 1, 1, 1, 1, 1]); end
    if N_a2_exp == 1 && N_a2 > 1; EV_coarse = repmat(EV_coarse, [1, 1, 1, N_a2, 1, 1]); end

    DiscountedEV_d2 = EV_coarse;
    if N_d1 > 1
        DiscountedEV = repelem(DiscountedEV_d2, N_d1, 1, 1, 1, 1, 1);
    else
        DiscountedEV = DiscountedEV_d2;
    end
    DiscountedEV = reshape(DiscountedEV, [N_d_inner, N_a1, 1, N_a2, N_bothz, N_e_interp]);
end

function EV_coarse = interp_expasset_local(EV_val, a2p_idx_base, a2p_weight_base, pi_bz, N_e_interp, Dims, vfoptions)
    N_a1 = Dims.N_a1; N_bothz = Dims.N_bothz; N_semiz = Dims.N_semiz; n_a2 = Dims.n_a2;
    if length(n_a2) == 1
        N_d_exp  = size(a2p_idx_base, 1);
        N_a2_exp = size(a2p_idx_base, 2);
        N_3rd    = max(size(a2p_idx_base, 3), 1);

        if Dims.exp_flag ~= 6 && N_3rd > 1 && N_3rd ~= N_bothz
            a2p_idx_base    = reshape(repmat(reshape(a2p_idx_base, [N_d_exp, N_a2_exp, 1, N_3rd]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
            a2p_weight_base = reshape(repmat(reshape(a2p_weight_base, [N_d_exp, N_a2_exp, 1, N_3rd]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
            N_3rd = N_bothz;
        end

        a1_base  = repelem(gpuArray(1:1:N_a1)', N_d_exp, N_a2_exp, N_3rd);
        idx_lo   = a1_base + N_a1 * repmat(a2p_idx_base - 1, N_a1, 1, 1);
        idx_hi   = a1_base + N_a1 * repmat(a2p_idx_base,     N_a1, 1, 1);
        prob_exp = repmat(a2p_weight_base, N_a1, 1, 1, N_bothz);

        Vlower = reshape(EV_val(idx_lo(:), :), [N_d_exp*N_a1, N_a2_exp, N_3rd, N_bothz]);
        Vupper = reshape(EV_val(idx_hi(:), :), [N_d_exp*N_a1, N_a2_exp, N_3rd, N_bothz]);

        prob_exp(Vlower == Vupper) = 0;
        EV_u = prob_exp .* Vlower + (1 - prob_exp) .* Vupper;
        EV_u(prob_exp == 0) = Vupper(prob_exp == 0);
        EV_u(prob_exp == 1) = Vlower(prob_exp == 1);
    else
        N_d_exp  = size(a2p_idx_base, 2);
        N_a2_exp = size(a2p_idx_base, 3);
        N_3rd    = max(size(a2p_idx_base, 4), 1);

        loIdx_1 = reshape(a2p_idx_base(1,:,:,:), [N_d_exp, N_a2_exp, N_3rd]);
        loIdx_2 = reshape(a2p_idx_base(2,:,:,:), [N_d_exp, N_a2_exp, N_3rd]);
        prob_1  = reshape(a2p_weight_base(1,:,:,:), [N_d_exp, N_a2_exp, N_3rd]);
        prob_2  = reshape(a2p_weight_base(2,:,:,:), [N_d_exp, N_a2_exp, N_3rd]);

        if Dims.exp_flag ~= 6 && N_3rd > 1 && N_3rd ~= N_bothz
            loIdx_1 = reshape(repmat(reshape(loIdx_1, [N_d_exp, N_a2_exp, 1, N_3rd]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
            loIdx_2 = reshape(repmat(reshape(loIdx_2, [N_d_exp, N_a2_exp, 1, N_3rd]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
            prob_1  = reshape(repmat(reshape(prob_1, [N_d_exp, N_a2_exp, 1, N_3rd]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
            prob_2  = reshape(repmat(reshape(prob_2, [N_d_exp, N_a2_exp, 1, N_3rd]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
            N_3rd = N_bothz;
        end

        a1_base = repelem(gpuArray(1:1:N_a1)', N_d_exp, N_a2_exp, N_3rd);
        n_a2_1  = n_a2(1);

        idx_ll = a1_base + N_a1 * repmat(loIdx_1 + n_a2_1*(loIdx_2-1) - 1,     N_a1, 1, 1);
        idx_hl = a1_base + N_a1 * repmat((loIdx_1+1) + n_a2_1*(loIdx_2-1) - 1, N_a1, 1, 1);
        idx_lh = a1_base + N_a1 * repmat(loIdx_1 + n_a2_1*loIdx_2 - 1,         N_a1, 1, 1);
        idx_hh = a1_base + N_a1 * repmat((loIdx_1+1) + n_a2_1*loIdx_2 - 1,     N_a1, 1, 1);

        V_ll = reshape(EV_val(idx_ll(:), :), [N_d_exp*N_a1, N_a2_exp, N_3rd, N_bothz]);
        V_hl = reshape(EV_val(idx_hl(:), :), [N_d_exp*N_a1, N_a2_exp, N_3rd, N_bothz]);
        V_lh = reshape(EV_val(idx_lh(:), :), [N_d_exp*N_a1, N_a2_exp, N_3rd, N_bothz]);
        V_hh = reshape(EV_val(idx_hh(:), :), [N_d_exp*N_a1, N_a2_exp, N_3rd, N_bothz]);

        prb1_full = repmat(prob_1, N_a1, 1, 1, N_bothz);
        prb2_full = repmat(prob_2, N_a1, 1, 1, N_bothz);

        p1_loy = prb1_full; p1_loy(V_ll == V_hl) = 0;
        c_ll = p1_loy .* V_ll; c_ll(isnan(c_ll)) = 0;
        c_hl = (1-p1_loy) .* V_hl; c_hl(isnan(c_hl)) = 0;
        EV_loy = c_ll + c_hl;

        p1_hiy = prb1_full; p1_hiy(V_lh == V_hh) = 0;
        c_lh = p1_hiy .* V_lh; c_lh(isnan(c_lh)) = 0;
        c_hh = (1-p1_hiy) .* V_hh; c_hh(isnan(c_hh)) = 0;
        EV_hiy = c_lh + c_hh;

        p2 = prb2_full; p2(EV_loy == EV_hiy) = 0;
        c_loy = p2 .* EV_loy; c_loy(isnan(c_loy)) = 0;
        c_hiy = (1-p2) .* EV_hiy; c_hiy(isnan(c_hiy)) = 0;
        EV_u = c_loy + c_hiy;
    end

    if Dims.exp_flag == 6
        pi_u = shiftdim(gpuArray(vfoptions.pi_u), -2);
        EV = squeeze(sum(EV_u .* pi_u, 3));
        EV = EV .* shiftdim(pi_bz', -2);
        EV(isnan(EV)) = 0;
        EV_coarse = squeeze(sum(EV, 3));
    elseif Dims.exp_flag == 3 || Dims.exp_flag == 4
        N_z = Dims.N_z;
        EV_u = reshape(EV_u, [N_d_exp*N_a1, N_a2_exp, N_z, N_e_interp, N_bothz]);
        pi_shift = reshape(pi_bz, [1, 1, N_z, 1, N_bothz]);
        EV = EV_u .* pi_shift;
        EV(isnan(EV)) = 0;
        EV_coarse = sum(EV, 5);
    elseif N_3rd > 1
        EV = EV_u .* shiftdim(pi_bz, -2);
        EV(isnan(EV)) = 0;
        EV_coarse = sum(EV, 4);
    else
        EV = reshape(EV_u, [N_d_exp*N_a1, N_a2_exp, N_bothz]) .* shiftdim(pi_bz', -2);
        EV(isnan(EV)) = 0;
        EV_coarse = sum(EV, 3);
    end
    EV_coarse = reshape(EV_coarse, [N_d_exp, N_a1, 1, N_a2_exp, N_bothz, N_e_interp]);
end

function [V_j, Pol_j, max_d_out] = extract_pol_local(V_fordouter, P_fordouter, N_a1, N_a2, N_bothz, N_e)
    N_a = N_a1 * N_a2;
    [V_j, max_d_out] = max(V_fordouter, [], 5);
    lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);
    Pol_j = reshape(P_fordouter(lin_idx), [1, N_a1, N_a2, N_bothz, N_e]);
end

function slice = get_grid_slice_local(grid_J, idx, j, has_var)
    if ~has_var || isempty(grid_J); slice = []; return; end
    if strcmp(idx, 'all')
        if ndims(grid_J) >= 3; slice = grid_J(:, :, j); else; slice = grid_J; end
    else
        if ndims(grid_J) >= 3; slice = grid_J(idx, :, j); else; slice = grid_J(idx, :); end
    end
end
