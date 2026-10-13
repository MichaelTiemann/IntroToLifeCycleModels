function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;

N_semiz_raw = Dims.N_semiz_raw > 0;
N_e_raw     = Dims.N_e_raw > 0;
N_z_raw     = Dims.special_n_z(1) > 0;
has_bothz   = N_semiz_raw || N_z_raw;
has_e       = N_e_raw;
has_semiz   = N_semiz_raw;

n_semiz = Dims.n_semiz;
n_e = Dims.n_e;

N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
n_d1 = Dims.n_d1; n_d2 = Dims.n_d2;
N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;

% ExpAsset GI2A specific split: a1_c, a2_c (choices) vs a_exp
n_a1_c = Dims.n_a1(1);
n_a2_c = max(prod(Dims.n_a1(2:end)), 1);
n_exp = Dims.n_a2; N_exp = Dims.N_a2;

a1_c_gridvals = CreateGridvals(n_a1_c, vfoptions.a1_grid(1:n_a1_c), 1);
a2_c_gridvals = CreateGridvals(Dims.n_a1(2:end), vfoptions.a1_grid(n_a1_c+1:end), 1);
exp_gridvals  = CreateGridvals(n_exp, vfoptions.a2_grid, 1);

n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
a1prime_grid_col = gpuArray(interp1(1:1:n_a1_c, gather(a1_c_gridvals), linspace(1, n_a1_c, n_a1_c+(n_a1_c-1)*n2short))');
a1prime_grid_col = reshape(a1prime_grid_col, [], 1);
N_a1fine = length(a1prime_grid_col);

% --- Cache Setup ---
RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer);
last_ReturnFnParamsVec = [];

% --- 2. Determine Iteration Strategy ---
if vfoptions.lowmemory == 0
    IterStrategy = 'Full';
elseif vfoptions.lowmemory == 1
    if has_e; IterStrategy = 'Loop_e';
    elseif has_semiz && has_z; IterStrategy = 'Loop_z_parallel_semiz';
    else; IterStrategy = 'Loop_bothz'; end
elseif vfoptions.lowmemory == 2
    if has_e && has_semiz && has_z; IterStrategy = 'Loop_z_parallel_semiz_e';
    elseif has_e; IterStrategy = 'Loop_bothz_e';
    elseif has_semiz && has_z; IterStrategy = 'Loop_bothz';
    else; error('Invalid lowmemory=2 for this shock combination.'); end
elseif vfoptions.lowmemory == 3
    if has_e && has_semiz && has_z; IterStrategy = 'Loop_bothz_e';
    else; error('Invalid lowmemory=3 for this shock combination.'); end
end

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 3. Introspect ExpAsset Parameters ---
if Dims.exp_flag > 0
    l_d2_len = length(n_d2); if sum(n_d2) == 0; l_d2_len = 0; end
    l_exp_len = length(n_exp); if sum(n_exp) == 0; l_exp_len = 0; end
    l_z_len  = length(n_z);  if sum(n_z) == 0;  l_z_len = 0; end
    l_e_len  = length(n_e);  if sum(n_e) == 0;  l_e_len = 0; end
    l_semiz_len = length(n_semiz); if sum(n_semiz) == 0; l_semiz_len = 0; end
    if isfield(vfoptions, 'n_u'); l_u_len = length(vfoptions.n_u); if sum(vfoptions.n_u) == 0; l_u_len = 0; end; else; l_u_len = 0; end

    if Dims.exp_flag == 3 || Dims.exp_flag == 4; num_state_args = l_d2_len + l_exp_len + l_z_len + l_e_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 2; num_state_args = l_d2_len + l_exp_len + l_z_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 5; num_state_args = l_d2_len + l_exp_len + l_semiz_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 6; num_state_args = l_d2_len + l_exp_len + l_u_len + (l_exp_len >= 2);
    else; num_state_args = l_d2_len + l_exp_len + (l_exp_len >= 2); end

    temp_args = getAnonymousFnInputNames(vfoptions.aprimeFn);
    if length(temp_args) > num_state_args; aprimeFnParamNames = temp_args(num_state_args + 1 : end); else; aprimeFnParamNames = {}; end
else
    aprimeFnParamNames = {};
end

% --- 4. Preallocate Outputs & GPU Constants ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');

n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1, 1, 1, 1]);
d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));
d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1, 1, 1]);
a2p_off_L2 = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1, 1, 1, 1, 1]);
exp_off_L2 = reshape(gpuArray(0:N_exp-1), [1, 1, 1, 1, 1, N_exp, 1, 1]);

% --- 5. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]); else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    has_cont = ~isempty(V_next);
    if has_cont
        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, j);
        d2_gridvals = CreateGridvals(n_d2, vfoptions.d2_grid, 1);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, Grids.z_gridvals_J(:,:,j), n_e, get_grid_slice(Grids.e_gridvals_J, 'all', j, N_e_raw));
        elseif Dims.exp_flag == 2
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, z_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 5
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_semiz, get_grid_slice(vfoptions.semiz_gridvals_J, 'all', j, N_semiz_raw));
        elseif Dims.exp_flag == 6
            u_gridvals = CreateGridvals(vfoptions.n_u, vfoptions.u_grid, 1);
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, vfoptions.n_u, u_gridvals);
        else
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2);
        end
        ExpAsset_J.a2primeIndex = expIndex;
        ExpAsset_J.a2primeProbs = expProbs;
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
% SPECIALIZED SOLVE FUNCTIONS (Vectorised over N_d_inner and n_a2_c)
% =====================================================================
    function [V_j, Pol_j] = solve_period_full(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec)
            RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer);
            last_ReturnFnParamsVec = ReturnFnParamsVec;
        end
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if isempty(RM_cache{1, 1, d_outer_c})
                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_cache{1, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, n_a1_c, n_a2_c, N_exp, N_bothz, N_e]);
            end
            RM_L1 = RM_cache{1, 1, d_outer_c};

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims, a1_c_gridvals, a1prime_grid_col);
                EV_fine_flat = reshape(EV_fine, [N_d2 * N_a1fine * n_a2_c * N_exp * N_bothz * N_e_interp, 1]);
                DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_bothz, N_e_interp]);
                RHS_L1 = RM_L1 + DiscountedEV_L1;
            else
                RHS_L1 = RM_L1;
            end

            % Phase 1: Maximize STRICTLY over a1prime (dimension 2)
            [~, maxindex1] = max(RHS_L1, [], 2);
            midpoint = max(min(maxindex1, n_a1_c-1), 2);

            a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
            a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_bothz, N_e]);

            RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
            RM_L2 = reshape(RM_L2, [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_bothz, N_e]);

            if has_cont
                bz_off_L2 = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, 1, N_bothz, 1]);
                e_off_L2  = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, 1, 1, N_e_interp]);
                EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2p_off_L2 + N_d2*N_a1fine*n_a2_c*exp_off_L2 + N_d2*N_a1fine*n_a2_c*N_exp*bz_off_L2 + N_d2*N_a1fine*n_a2_c*N_exp*N_bothz*e_off_L2;
                RHS_L2 = RM_L2 + df * EV_fine_flat(EV_idx);
            else
                RHS_L2 = RM_L2;
            end

            RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * n_a2_c, n_a1_c, n_a2_c, N_exp, N_bothz, N_e]);
            [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

            d_ind = rem(p_tmp - 1, N_d_inner) + 1;
            L2_and_a2p = ceil(p_tmp / N_d_inner);
            L2offset = rem(L2_and_a2p - 1, n2long) + 1;
            a2p_idx_chosen = ceil(L2_and_a2p / n2long);

            S1 = N_d_inner;
            S2 = S1 * n2long;
            S3 = S2 * n_a2_c;
            S4 = S3 * n_a1_c;
            S5 = S4 * n_a2_c;
            S6 = S5 * N_exp;
            S7 = S6 * N_bothz;

            a1_off  = reshape(gpuArray(0:n_a1_c-1),    [1, n_a1_c, 1, 1, 1, 1]);
            a2_off  = reshape(gpuArray(0:n_a2_c-1),    [1, 1, n_a2_c, 1, 1, 1]);
            exp_off = reshape(gpuArray(0:N_exp-1),     [1, 1, 1, N_exp, 1, 1]);
            bz_off  = reshape(gpuArray(0:N_bothz-1),   [1, 1, 1, 1, N_bothz, 1]);
            e_off   = reshape(gpuArray(0:N_e-1),       [1, 1, 1, 1, 1, N_e]);

            linidx_lower = d_ind + S1*(1 - 1)      + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*bz_off + S7*e_off;
            linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*bz_off + S7*e_off;

            isInfLower = (RM_L2(linidx_lower) == -Inf);
            isInfUpper = (RM_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            S_mid1 = N_d_inner;
            S_mid2 = S_mid1 * 1;
            S_mid3 = S_mid2 * n_a2_c;
            S_mid4 = S_mid3 * n_a1_c;
            S_mid5 = S_mid4 * n_a2_c;
            S_mid6 = S_mid5 * N_exp;
            S_mid7 = S_mid6 * N_bothz;

            midpt_idx = d_ind + S_mid1*(1 - 1) + S_mid2*(a2p_idx_chosen - 1) + S_mid3*a1_off + S_mid4*a2_off + S_mid5*exp_off + S_mid6*bz_off + S_mid7*e_off;
            chosen_midpt = reshape(midpoint(midpt_idx), [1, n_a1_c, n_a2_c, N_exp, N_bothz, N_e]);

            joint_a_idx = chosen_midpt + n_a1_c * (a2p_idx_chosen - 1);
            d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

            V_fordouter(:,:,:,:,d_outer_c) = reshape(v_tmp, [N_a, N_bothz, N_e]);
            P1_fordouter(:,:,:,:,d_outer_c) = reshape(d_idx_val, [N_a, N_bothz, N_e]);
            P2_fordouter(:,:,:,:,d_outer_c) = reshape(joint_a_idx, [N_a, N_bothz, N_e]);
            P3_fordouter(:,:,:,:,d_outer_c) = reshape(L2offset, [N_a, N_bothz, N_e]);
            P4_fordouter(:,:,:,:,d_outer_c) = reshape(flag_val, [N_a, N_bothz, N_e]);
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec)
            RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer);
            last_ReturnFnParamsVec = ReturnFnParamsVec;
        end
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims, a1_c_gridvals, a1prime_grid_col);
            end

            for e_c = 1:N_e
                e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                if isempty(RM_cache{1, e_c, d_outer_c})
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{1, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, n_a1_c, n_a2_c, N_exp, N_bothz, 1]);
                end
                RM_L1 = RM_cache{1, e_c, d_outer_c};

                if has_cont
                    e_idx = min(e_c, N_e_interp);
                    EV_fine_flat_e = reshape(EV_fine(:,:,:,:,:,e_idx), [N_d2 * N_a1fine * n_a2_c * N_exp * N_bothz, 1]);
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, e_idx), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_bothz, 1]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                else
                    RHS_L1 = RM_L1;
                end

                [~, maxindex1] = max(RHS_L1, [], 2);
                midpoint = max(min(maxindex1, n_a1_c-1), 2);

                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_bothz, 1]);

                RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_bothz, 1]);

                if has_cont
                    bz_off_L2 = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, 1, N_bothz, 1]);
                    EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2p_off_L2 + N_d2*N_a1fine*n_a2_c*exp_off_L2 + N_d2*N_a1fine*n_a2_c*N_exp*bz_off_L2;
                    RHS_L2 = RM_L2 + df * EV_fine_flat_e(EV_idx);
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * n_a2_c, n_a1_c, n_a2_c, N_exp, N_bothz, 1]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                S1 = N_d_inner;
                S2 = S1 * n2long;
                S3 = S2 * n_a2_c;
                S4 = S3 * n_a1_c;
                S5 = S4 * n_a2_c;
                S6 = S5 * N_exp;

                a1_off  = reshape(gpuArray(0:n_a1_c-1),    [1, n_a1_c, 1, 1, 1]);
                a2_off  = reshape(gpuArray(0:n_a2_c-1),    [1, 1, n_a2_c, 1, 1]);
                exp_off = reshape(gpuArray(0:N_exp-1),     [1, 1, 1, N_exp, 1]);
                bz_off  = reshape(gpuArray(0:N_bothz-1),   [1, 1, 1, 1, N_bothz]);

                linidx_lower = d_ind + S1*(1 - 1)      + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*bz_off;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*bz_off;

                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                S_mid1 = N_d_inner;
                S_mid2 = S_mid1 * 1;
                S_mid3 = S_mid2 * n_a2_c;
                S_mid4 = S_mid3 * n_a1_c;
                S_mid5 = S_mid4 * n_a2_c;
                S_mid6 = S_mid5 * N_exp;

                midpt_idx = d_ind + S_mid1*(1 - 1) + S_mid2*(a2p_idx_chosen - 1) + S_mid3*a1_off + S_mid4*a2_off + S_mid5*exp_off + S_mid6*bz_off;
                chosen_midpt = reshape(midpoint(midpt_idx), [1, n_a1_c, n_a2_c, N_exp, N_bothz, 1]);

                joint_a_idx = chosen_midpt + n_a1_c * (a2p_idx_chosen - 1);
                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:,:,:,e_c,d_outer_c) = reshape(v_tmp, [N_a, N_bothz, 1]);
                P1_fordouter(:,:,:,e_c,d_outer_c) = reshape(d_idx_val, [N_a, N_bothz, 1]);
                P2_fordouter(:,:,:,e_c,d_outer_c) = reshape(joint_a_idx, [N_a, N_bothz, 1]);
                P3_fordouter(:,:,:,e_c,d_outer_c) = reshape(L2offset, [N_a, N_bothz, 1]);
                P4_fordouter(:,:,:,e_c,d_outer_c) = reshape(flag_val, [N_a, N_bothz, 1]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec)
            RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer);
            last_ReturnFnParamsVec = ReturnFnParamsVec;
        end
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims, a1_c_gridvals, a1prime_grid_col);
                EV_coarse = reshape(EV_coarse, [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_semiz, N_z, N_e_interp]);
                EV_fine   = reshape(EV_fine,   [N_d2, N_a1fine, n_a2_c, N_exp, N_semiz, N_z, N_e_interp]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                if isempty(RM_cache{z_c, 1, d_outer_c})
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{z_c, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, n_a1_c, n_a2_c, N_exp, N_semiz, N_e]);
                end
                RM_L1 = RM_cache{z_c, 1, d_outer_c};

                if has_cont
                    EV_fine_flat_z = reshape(EV_fine(:,:,:,:,:,z_c,:), [N_d2 * N_a1fine * n_a2_c * N_exp * N_semiz * N_e_interp, 1]);
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :, :, z_c, :), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_semiz, N_e_interp]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                else
                    RHS_L1 = RM_L1;
                end

                [~, maxindex1] = max(RHS_L1, [], 2);
                midpoint = max(min(maxindex1, n_a1_c-1), 2);

                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_semiz, N_e]);

                RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_semiz, N_e]);

                if has_cont
                    sb_off_L2 = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, 1, N_semiz, 1]);
                    e_off_L2  = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, 1, 1, N_e_interp]);
                    EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2p_off_L2 + N_d2*N_a1fine*n_a2_c*exp_off_L2 + N_d2*N_a1fine*n_a2_c*N_exp*sb_off_L2 + N_d2*N_a1fine*n_a2_c*N_exp*N_semiz*e_off_L2;
                    RHS_L2 = RM_L2 + df * EV_fine_flat_z(EV_idx);
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * n_a2_c, n_a1_c, n_a2_c, N_exp, N_semiz, N_e]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                S1 = N_d_inner;
                S2 = S1 * n2long;
                S3 = S2 * n_a2_c;
                S4 = S3 * n_a1_c;
                S5 = S4 * n_a2_c;
                S6 = S5 * N_exp;
                S7 = S6 * N_semiz;

                a1_off  = reshape(gpuArray(0:n_a1_c-1),    [1, n_a1_c, 1, 1, 1, 1]);
                a2_off  = reshape(gpuArray(0:n_a2_c-1),    [1, 1, n_a2_c, 1, 1, 1]);
                exp_off = reshape(gpuArray(0:N_exp-1),     [1, 1, 1, N_exp, 1, 1]);
                sb_off  = reshape(gpuArray(0:N_semiz-1),   [1, 1, 1, 1, N_semiz, 1]);
                e_off   = reshape(gpuArray(0:N_e-1),       [1, 1, 1, 1, 1, N_e]);

                linidx_lower = d_ind + S1*(1 - 1)      + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*sb_off + S7*e_off;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*sb_off + S7*e_off;

                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                S_mid1 = N_d_inner;
                S_mid2 = S_mid1 * 1;
                S_mid3 = S_mid2 * n_a2_c;
                S_mid4 = S_mid3 * n_a1_c;
                S_mid5 = S_mid4 * n_a2_c;
                S_mid6 = S_mid5 * N_exp;
                S_mid7 = S_mid6 * N_semiz;

                midpt_idx = d_ind + S_mid1*(1 - 1) + S_mid2*(a2p_idx_chosen - 1) + S_mid3*a1_off + S_mid4*a2_off + S_mid5*exp_off + S_mid6*sb_off + S_mid7*e_off;
                chosen_midpt = reshape(midpoint(midpt_idx), [1, n_a1_c, n_a2_c, N_exp, N_semiz, N_e]);

                joint_a_idx = chosen_midpt + n_a1_c * (a2p_idx_chosen - 1);
                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:, sb, :, d_outer_c) = reshape(v_tmp, [N_a, N_semiz, N_e]);
                P1_fordouter(:, sb, :, d_outer_c) = reshape(d_idx_val, [N_a, N_semiz, N_e]);
                P2_fordouter(:, sb, :, d_outer_c) = reshape(joint_a_idx, [N_a, N_semiz, N_e]);
                P3_fordouter(:, sb, :, d_outer_c) = reshape(L2offset, [N_a, N_semiz, N_e]);
                P4_fordouter(:, sb, :, d_outer_c) = reshape(flag_val, [N_a, N_semiz, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec)
            RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer);
            last_ReturnFnParamsVec = ReturnFnParamsVec;
        end
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims, a1_c_gridvals, a1prime_grid_col);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                if isempty(RM_cache{z_c, 1, d_outer_c})
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{z_c, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, n_a1_c, n_a2_c, N_exp, 1, N_e]);
                end
                RM_L1 = RM_cache{z_c, 1, d_outer_c};

                if has_cont
                    EV_fine_flat_z = reshape(EV_fine(:,:,:,:,z_c,:), [N_d2 * N_a1fine * n_a2_c * N_exp * N_e_interp, 1]);
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, z_c, :), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, 1, N_e_interp]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                else
                    RHS_L1 = RM_L1;
                end

                [~, maxindex1] = max(RHS_L1, [], 2);
                midpoint = max(min(maxindex1, n_a1_c-1), 2);

                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, 1, N_e]);

                RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, 1, N_e]);

                if has_cont
                    e_off_L2  = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, 1, 1, N_e_interp]);
                    EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2p_off_L2 + N_d2*N_a1fine*n_a2_c*exp_off_L2 + N_d2*N_a1fine*n_a2_c*N_exp*e_off_L2;
                    RHS_L2 = RM_L2 + df * EV_fine_flat_z(EV_idx);
                else
                    RHS_L2 = RM_L2;
                end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * n_a2_c, n_a1_c, n_a2_c, N_exp, 1, N_e]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p = ceil(p_tmp / N_d_inner);
                L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                S1 = N_d_inner;
                S2 = S1 * n2long;
                S3 = S2 * n_a2_c;
                S4 = S3 * n_a1_c;
                S5 = S4 * n_a2_c;
                S6 = S5 * N_exp;

                a1_off  = reshape(gpuArray(0:n_a1_c-1),    [1, n_a1_c, 1, 1, 1]);
                a2_off  = reshape(gpuArray(0:n_a2_c-1),    [1, 1, n_a2_c, 1, 1]);
                exp_off = reshape(gpuArray(0:N_exp-1),     [1, 1, 1, N_exp, 1]);
                e_off   = reshape(gpuArray(0:N_e-1),       [1, 1, 1, 1, N_e]);

                linidx_lower = d_ind + S1*(1 - 1)      + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*e_off;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*e_off;

                isInfLower = (RM_L2(linidx_lower) == -Inf);
                isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                S_mid1 = N_d_inner;
                S_mid2 = S_mid1 * 1;
                S_mid3 = S_mid2 * n_a2_c;
                S_mid4 = S_mid3 * n_a1_c;
                S_mid5 = S_mid4 * n_a2_c;
                S_mid6 = S_mid5 * N_exp;

                midpt_idx = d_ind + S_mid1*(1 - 1) + S_mid2*(a2p_idx_chosen - 1) + S_mid3*a1_off + S_mid4*a2_off + S_mid5*exp_off + S_mid6*e_off;
                chosen_midpt = reshape(midpoint(midpt_idx), [1, n_a1_c, n_a2_c, N_exp, 1, N_e]);

                joint_a_idx = chosen_midpt + n_a1_c * (a2p_idx_chosen - 1);
                d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:, z_c, :, d_outer_c) = reshape(v_tmp, [N_a, 1, N_e]);
                P1_fordouter(:, z_c, :, d_outer_c) = reshape(d_idx_val, [N_a, 1, N_e]);
                P2_fordouter(:, z_c, :, d_outer_c) = reshape(joint_a_idx, [N_a, 1, N_e]);
                P3_fordouter(:, z_c, :, d_outer_c) = reshape(L2offset, [N_a, 1, N_e]);
                P4_fordouter(:, z_c, :, d_outer_c) = reshape(flag_val, [N_a, 1, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec)
            RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer);
            last_ReturnFnParamsVec = ReturnFnParamsVec;
        end
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims, a1_c_gridvals, a1prime_grid_col);
                EV_coarse = reshape(EV_coarse, [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_semiz, N_z, N_e_interp]);
                EV_fine   = reshape(EV_fine,   [N_d2, N_a1fine, n_a2_c, N_exp, N_semiz, N_z, N_e_interp]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    if isempty(RM_cache{z_c, e_c, d_outer_c})
                        RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                        RM_cache{z_c, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, n_a1_c, n_a2_c, N_exp, N_semiz, 1]);
                    end
                    RM_L1 = RM_cache{z_c, e_c, d_outer_c};

                    if has_cont
                        e_idx = min(e_c, N_e_interp);
                        EV_fine_flat_ze = reshape(EV_fine(:,:,:,:,:,z_c,e_idx), [N_d2 * N_a1fine * n_a2_c * N_exp * N_semiz, 1]);
                        DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :, :, z_c, e_idx), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_semiz, 1]);
                        RHS_L1 = RM_L1 + DiscountedEV_L1;
                    else
                        RHS_L1 = RM_L1;
                    end

                    [~, maxindex1] = max(RHS_L1, [], 2);
                    midpoint = max(min(maxindex1, n_a1_c-1), 2);

                    a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                    a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_semiz, 1]);

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_semiz, 1]);

                    if has_cont
                        sb_off_L2 = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, 1, N_semiz, 1]);
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2p_off_L2 + N_d2*N_a1fine*n_a2_c*exp_off_L2 + N_d2*N_a1fine*n_a2_c*N_exp*sb_off_L2;
                        RHS_L2 = RM_L2 + df * EV_fine_flat_ze(EV_idx);
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * n_a2_c, n_a1_c, n_a2_c, N_exp, N_semiz, 1]);
                    [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                    d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                    L2_and_a2p = ceil(p_tmp / N_d_inner);
                    L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                    a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                    S1 = N_d_inner;
                    S2 = S1 * n2long;
                    S3 = S2 * n_a2_c;
                    S4 = S3 * n_a1_c;
                    S5 = S4 * n_a2_c;
                    S6 = S5 * N_exp;

                    a1_off  = reshape(gpuArray(0:n_a1_c-1), [1, n_a1_c, 1, 1, 1]);
                    a2_off  = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1, 1]);
                    exp_off = reshape(gpuArray(0:N_exp-1),  [1, 1, 1, N_exp, 1]);
                    sb_off  = reshape(gpuArray(0:N_semiz-1),[1, 1, 1, 1, N_semiz]);

                    linidx_lower = d_ind + S1*(1 - 1)      + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*sb_off;
                    linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off + S6*sb_off;

                    isInfLower = (RM_L2(linidx_lower) == -Inf);
                    isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    S_mid1 = N_d_inner;
                    S_mid2 = S_mid1 * 1;
                    S_mid3 = S_mid2 * n_a2_c;
                    S_mid4 = S_mid3 * n_a1_c;
                    S_mid5 = S_mid4 * n_a2_c;
                    S_mid6 = S_mid5 * N_exp;

                    midpt_idx = d_ind + S_mid1*(1 - 1) + S_mid2*(a2p_idx_chosen - 1) + S_mid3*a1_off + S_mid4*a2_off + S_mid5*exp_off + S_mid6*sb_off;
                    chosen_midpt = reshape(midpoint(midpt_idx), [1, n_a1_c, n_a2_c, N_exp, N_semiz, 1]);

                    joint_a_idx = chosen_midpt + n_a1_c * (a2p_idx_chosen - 1);
                    d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                    V_fordouter(:, sb, e_c, d_outer_c) = reshape(v_tmp, [N_a, N_semiz, 1]);
                    P1_fordouter(:, sb, e_c, d_outer_c) = reshape(d_idx_val, [N_a, N_semiz, 1]);
                    P2_fordouter(:, sb, e_c, d_outer_c) = reshape(joint_a_idx, [N_a, N_semiz, 1]);
                    P3_fordouter(:, sb, e_c, d_outer_c) = reshape(L2offset, [N_a, N_semiz, 1]);
                    P4_fordouter(:, sb, e_c, d_outer_c) = reshape(flag_val, [N_a, N_semiz, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec)
            RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer);
            last_ReturnFnParamsVec = ReturnFnParamsVec;
        end
        has_cont = ~isempty(V_next);
        N_e_interp = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp = 1; end

        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [n_a1_c * n_a2_c * N_exp, N_bothz]);
        end

        V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims, a1_c_gridvals, a1prime_grid_col);
            end

            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);

                for e_c = 1:N_e
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);

                    if isempty(RM_cache{z_c, e_c, d_outer_c})
                        RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                        RM_cache{z_c, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, n_a1_c, n_a2_c, N_exp, 1, 1]);
                    end
                    RM_L1 = RM_cache{z_c, e_c, d_outer_c};

                    if has_cont
                        e_idx = min(e_c, N_e_interp);
                        EV_fine_flat_ze = reshape(EV_fine(:,:,:,:,z_c,e_idx), [N_d2 * N_a1fine * n_a2_c * N_exp, 1]);
                        DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, z_c, e_idx), [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, 1, 1]);
                        RHS_L1 = RM_L1 + DiscountedEV_L1;
                    else
                        RHS_L1 = RM_L1;
                    end

                    [~, maxindex1] = max(RHS_L1, [], 2);
                    midpoint = max(min(maxindex1, n_a1_c-1), 2);

                    a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                    a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, 1, 1]);

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, 1, 1]);

                    if has_cont
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2p_off_L2 + N_d2*N_a1fine*n_a2_c*exp_off_L2;
                        RHS_L2 = RM_L2 + df * EV_fine_flat_ze(EV_idx);
                    else
                        RHS_L2 = RM_L2;
                    end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long * n_a2_c, n_a1_c, n_a2_c, N_exp, 1, 1]);
                    [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                    d_ind = rem(p_tmp - 1, N_d_inner) + 1;
                    L2_and_a2p = ceil(p_tmp / N_d_inner);
                    L2offset = rem(L2_and_a2p - 1, n2long) + 1;
                    a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                    S1 = N_d_inner;
                    S2 = S1 * n2long;
                    S3 = S2 * n_a2_c;
                    S4 = S3 * n_a1_c;
                    S5 = S4 * n_a2_c;

                    a1_off  = reshape(gpuArray(0:n_a1_c-1), [1, n_a1_c, 1, 1]);
                    a2_off  = reshape(gpuArray(0:n_a2_c-1), [1, 1, n_a2_c, 1]);
                    exp_off = reshape(gpuArray(0:N_exp-1),  [1, 1, 1, N_exp]);

                    linidx_lower = d_ind + S1*(1 - 1)      + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off;
                    linidx_upper = d_ind + S1*(n2long - 1) + S2*(a2p_idx_chosen - 1) + S3*a1_off + S4*a2_off + S5*exp_off;

                    isInfLower = (RM_L2(linidx_lower) == -Inf);
                    isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    S_mid1 = N_d_inner;
                    S_mid2 = S_mid1 * 1;
                    S_mid3 = S_mid2 * n_a2_c;
                    S_mid4 = S_mid3 * n_a1_c;
                    S_mid5 = S_mid4 * n_a2_c;

                    midpt_idx = d_ind + S_mid1*(1 - 1) + S_mid2*(a2p_idx_chosen - 1) + S_mid3*a1_off + S_mid4*a2_off + S_mid5*exp_off;
                    chosen_midpt = reshape(midpoint(midpt_idx), [1, n_a1_c, n_a2_c, N_exp, 1, 1]);

                    joint_a_idx = chosen_midpt + n_a1_c * (a2p_idx_chosen - 1);
                    d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                    V_fordouter(:, z_c, e_c, d_outer_c) = reshape(v_tmp, [N_a, 1, 1]);
                    P1_fordouter(:, z_c, e_c, d_outer_c) = reshape(d_idx_val, [N_a, 1, 1]);
                    P2_fordouter(:, z_c, e_c, d_outer_c) = reshape(joint_a_idx, [N_a, 1, 1]);
                    P3_fordouter(:, z_c, e_c, d_outer_c) = reshape(L2offset, [N_a, 1, 1]);
                    P4_fordouter(:, z_c, e_c, d_outer_c) = reshape(flag_val, [N_a, 1, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function val = get_grid_slice(grid, idx, j, has_flag)
        if has_flag
            if ischar(idx)
                val = grid(:,:,j);
            else
                val = grid(idx,:,j);
            end
        else
            val = [];
        end
    end

    function [V_out, Pol_out] = extract_pol(V_douter, P1, P2, P3, P4, N_a, N_bothz, N_e, N_d_outer)
        if N_d_outer > 1
            [V_out, maxidx] = max(V_douter, [], 5);
            lin_idx = (1:(N_a*N_bothz*N_e))' + (N_a*N_bothz*N_e)*(maxidx(:) - 1);
            Pol_out = zeros(4, N_a, N_bothz, N_e, 'gpuArray');
            Pol_out(1,:,:,:) = reshape(P1(lin_idx), [1, N_a, N_bothz, N_e]);
            Pol_out(2,:,:,:) = reshape(P2(lin_idx), [1, N_a, N_bothz, N_e]);
            Pol_out(3,:,:,:) = reshape(P3(lin_idx), [1, N_a, N_bothz, N_e]);
            Pol_out(4,:,:,:) = reshape(P4(lin_idx), [1, N_a, N_bothz, N_e]);
        else
            V_out = reshape(V_douter, [N_a, N_bothz, N_e]);
            Pol_out = zeros(4, N_a, N_bothz, N_e, 'gpuArray');
            Pol_out(1,:,:,:) = reshape(P1, [1, N_a, N_bothz, N_e]);
            Pol_out(2,:,:,:) = reshape(P2, [1, N_a, N_bothz, N_e]);
            Pol_out(3,:,:,:) = reshape(P3, [1, N_a, N_bothz, N_e]);
            Pol_out(4,:,:,:) = reshape(P4, [1, N_a, N_bothz, N_e]);
        end
    end

    function [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J, n_a1_c, n_a2_c, n_exp, N_d2, N_bothz, vfoptions, Dims, a1_c_gridvals, a1prime_grid_col)
        EVinf = (EVpre_flat == -Inf);
        EV_val = EVpre_flat;
        EV_val(EVinf) = -1e250;
        EV_expected = EV_val * pi_bz';
        EV_expected((EVinf * pi_bz') > 0) = -Inf;

        % ExpAsset shape logic
        EV_reshaped = reshape(EV_expected, [1, n_a1_c, n_a2_c, N_exp, N_bothz, N_e_interp]);

        a2p = ExpAsset_J.a2primeIndex;
        a2p_probs = ExpAsset_J.a2primeProbs;
        N_prob = size(a2p, 2);

        EV_coarse = zeros(N_d2, n_a1_c, n_a2_c, N_exp, N_bothz, N_e_interp, 'gpuArray');
        for u_idx = 1:N_prob
            a2p_u = a2p(:, u_idx, :, :); % [d2, 1, bz, e]
            EV_coarse = EV_coarse + a2p_probs(u_idx) .* EV_reshaped(1, :, :, a2p_u, :, :);
        end

        % Standard models expect V(a1', a2', a_exp', z'). Interpolate over a1' grid.
        EV_perm = permute(EV_coarse, [2, 1, 3, 4, 5, 6]); % [n_a1_c, N_d2, n_a2_c, N_exp, N_bothz, N_e_interp]
        EV_interp_next = interp1(a1_c_gridvals, EV_perm, a1prime_grid_col);
        EV_fine = permute(EV_interp_next, [2, 1, 3, 4, 5, 6]); % [N_d2, n_a1fine, n_a2_c, N_exp, N_bothz, N_e_interp]
    end


end
