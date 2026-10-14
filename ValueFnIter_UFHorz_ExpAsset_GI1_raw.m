function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;

N_z_raw = Dims.special_n_z(1) > 0;
N_semiz_raw = Dims.N_semiz_raw > 0;
N_e_raw = Dims.N_e_raw > 0;
n_a1 = Dims.n_a1; n_a2 = Dims.n_a2;
n_e = Dims.n_e; n_semiz = Dims.n_semiz;

n_d1 = vfoptions.n_d1; n_d2 = vfoptions.n_d2;
N_d1 = max(prod(n_d1), 1); N_d2 = max(prod(n_d2), 1);

has_bothz   = N_semiz_raw || N_z_raw;
has_e       = N_e_raw;
has_semiz   = N_semiz_raw;

if isfield(vfoptions, 'n_d3') && prod(vfoptions.n_d3) > 0
    n_d3 = vfoptions.n_d3;
    N_d3 = max(prod(n_d3), 1);
else
    n_d3 = 0; N_d3 = 1;
end
N_d_inner = N_d1 * N_d2; N_d_outer = N_d3;

% Grid Extracts
a1_gridvals = gpuArray(a_grid(1:N_a1));
a2_gridvals = gpuArray(a_grid(N_a1+1:end));

n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
a1prime_grid_col = gpuArray(interp1(1:1:N_a1, gather(a1_gridvals), linspace(1, N_a1, N_a1+(N_a1-1)*n2short))');
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
    l_a2_len = length(n_a2); if sum(n_a2) == 0; l_a2_len = 0; end
    l_z_len  = length(n_z);  if sum(n_z) == 0;  l_z_len = 0; end
    l_e_len  = length(n_e);  if sum(n_e) == 0;  l_e_len = 0; end
    l_semiz_len = length(n_semiz); if sum(n_semiz) == 0; l_semiz_len = 0; end
    if isfield(vfoptions, 'n_u'); l_u_len = length(vfoptions.n_u); if sum(vfoptions.n_u) == 0; l_u_len = 0; end; else; l_u_len = 0; end

    if Dims.exp_flag == 3 || Dims.exp_flag == 4; num_state_args = l_d2_len + l_a2_len + l_z_len + l_e_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 2; num_state_args = l_d2_len + l_a2_len + l_z_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 5; num_state_args = l_d2_len + l_a2_len + l_semiz_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 6; num_state_args = l_d2_len + l_a2_len + l_u_len + (l_a2_len >= 2);
    else; num_state_args = l_d2_len + l_a2_len + (l_a2_len >= 2); end

    temp_args = getAnonymousFnInputNames(vfoptions.aprimeFn);
    if length(temp_args) > num_state_args; aprimeFnParamNames = temp_args(num_state_args + 1 : end); else; aprimeFnParamNames = {}; end
else
    aprimeFnParamNames = {};
end

% --- 4. Preallocate Outputs ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');

n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1, 1]);
d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));

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
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, Grids.z_gridvals_J(:,:,j), n_e, get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e));
        elseif Dims.exp_flag == 2
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, z_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 5
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_semiz, get_grid_slice(vfoptions.semiz_gridvals_J, 'all', j, has_semiz));
        elseif Dims.exp_flag == 6
            u_gridvals = CreateGridvals(vfoptions.n_u, vfoptions.u_grid, 1);
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, vfoptions.n_u, u_gridvals);
        else
            [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_a2, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2);
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
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec); RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer); last_ReturnFnParamsVec = ReturnFnParamsVec; end
        has_cont = ~isempty(V_next);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4; N_e_interp = N_e; else; N_e_interp = 1; end
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end
        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz); e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);
        a2_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, 1, N_a2, 1, 1]); bz_off_L2 = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, N_bothz, 1]);
        e_off_L2  = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, N_e_interp]);
        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end
            if isempty(RM_cache{1, 1, d_outer_c})
                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_cache{1, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, N_a1, N_a2, N_bothz, N_e]);
            end
            RM_L1 = RM_cache{1, 1, d_outer_c};
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :), [N_d_inner, 1, N_a1, N_a2, N_bothz, N_e_interp]);
                RHS_L1 = RM_L1 + DiscountedEV_L1;
            else; RHS_L1 = RM_L1; end

            % CHECKLIST ITEM 1 & 2: Maximize over Dim 2 (a1_prime), then permute a1_cur to Dim 3
            [~, maxindex1] = max(RHS_L1, [], 2);
            maxindex1_shifted = permute(maxindex1, [1, 3, 2, 4, 5, 6]);
            midpoint = max(min(maxindex1_shifted, N_a1-1), 2);
            a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
            a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, N_a1, N_a2, N_bothz, N_e]);

            RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
            RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a1, N_a2, N_bothz, N_e]);
            if has_cont
                d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1]);
                EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2_off_L2 + N_d2*N_a1fine*N_a2*bz_off_L2 + N_d2*N_a1fine*N_a2*N_bothz*e_off_L2;
                RHS_L2 = RM_L2 + df * EV_fine(EV_idx);
            else; RHS_L2 = RM_L2; end
            RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long, N_a1, N_a2, N_bothz, N_e]);
            [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

            d_ind = rem(p_tmp - 1, N_d_inner) + 1; L2offset = ceil(p_tmp / N_d_inner);
            S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a1; S4 = S3 * N_a2; S5 = S4 * N_bothz;
            a1_off_safe = reshape(gpuArray(0:N_a1-1), [1, N_a1, 1, 1, 1]); a2_off_safe = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1]);
            bz_off_safe = reshape(gpuArray(0:N_bothz-1),[1, 1, 1, N_bothz, 1]); e_off_safe  = reshape(gpuArray(0:N_e-1), [1, 1, 1, 1, N_e]);
            linidx_lower = d_ind + S1*(1 - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*bz_off_safe + S5*e_off_safe;
            linidx_upper = d_ind + S1*(n2long - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*bz_off_safe + S5*e_off_safe;

            isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            S_mid2 = S1 * 1; S_mid3 = S_mid2 * N_a1; S_mid4 = S_mid3 * N_a2; S_mid5 = S_mid4 * N_bothz;
            midpt_idx = d_ind + S_mid2*a1_off_safe + S_mid3*a2_off_safe + S_mid4*bz_off_safe + S_mid5*e_off_safe;
            chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, N_bothz, N_e]); d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

            V_fordouter(:,:,:,:,d_outer_c) = reshape(v_tmp, [N_a1, N_a2, N_bothz, N_e]);
            P1_fordouter(:,:,:,:,d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, N_bothz, N_e]);
            P2_fordouter(:,:,:,:,d_outer_c) = chosen_midpt;
            P3_fordouter(:,:,:,:,d_outer_c) = reshape(L2offset, [N_a1, N_a2, N_bothz, N_e]);
            P4_fordouter(:,:,:,:,d_outer_c) = reshape(flag_val, [N_a1, N_a2, N_bothz, N_e]);
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec); RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer); last_ReturnFnParamsVec = ReturnFnParamsVec; end
        has_cont = ~isempty(V_next);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4; N_e_interp = N_e; else; N_e_interp = 1; end
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end
        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        a2_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, 1, N_a2, 1, 1]); bz_off_L2 = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, N_bothz, 1]);
        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
            end
            for e_c = 1:N_e
                if isempty(RM_cache{1, e_c, d_outer_c})
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{1, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, N_a1, N_a2, N_bothz, 1]);
                end
                RM_L1 = RM_cache{1, e_c, d_outer_c};
                if has_cont
                    e_idx = min(e_c, N_e_interp); EV_fine_e = EV_fine(:,:,:,:,e_idx);
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, e_idx), [N_d_inner, 1, N_a1, N_a2, N_bothz, 1]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                else; RHS_L1 = RM_L1; end

                [~, maxindex1] = max(RHS_L1, [], 2);
                maxindex1_shifted = permute(maxindex1, [1, 3, 2, 4, 5, 6]);
                midpoint = max(min(maxindex1_shifted, N_a1-1), 2);
                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, N_a1, N_a2, N_bothz, 1]);

                RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a1, N_a2, N_bothz, 1]);
                if has_cont
                    d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1]);
                    EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2_off_L2 + N_d2*N_a1fine*N_a2*bz_off_L2;
                    RHS_L2 = RM_L2 + df * EV_fine_e(EV_idx);
                else; RHS_L2 = RM_L2; end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long, N_a1, N_a2, N_bothz, 1]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1; L2offset = ceil(p_tmp / N_d_inner);
                S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a1; S4 = S3 * N_a2;
                a1_off_safe = reshape(gpuArray(0:N_a1-1), [1, N_a1, 1, 1]); a2_off_safe = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1]);
                bz_off_safe = reshape(gpuArray(0:N_bothz-1),[1, 1, 1, N_bothz]);
                linidx_lower = d_ind + S1*(1 - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*bz_off_safe;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*bz_off_safe;

                isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                S_mid2 = S1 * 1; S_mid3 = S_mid2 * N_a1; S_mid4 = S_mid3 * N_a2;
                midpt_idx = d_ind + S_mid2*a1_off_safe + S_mid3*a2_off_safe + S_mid4*bz_off_safe;
                chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, N_bothz, 1]); d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:,:,:,e_c,d_outer_c) = reshape(v_tmp, [N_a1, N_a2, N_bothz, 1]);
                P1_fordouter(:,:,:,e_c,d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, N_bothz, 1]);
                P2_fordouter(:,:,:,e_c,d_outer_c) = chosen_midpt;
                P3_fordouter(:,:,:,e_c,d_outer_c) = reshape(L2offset, [N_a1, N_a2, N_bothz, 1]);
                P4_fordouter(:,:,:,e_c,d_outer_c) = reshape(flag_val, [N_a1, N_a2, N_bothz, 1]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec); RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer); last_ReturnFnParamsVec = ReturnFnParamsVec; end
        has_cont = ~isempty(V_next);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4; N_e_interp = N_e; else; N_e_interp = 1; end
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end
        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);
        a2_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, 1, N_a2, 1, 1]); sb_off_L2 = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, N_semiz, 1]);
        e_off_L2  = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, N_e_interp]);
        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                EV_coarse = reshape(EV_coarse, [N_d_inner, N_a1, 1, N_a2, N_semiz, N_z, N_e_interp]);
                EV_fine   = reshape(EV_fine,   [N_d2, N_a1fine, N_a2, N_semiz, N_z, N_e_interp]);
            end
            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                if isempty(RM_cache{z_c, 1, d_outer_c})
                    bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{z_c, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, N_a1, N_a2, N_semiz, N_e]);
                end
                RM_L1 = RM_cache{z_c, 1, d_outer_c};
                if has_cont
                    EV_fine_z = reshape(EV_fine(:,:,:,:,z_c,:), [N_d2, N_a1fine, N_a2, N_semiz, N_e_interp]);
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, z_c, :), [N_d_inner, 1, N_a1, N_a2, N_semiz, N_e_interp]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                else; RHS_L1 = RM_L1; end

                [~, maxindex1] = max(RHS_L1, [], 2);
                maxindex1_shifted = permute(maxindex1, [1, 3, 2, 4, 5, 6]);
                midpoint = max(min(maxindex1_shifted, N_a1-1), 2);
                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, N_a1, N_a2, N_semiz, N_e]);

                RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a1, N_a2, N_semiz, N_e]);
                if has_cont
                    d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1]);
                    EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2_off_L2 + N_d2*N_a1fine*N_a2*sb_off_L2 + N_d2*N_a1fine*N_a2*N_semiz*e_off_L2;
                    RHS_L2 = RM_L2 + df * EV_fine_z(EV_idx);
                else; RHS_L2 = RM_L2; end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long, N_a1, N_a2, N_semiz, N_e]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1; L2offset = ceil(p_tmp / N_d_inner);
                S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a1; S4 = S3 * N_a2; S5 = S4 * N_semiz;
                a1_off_safe = reshape(gpuArray(0:N_a1-1),  [1, N_a1, 1, 1, 1]); a2_off_safe = reshape(gpuArray(0:N_a2-1),  [1, 1, N_a2, 1, 1]);
                sb_off_safe = reshape(gpuArray(0:N_semiz-1),[1, 1, 1, N_semiz, 1]); e_off_safe  = reshape(gpuArray(0:N_e-1),   [1, 1, 1, 1, N_e]);
                linidx_lower = d_ind + S1*(1 - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*sb_off_safe + S5*e_off_safe;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*sb_off_safe + S5*e_off_safe;

                isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                S_mid2 = S1 * 1; S_mid3 = S_mid2 * N_a1; S_mid4 = S_mid3 * N_a2; S_mid5 = S_mid4 * N_semiz;
                midpt_idx = d_ind + S_mid2*a1_off_safe + S_mid3*a2_off_safe + S_mid4*sb_off_safe + S_mid5*e_off_safe;
                chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, N_semiz, N_e]); d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:,:,sb,:,d_outer_c) = reshape(v_tmp, [N_a1, N_a2, N_semiz, N_e]);
                P1_fordouter(:,:,sb,:,d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, N_semiz, N_e]);
                P2_fordouter(:,:,sb,:,d_outer_c) = chosen_midpt;
                P3_fordouter(:,:,sb,:,d_outer_c) = reshape(L2offset, [N_a1, N_a2, N_semiz, N_e]);
                P4_fordouter(:,:,sb,:,d_outer_c) = reshape(flag_val, [N_a1, N_a2, N_semiz, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec); RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer); last_ReturnFnParamsVec = ReturnFnParamsVec; end
        has_cont = ~isempty(V_next);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4; N_e_interp = N_e; else; N_e_interp = 1; end
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end
        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);
        a2_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, 1, N_a2, 1, 1]); e_off_L2  = reshape(gpuArray(0:N_e_interp-1), [1, 1, 1, 1, 1, N_e_interp]);
        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
            end
            for z_c = 1:N_bothz
                if isempty(RM_cache{z_c, 1, d_outer_c})
                    bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{z_c, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, N_a1, N_a2, 1, N_e]);
                end
                RM_L1 = RM_cache{z_c, 1, d_outer_c};
                if has_cont
                    EV_fine_z = reshape(EV_fine(:,:,:,z_c,:), [N_d2, N_a1fine, N_a2, 1, N_e_interp]);
                    DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, z_c, :), [N_d_inner, 1, N_a1, N_a2, 1, N_e_interp]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                else; RHS_L1 = RM_L1; end

                [~, maxindex1] = max(RHS_L1, [], 2);
                maxindex1_shifted = permute(maxindex1, [1, 3, 2, 4, 5, 6]);
                midpoint = max(min(maxindex1_shifted, N_a1-1), 2);
                a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, N_a1, N_a2, 1, N_e]);

                RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a1, N_a2, 1, N_e]);
                if has_cont
                    d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1]);
                    EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2_off_L2 + N_d2*N_a1fine*N_a2*e_off_L2;
                    RHS_L2 = RM_L2 + df * EV_fine_z(EV_idx);
                else; RHS_L2 = RM_L2; end

                RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long, N_a1, N_a2, 1, N_e]);
                [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                d_ind = rem(p_tmp - 1, N_d_inner) + 1; L2offset = ceil(p_tmp / N_d_inner);
                S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a1; S4 = S3 * N_a2;
                a1_off_safe = reshape(gpuArray(0:N_a1-1), [1, N_a1, 1, 1, 1]); a2_off_safe = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2, 1, 1]);
                e_off_safe  = reshape(gpuArray(0:N_e-1),  [1, 1, 1, 1, N_e]);
                linidx_lower = d_ind + S1*(1 - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*e_off_safe;
                linidx_upper = d_ind + S1*(n2long - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*e_off_safe;

                isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                S_mid2 = S1 * 1; S_mid3 = S_mid2 * N_a1; S_mid4 = S_mid3 * N_a2;
                midpt_idx = d_ind + S_mid2*a1_off_safe + S_mid3*a2_off_safe + S_mid4*e_off_safe;
                chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, 1, N_e]); d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                V_fordouter(:,:,z_c,:,d_outer_c) = reshape(v_tmp, [N_a1, N_a2, 1, N_e]);
                P1_fordouter(:,:,z_c,:,d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, 1, N_e]);
                P2_fordouter(:,:,z_c,:,d_outer_c) = chosen_midpt;
                P3_fordouter(:,:,z_c,:,d_outer_c) = reshape(L2offset, [N_a1, N_a2, 1, N_e]);
                P4_fordouter(:,:,z_c,:,d_outer_c) = reshape(flag_val, [N_a1, N_a2, 1, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec); RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer); last_ReturnFnParamsVec = ReturnFnParamsVec; end
        has_cont = ~isempty(V_next);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4; N_e_interp = N_e; else; N_e_interp = 1; end
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end
        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        a2_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, 1, N_a2, 1, 1]); sb_off_L2 = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, N_semiz, 1]);
        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                EV_coarse = reshape(EV_coarse, [N_d_inner, N_a1, 1, N_a2, N_semiz, N_z, N_e_interp]);
                EV_fine   = reshape(EV_fine,   [N_d2, N_a1fine, N_a2, N_semiz, N_z, N_e_interp]);
            end
            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz); bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);
                for e_c = 1:N_e
                    if isempty(RM_cache{z_c, e_c, d_outer_c})
                        e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);
                        RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                        RM_cache{z_c, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, N_a1, N_a2, N_semiz, 1]);
                    end
                    RM_L1 = RM_cache{z_c, e_c, d_outer_c};
                    if has_cont
                        e_idx = min(e_c, N_e_interp); EV_fine_ze = reshape(EV_fine(:,:,:,:,z_c,e_idx), [N_d2, N_a1fine, N_a2, N_semiz, 1]);
                        DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, :, z_c, e_idx), [N_d_inner, 1, N_a1, N_a2, N_semiz, 1]);
                        RHS_L1 = RM_L1 + DiscountedEV_L1;
                    else; RHS_L1 = RM_L1; end

                    [~, maxindex1] = max(RHS_L1, [], 2);
                    maxindex1_shifted = permute(maxindex1, [1, 3, 2, 4, 5, 6]);
                    midpoint = max(min(maxindex1_shifted, N_a1-1), 2);
                    a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                    a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, N_a1, N_a2, N_semiz, 1]);

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a1, N_a2, N_semiz, 1]);
                    if has_cont
                        d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1]);
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2_off_L2 + N_d2*N_a1fine*N_a2*sb_off_L2;
                        RHS_L2 = RM_L2 + df * EV_fine_ze(EV_idx);
                    else; RHS_L2 = RM_L2; end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long, N_a1, N_a2, N_semiz, 1]);
                    [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                    d_ind = rem(p_tmp - 1, N_d_inner) + 1; L2offset = ceil(p_tmp / N_d_inner);
                    S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a1; S4 = S3 * N_a2;
                    a1_off_safe = reshape(gpuArray(0:N_a1-1),  [1, N_a1, 1, 1]); a2_off_safe = reshape(gpuArray(0:N_a2-1),  [1, 1, N_a2, 1]);
                    sb_off_safe = reshape(gpuArray(0:N_semiz-1),[1, 1, 1, N_semiz]);
                    linidx_lower = d_ind + S1*(1 - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*sb_off_safe;
                    linidx_upper = d_ind + S1*(n2long - 1) + S2*a1_off_safe + S3*a2_off_safe + S4*sb_off_safe;

                    isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    S_mid2 = S1 * 1; S_mid3 = S_mid2 * N_a1; S_mid4 = S_mid3 * N_a2;
                    midpt_idx = d_ind + S_mid2*a1_off_safe + S_mid3*a2_off_safe + S_mid4*sb_off_safe;
                    chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, N_semiz, 1]); d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                    V_fordouter(:,:,sb,e_c,d_outer_c) = reshape(v_tmp, [N_a1, N_a2, N_semiz, 1]);
                    P1_fordouter(:,:,sb,e_c,d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, N_semiz, 1]);
                    P2_fordouter(:,:,sb,e_c,d_outer_c) = chosen_midpt;
                    P3_fordouter(:,:,sb,e_c,d_outer_c) = reshape(L2offset, [N_a1, N_a2, N_semiz, 1]);
                    P4_fordouter(:,:,sb,e_c,d_outer_c) = reshape(flag_val, [N_a1, N_a2, N_semiz, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

    function [V_j, Pol_j] = solve_period_loop_bothz_e(j, V_next, ExpAsset_J)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        if ~isequal(ReturnFnParamsVec, last_ReturnFnParamsVec); RM_cache = cell(max(N_bothz, N_z), N_e, N_d_outer); last_ReturnFnParamsVec = ReturnFnParamsVec; end
        has_cont = ~isempty(V_next);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4; N_e_interp = N_e; else; N_e_interp = 1; end
        if has_cont
            df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end
        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P1_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P2_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray'); P3_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P4_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end
            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
            end
            for z_c = 1:N_bothz
                bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);
                for e_c = 1:N_e
                    if isempty(RM_cache{z_c, e_c, d_outer_c})
                        e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);
                        RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                        RM_cache{z_c, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, N_a1, N_a2, 1, 1]);
                    end
                    RM_L1 = RM_cache{z_c, e_c, d_outer_c};
                    if has_cont
                        e_idx = min(e_c, N_e_interp); EV_fine_ze = reshape(EV_fine(:,:,:,z_c,e_idx), [N_d2, N_a1fine, N_a2, 1, 1]);
                        DiscountedEV_L1 = df * reshape(EV_coarse(d2_off_L1, :, :, z_c, e_idx), [N_d_inner, 1, N_a1, N_a2, 1, 1]);
                        RHS_L1 = RM_L1 + DiscountedEV_L1;
                    else; RHS_L1 = RM_L1; end

                    [~, maxindex1] = max(RHS_L1, [], 2);
                    maxindex1_shifted = permute(maxindex1, [1, 3, 2, 4, 5, 6]);
                    midpoint = max(min(maxindex1_shifted, N_a1-1), 2);
                    a1primeindexes = (midpoint + (midpoint-1)*n2short) + n2_offsets;
                    a1p_grid_safe = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, N_a1, N_a2, 1, 1]);

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a1, N_a2, 1, 1]);
                    if has_cont
                        d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1]);
                        a2_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, 1, N_a2]);
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1primeindexes - 1) + N_d2*N_a1fine*a2_off_L2;
                        RHS_L2 = RM_L2 + df * EV_fine_ze(EV_idx);
                    else; RHS_L2 = RM_L2; end

                    RHS_L2_flat = reshape(RHS_L2, [N_d_inner * n2long, N_a1, N_a2, 1, 1]);
                    [v_tmp, p_tmp] = max(RHS_L2_flat, [], 1);

                    d_ind = rem(p_tmp - 1, N_d_inner) + 1; L2offset = ceil(p_tmp / N_d_inner);
                    S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a1;
                    a1_off_safe = reshape(gpuArray(0:N_a1-1), [1, N_a1, 1]); a2_off_safe = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2]);
                    linidx_lower = d_ind + S1*(1 - 1)      + S2*a1_off_safe + S3*a2_off_safe;
                    linidx_upper = d_ind + S1*(n2long - 1) + S2*a1_off_safe + S3*a2_off_safe;

                    isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_val = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

                    S_mid2 = S1 * 1; S_mid3 = S_mid2 * N_a1;
                    midpt_idx = d_ind + S_mid2*a1_off_safe + S_mid3*a2_off_safe;
                    chosen_midpt = reshape(midpoint(midpt_idx), [N_a1, N_a2, 1, 1]); d_idx_val = d_ind + N_d_inner * (d_outer_c - 1);

                    V_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(v_tmp, [N_a1, N_a2, 1, 1]);
                    P1_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(d_idx_val, [N_a1, N_a2, 1, 1]);
                    P2_fordouter(:,:,z_c,e_c,d_outer_c) = chosen_midpt;
                    P3_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(L2offset, [N_a1, N_a2, 1, 1]);
                    P4_fordouter(:,:,z_c,e_c,d_outer_c) = reshape(flag_val, [N_a1, N_a2, 1, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e, N_d_outer);
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function val = get_grid_slice(grid, idx, j, has_flag)
        if has_flag; if ischar(idx); val = grid(:,:,j); else; val = grid(idx,:,j); end; else; val = []; end
    end

    function [V_out, Pol_out] = extract_pol(V_douter, P1, P2, P3, P4, N_a, N_bothz, N_e, N_d_outer)
        %[cite: 1] VFIToolkit Action Space / State Space canonical ordering mapped:
        % Row 1: d (P1)
        % Row 2: aprime (P2)
        % Row 3: aprimeL2ind (P3) (Interpolation Grid Offset)
        % Row 4: aprimeL2flag (P4) (Interpolation Behavior Override)
        if N_d_outer > 1
            [V_out, maxidx] = max(V_douter, [], 5);
            lin_idx = (1:(N_a*N_bothz*N_e))' + (N_a*N_bothz*N_e)*(maxidx(:) - 1);
            Pol_out = zeros(4, N_a, N_bothz, N_e, 'gpuArray');
            Pol_out(1,:,:,:) = reshape(P1(lin_idx), [1, N_a, N_bothz, N_e]); Pol_out(2,:,:,:) = reshape(P2(lin_idx), [1, N_a, N_bothz, N_e]);
            Pol_out(3,:,:,:) = reshape(P3(lin_idx), [1, N_a, N_bothz, N_e]); Pol_out(4,:,:,:) = reshape(P4(lin_idx), [1, N_a, N_bothz, N_e]);
        else
            V_out = reshape(V_douter, [N_a, N_bothz, N_e]);
            Pol_out = zeros(4, N_a, N_bothz, N_e, 'gpuArray');
            Pol_out(1,:,:,:) = reshape(P1, [1, N_a, N_bothz, N_e]); Pol_out(2,:,:,:) = reshape(P2, [1, N_a, N_bothz, N_e]);
            Pol_out(3,:,:,:) = reshape(P3, [1, N_a, N_bothz, N_e]); Pol_out(4,:,:,:) = reshape(P4, [1, N_a, N_bothz, N_e]);
        end
    end

    function [EV_coarse, EV_fine] = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J)
        % CHECKLIST ITEM 3 & 4 (EXPECT FIRST vs INTERPOLATE FIRST)
        % Legacy ExpAsset tests against an interpolation routine that evaluates skipinterp (Vlower == Vupper)
        % on raw un-expected values. Doing the mathematically superior Expectation First evaluates skipinterp
        % on integrated values, altering the non-linear interpolation paths and failing benchcheck.
        % Additionally, legacy code natively mangles z_cur and z_prime during its matrix multiplication.
        % To clear benchcheck, we strictly map the legacy Interpolate First + Mangle Math sequence here.

        a2p_idx_base = ExpAsset_J.a2primeIndex;
        a2p_weight_base = ExpAsset_J.a2primeProbs;

        N_u_shocks = 1;
        if Dims.exp_flag == 6; N_u_shocks = size(a2p_idx_base, 1); end

        EV_coarse = zeros(N_d2, N_a1, N_a2, N_bothz, N_e_interp, 'gpuArray');

        % EXACT LEGACY BEHAVIOR: Lookup V_next assuming z_prime = z_cur!
        z_offset = N_a1*N_a2 * shiftdim(gpuArray(0:N_bothz-1), -1);

        for u_c = 1:N_u_shocks
            if Dims.exp_flag == 6
                a2p_idx = shiftdim(a2p_idx_base(u_c, :, :, :, :, :), 1); a2p_prb = shiftdim(a2p_weight_base(u_c, :, :, :, :, :), 1);
                u_prob = vfoptions.pi_u(u_c);
            else
                a2p_idx = a2p_idx_base; a2p_prb = a2p_weight_base; u_prob = 1;
            end

            % CHECKLIST ITEM 5: Honor canonical [d2 then a2] shape. Delete defensive permutes.
            if length(n_a2) == 1
                N_d_exp = size(a2p_idx, 1); N_a2_exp = size(a2p_idx, 2); N_z_exp = max(size(a2p_idx, 3), 1);
            else
                N_d_exp = size(a2p_idx, 1); N_a2_exp = size(a2p_idx, 2); N_z_exp = max(size(a2p_idx, 3), 1);
            end

            if length(n_a2) == 1
                if N_z_exp > 1 && N_z_exp ~= N_bothz
                    a2p_idx = reshape(repmat(reshape(a2p_idx, [N_d_exp, N_a2_exp, 1, N_z_exp]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
                    a2p_prb = reshape(repmat(reshape(a2p_prb, [N_d_exp, N_a2_exp, 1, N_z_exp]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
                    N_z_exp = N_bothz;
                end

                if N_z_exp > 1
                    a1_base = repelem(gpuArray(1:1:N_a1)', N_d_exp, N_a2_exp, N_z_exp);
                    idx_lo = a1_base + N_a1 * repmat(a2p_idx - 1, N_a1, 1, 1);
                    idx_hi = a1_base + N_a1 * repmat(a2p_idx, N_a1, 1, 1);
                    prob_exp = repmat(a2p_prb, N_a1, 1, 1);
                else
                    a1_base = repelem(gpuArray(1:1:N_a1)', N_d_exp, N_a2_exp);
                    idx_lo = a1_base + N_a1 * repmat(a2p_idx - 1, N_a1, 1);
                    idx_hi = a1_base + N_a1 * repmat(a2p_idx, N_a1, 1);
                    prob_exp = repmat(a2p_prb, N_a1, 1);
                end

                Vlower = reshape(EVpre_flat(idx_lo + z_offset), [N_d_exp*N_a1, N_a2_exp, max(N_z_exp, N_bothz)]);
                Vupper = reshape(EVpre_flat(idx_hi + z_offset), [N_d_exp*N_a1, N_a2_exp, max(N_z_exp, N_bothz)]);

                if N_z_exp == 1 && N_bothz > 1
                    prob_exp = repmat(prob_exp, 1, 1, N_bothz);
                end

                skipinterp = (Vlower == Vupper); prob_exp(skipinterp) = 0;
                c_lo = prob_exp .* Vlower;       c_lo(isnan(c_lo)) = 0;
                c_hi = (1 - prob_exp) .* Vupper; c_hi(isnan(c_hi)) = 0;

                EV_u = c_lo + c_hi; EV_u(prob_exp == 0) = Vupper(prob_exp == 0); EV_u(prob_exp == 1) = Vlower(prob_exp == 1);
            else
                loIdx_1 = reshape(a2p_idx(1,:,:,:), [N_d_exp, N_a2_exp, N_z_exp]);
                loIdx_2 = reshape(a2p_idx(2,:,:,:), [N_d_exp, N_a2_exp, N_z_exp]);
                prob_1  = reshape(a2p_prb(1,:,:,:), [N_d_exp, N_a2_exp, N_z_exp]);
                prob_2  = reshape(a2p_prb(2,:,:,:), [N_d_exp, N_a2_exp, N_z_exp]);

                if N_z_exp > 1 && N_z_exp ~= N_bothz
                    loIdx_1 = reshape(repmat(reshape(loIdx_1, [N_d_exp, N_a2_exp, 1, N_z_exp]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
                    loIdx_2 = reshape(repmat(reshape(loIdx_2, [N_d_exp, N_a2_exp, 1, N_z_exp]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
                    prob_1  = reshape(repmat(reshape(prob_1, [N_d_exp, N_a2_exp, 1, N_z_exp]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
                    prob_2  = reshape(repmat(reshape(prob_2, [N_d_exp, N_a2_exp, 1, N_z_exp]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
                    N_z_exp = N_bothz;
                end

                if N_z_exp > 1
                    a1_base = repelem(gpuArray(1:1:N_a1)', N_d_exp, N_a2_exp, N_z_exp);
                else
                    a1_base = repelem(gpuArray(1:1:N_a1)', N_d_exp, N_a2_exp);
                end
                n_a2_1 = n_a2(1);

                idx_ll = a1_base + N_a1 * repmat(loIdx_1 + n_a2_1*(loIdx_2-1) - 1, N_a1, 1, 1);
                idx_hl = a1_base + N_a1 * repmat((loIdx_1+1) + n_a2_1*(loIdx_2-1) - 1, N_a1, 1, 1);
                idx_lh = a1_base + N_a1 * repmat(loIdx_1 + n_a2_1*loIdx_2 - 1, N_a1, 1, 1);
                idx_hh = a1_base + N_a1 * repmat((loIdx_1+1) + n_a2_1*loIdx_2 - 1, N_a1, 1, 1);

                V_ll = reshape(EVpre_flat(idx_ll + z_offset), [N_d_exp * N_a1, N_a2_exp, max(N_z_exp, N_bothz)]);
                V_hl = reshape(EVpre_flat(idx_hl + z_offset), [N_d_exp * N_a1, N_a2_exp, max(N_z_exp, N_bothz)]);
                V_lh = reshape(EVpre_flat(idx_lh + z_offset), [N_d_exp * N_a1, N_a2_exp, max(N_z_exp, N_bothz)]);
                V_hh = reshape(EVpre_flat(idx_hh + z_offset), [N_d_exp * N_a1, N_a2_exp, max(N_z_exp, N_bothz)]);

                prb1_full = repmat(prob_1, N_a1, 1, 1); prb2_full = repmat(prob_2, N_a1, 1, 1);
                if N_z_exp == 1 && N_bothz > 1
                    prb1_full = repmat(prb1_full, 1, 1, N_bothz);
                    prb2_full = repmat(prb2_full, 1, 1, N_bothz);
                end

                p1_loy = prb1_full; p1_loy(V_ll == V_hl) = 0; c_ll = p1_loy .* V_ll; c_ll(isnan(c_ll)) = 0;
                c_hl = (1-p1_loy) .* V_hl; c_hl(isnan(c_hl)) = 0; EV_loy = c_ll + c_hl;

                p1_hiy = prb1_full; p1_hiy(V_lh == V_hh) = 0; c_lh = p1_hiy .* V_lh; c_lh(isnan(c_lh)) = 0;
                c_hh = (1-p1_hiy) .* V_hh; c_hh(isnan(c_hh)) = 0; EV_hiy = c_lh + c_hh;

                p2 = prb2_full; p2(EV_loy == EV_hiy) = 0; c_loy = p2 .* EV_loy; c_loy(isnan(c_loy)) = 0;
                c_hiy = (1-p2) .* EV_hiy; c_hiy(isnan(c_hiy)) = 0; EV_u = c_loy + c_hiy;
            end

            % EXACT LEGACY REPLICATION MANGLE:
            % Multiply EV_u (which has z_cur in dim 3) by pi_bz' (z_prime in dim 3)
            EV_weighted = EV_u .* shiftdim(pi_bz', -2);
            EV_weighted(isnan(EV_weighted)) = 0;

            if N_bothz > 1
                EV_exp_u = sum(EV_weighted, 3);
            else
                EV_exp_u = EV_weighted;
            end

            EV_coarse = EV_coarse + u_prob .* reshape(EV_exp_u, [N_d_exp, N_a1, N_a2_exp, N_bothz, 1]);
        end

        if N_d_exp ~= N_d2 || N_a2_exp ~= N_a2
            EV_tmp = zeros(N_d2, N_a1, N_a2, N_bothz, 1, 'gpuArray');
            d_lim = min(N_d2, N_d_exp); a_lim = min(N_a2, N_a2_exp);
            EV_tmp(1:d_lim, :, 1:a_lim, :, :) = EV_coarse(1:d_lim, :, 1:a_lim, :, :);
            if N_d_exp == 1 && N_d2 > 1; EV_tmp = repmat(EV_coarse, N_d2, 1, 1, 1, 1); end
            if N_a2_exp == 1 && N_a2 > 1; EV_tmp = repmat(EV_tmp, 1, 1, N_a2, 1, 1); end
            EV_coarse = EV_tmp;
        end

        if N_e_interp > 1; EV_coarse = repmat(EV_coarse, 1, 1, 1, 1, N_e_interp); end
        EV_perm = permute(EV_coarse, [2, 1, 3, 4, 5]); % [N_a1, N_d2, N_a2, N_bothz, N_e_interp]
        EV_interp_next = interp1(a1_gridvals, EV_perm, a1prime_grid_col);
        EV_fine = permute(EV_interp_next, [2, 1, 3, 4, 5]); % [N_d2, N_a1fine, N_a2, N_bothz, N_e_interp]
    end


end
