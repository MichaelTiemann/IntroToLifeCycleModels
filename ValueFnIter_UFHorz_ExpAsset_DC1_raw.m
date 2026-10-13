function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_z = Dims.N_z; N_semiz = Dims.N_semiz;

has_semiz = Dims.N_semiz_raw > 0;
has_e     = Dims.N_e_raw > 0;
has_z     = Dims.special_n_z(1) > 0;
has_bothz = has_semiz || has_z;

n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_e = Dims.n_e;
n_semiz = Dims.n_semiz;
n_d1 = Dims.n_d1; n_d2 = Dims.n_d2;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2;
N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;

a1_gridvals = gpuArray(a_grid(1:N_a1));
a2_gridvals = gpuArray(a_grid(N_a1+1:end));

level1n = Dims.level1n;
level1ii = round(linspace(1, N_a1, level1n));

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
V = zeros(N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:N_a1);

d2_off_L1 = gpuArray(repelem((1:N_d2)', N_d1));
d2_off_L2 = reshape(gpuArray(repelem(0:N_d2-1, N_d1)), [N_d_inner, 1, 1, 1, 1, 1]);
a2_off_L2 = reshape(gpuArray(0:N_a2-1), [1, 1, 1, N_a2, 1, 1]);
bz_off_L2 = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, N_bothz, 1]);
sb_off_L2 = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, N_semiz, 1]);
N_e_interp_max = N_e; if Dims.exp_flag ~= 3 && Dims.exp_flag ~= 4; N_e_interp_max = 1; end
e_off_L2  = reshape(gpuArray(0:N_e_interp_max-1), [1, 1, 1, 1, 1, N_e_interp_max]);

% --- 5. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a1, N_a2, N_bothz, N_e]); else; V_next = []; end
    else
        V_next = V(:, :, :, :, j+1);
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
        case 'Full'; [V(:,:,:,:,j), Pol_j] = solve_period_full(j, V_next, ExpAsset_J);
        case 'Loop_e'; [V(:,:,:,:,j), Pol_j] = solve_period_loop_e(j, V_next, ExpAsset_J);
        case 'Loop_z_parallel_semiz'; [V(:,:,:,:,j), Pol_j] = solve_period_loop_z_parallel_semiz(j, V_next, ExpAsset_J);
        case 'Loop_bothz'; [V(:,:,:,:,j), Pol_j] = solve_period_loop_bothz(j, V_next, ExpAsset_J);
        case 'Loop_z_parallel_semiz_e'; [V(:,:,:,:,j), Pol_j] = solve_period_loop_z_parallel_semiz_e(j, V_next, ExpAsset_J);
        case 'Loop_bothz_e'; [V(:,:,:,:,j), Pol_j] = solve_period_loop_bothz_e(j, V_next, ExpAsset_J);
    end
    Policy(:,:,:,:,:,j) = Pol_j;
end

% =====================================================================
% SPECIALIZED SOLVE FUNCTIONS (2-Stage DC midpoint)
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
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, 1, N_e]), 4); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);
        e_val  = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if isempty(RM_cache{1, 1, d_outer_c})
                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_cache{1, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, level1n, N_a2, N_bothz, N_e]);
            end
            RM_L1 = RM_cache{1, 1, d_outer_c};

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                DiscountedEV_L1_full = df * reshape(EV_coarse(d2_off_L1, :, :, :), [N_d_inner, N_a1, 1, N_a2, N_bothz, N_e_interp]);
                DiscountedEV_L1 = reshape(DiscountedEV_L1_full(:, level1ii, 1, :, :, :), [N_d_inner, 1, level1n, N_a2, N_bothz, N_e_interp]);
                RHS_L1 = RM_L1 + DiscountedEV_L1;
                EV_coarse_flat = reshape(EV_coarse, [N_d2 * N_a1 * N_a2 * N_bothz * N_e_interp, 1]);
            else
                RHS_L1 = RM_L1;
            end

            RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1, level1n, N_a2, N_bothz, N_e]);
            [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

            a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
            a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a1) + 1;

            d_inner_idx = rem(maxindex1 - 1, N_d_inner) + 1;
            linear_p = d_inner_idx + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_idx_L1 - 1);

            V_fordouter(level1ii, :, :, :, d_outer_c) = reshape(v_tmp, [level1n, N_a2, N_bothz, N_e]);
            P_fordouter(level1ii, :, :, :, d_outer_c) = reshape(linear_p, [level1n, N_a2, N_bothz, N_e]);

            diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, :) - a1p_idx_L1(1, 1:end-1, :, :, :));
            maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 4, 5]), [level1n - 1, 1]));
            if isempty(maxgap_vec); maxgap_vec = 0; end

            for ii = 1:(level1n-1)
                curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                L = length(curraindex);
                if L == 0; continue; end

                mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, :), N_a1 - mg), [1, 1, 1, N_a2, N_bothz, N_e]);
                a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                le_flat = repmat(loweredge, [1, 1, L, 1, 1, 1]);
                a1p_idx = le_flat + a1p_offsets;
                a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                RM_L2_DC = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals(curraindex), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_a2, N_bothz, N_e]);

                if has_cont
                    EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*bz_off_L2 + N_d2*N_a1*N_a2*N_bothz*e_off_L2;
                    RHS_L2_DC = RM_L2_DC + df * EV_coarse_flat(EV_idx);
                else
                    RHS_L2_DC = RM_L2_DC;
                end

                RHS_L2_DC_flat = reshape(RHS_L2_DC, [N_d_inner * (mg+1), L, N_a2, N_bothz, N_e]);
                [v_tmp_L2, maxindex2] = max(RHS_L2_DC_flat, [], 1);

                d_inner_idx_L2 = rem(maxindex2 - 1, N_d_inner) + 1;
                a1p_offset = ceil(maxindex2 / N_d_inner);
                a1p_abs = reshape(loweredge, [1, 1, N_a2, N_bothz, N_e]) + reshape(a1p_offset, [1, L, N_a2, N_bothz, N_e]) - 1;
                linear_p_L2 = d_inner_idx_L2 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);

                V_fordouter(curraindex, :, :, :, d_outer_c) = reshape(v_tmp_L2, [L, N_a2, N_bothz, N_e]);
                P_fordouter(curraindex, :, :, :, d_outer_c) = reshape(linear_p_L2, [L, N_a2, N_bothz, N_e]);
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P_fordouter, N_d_outer);
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
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, 1, N_e]), 4); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        bz_val = get_grid_slice(Grids.bothz_gridvals_J, 'all', j, has_bothz);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                EV_coarse = reshape(EV_coarse, [N_d2, N_a1, 1, N_a2, N_bothz, N_e_interp]);
                DiscountedEV_L1_full = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :), [N_d_inner, N_a1, 1, N_a2, N_bothz, N_e_interp]);
                EV_coarse_flat_e = reshape(EV_coarse, [N_d2 * N_a1 * N_a2 * N_bothz * N_e_interp, 1]);
            end

            for e_c = 1:N_e
                if isempty(RM_cache{1, e_c, d_outer_c})
                    e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{1, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, level1n, N_a2, N_bothz, 1]);
                end
                RM_L1 = RM_cache{1, e_c, d_outer_c};

                if has_cont
                    e_idx = min(e_c, N_e_interp);
                    DiscountedEV_L1 = reshape(DiscountedEV_L1_full(:, :, 1, :, :, e_idx), [N_d_inner, N_a1, 1, N_a2, N_bothz, 1]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                else
                    RHS_L1 = RM_L1;
                end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1, level1n, N_a2, N_bothz, 1]);
                [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a1) + 1;

                d_inner_idx = rem(maxindex1 - 1, N_d_inner) + 1;
                linear_p = d_inner_idx + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_idx_L1 - 1);

                V_fordouter(level1ii, :, :, e_c, d_outer_c) = reshape(v_tmp, [level1n, N_a2, N_bothz, 1]);
                P_fordouter(level1ii, :, :, e_c, d_outer_c) = reshape(linear_p, [level1n, N_a2, N_bothz, 1]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, 1) - a1p_idx_L1(1, 1:end-1, :, :, 1));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 4]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, 1), N_a1 - mg), [1, 1, 1, N_a2, N_bothz, 1]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                    le_flat = repmat(loweredge, [1, 1, L, 1, 1, 1]);
                    a1p_idx = le_flat + a1p_offsets;
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2_DC = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals(curraindex), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_a2, N_bothz, 1]);

                    if has_cont
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*bz_off_L2;
                        RHS_L2_DC = RM_L2_DC + df * EV_coarse_flat_e(EV_idx);
                    else
                        RHS_L2_DC = RM_L2_DC;
                    end

                    RHS_L2_DC_flat = reshape(RHS_L2_DC, [N_d_inner * (mg+1), L, N_a2, N_bothz, 1]);
                    [v_tmp_L2, maxindex2] = max(RHS_L2_DC_flat, [], 1);

                    d_inner_idx_L2 = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    a1p_abs = reshape(loweredge, [1, 1, N_a2, N_bothz, 1]) + reshape(a1p_offset, [1, L, N_a2, N_bothz, 1]) - 1;
                    linear_p_L2 = d_inner_idx_L2 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);

                    V_fordouter(curraindex, :, :, e_c, d_outer_c) = reshape(v_tmp_L2, [L, N_a2, N_bothz, 1]);
                    P_fordouter(curraindex, :, :, e_c, d_outer_c) = reshape(linear_p_L2, [L, N_a2, N_bothz, 1]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P_fordouter, N_d_outer);
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
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, 1, N_e]), 4); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                EV_coarse = reshape(EV_coarse, [N_d2, N_a1, 1, N_a2, N_semiz, N_z, N_e_interp]);
                DiscountedEV_L1_full = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :, :), [N_d_inner, N_a1, 1, N_a2, N_semiz, N_z, N_e_interp]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);
                if isempty(RM_cache{z_c, 1, d_outer_c})
                    bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{z_c, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, level1n, N_a2, N_semiz, N_e]);
                end
                RM_L1 = RM_cache{z_c, 1, d_outer_c};

                if has_cont
                    DiscountedEV_L1 = reshape(DiscountedEV_L1_full(:, :, 1, :, :, z_c, :), [N_d_inner, N_a1, 1, N_a2, N_semiz, N_e_interp]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                    EV_coarse_flat_z = reshape(EV_coarse(:,:,:,:,:,z_c,:), [N_d2 * N_a1 * N_a2 * N_semiz * N_e_interp, 1]);
                else
                    RHS_L1 = RM_L1;
                end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1, level1n, N_a2, N_semiz, N_e]);
                [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a1) + 1;

                d_inner_idx = rem(maxindex1 - 1, N_d_inner) + 1;
                linear_p = d_inner_idx + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_idx_L1 - 1);

                V_fordouter(level1ii, :, sb, :, d_outer_c) = reshape(v_tmp, [level1n, N_a2, N_semiz, N_e]);
                P_fordouter(level1ii, :, sb, :, d_outer_c) = reshape(linear_p, [level1n, N_a2, N_semiz, N_e]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, :) - a1p_idx_L1(1, 1:end-1, :, :, :));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 4, 5]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, :), N_a1 - mg), [1, 1, 1, N_a2, N_semiz, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                    le_flat = repmat(loweredge, [1, 1, L, 1, 1, 1]);
                    a1p_idx = le_flat + a1p_offsets;
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2_DC = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals(curraindex), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_a2, N_semiz, N_e]);

                    if has_cont
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*sb_off_L2 + N_d2*N_a1*N_a2*N_semiz*e_off_L2;
                        RHS_L2_DC = RM_L2_DC + df * EV_coarse_flat_z(EV_idx);
                    else
                        RHS_L2_DC = RM_L2_DC;
                    end

                    RHS_L2_DC_flat = reshape(RHS_L2_DC, [N_d_inner * (mg+1), L, N_a2, N_semiz, N_e]);
                    [v_tmp_L2, maxindex2] = max(RHS_L2_DC_flat, [], 1);

                    d_inner_idx_L2 = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    a1p_abs = reshape(loweredge, [1, 1, N_a2, N_semiz, N_e]) + reshape(a1p_offset, [1, L, N_a2, N_semiz, N_e]) - 1;
                    linear_p_L2 = d_inner_idx_L2 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);

                    V_fordouter(curraindex, :, sb, :, d_outer_c) = reshape(v_tmp_L2, [L, N_a2, N_semiz, N_e]);
                    P_fordouter(curraindex, :, sb, :, d_outer_c) = reshape(linear_p_L2, [L, N_a2, N_semiz, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P_fordouter, N_d_outer);
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
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, 1, N_e]), 4); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        e_val = get_grid_slice(Grids.e_gridvals_J, 'all', j, has_e);

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                EV_coarse = reshape(EV_coarse, [N_d2, N_a1, 1, N_a2, N_bothz, N_e_interp]);
                DiscountedEV_L1_full = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :), [N_d_inner, N_a1, 1, N_a2, N_bothz, N_e_interp]);
            end

            for z_c = 1:N_bothz
                if isempty(RM_cache{z_c, 1, d_outer_c})
                    bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);
                    RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                    RM_cache{z_c, 1, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, level1n, N_a2, 1, N_e]);
                end
                RM_L1 = RM_cache{z_c, 1, d_outer_c};

                if has_cont
                    DiscountedEV_L1 = reshape(DiscountedEV_L1_full(:, :, 1, :, z_c, :), [N_d_inner, N_a1, 1, N_a2, 1, N_e_interp]);
                    RHS_L1 = RM_L1 + DiscountedEV_L1;
                    EV_coarse_flat_bz = reshape(EV_coarse(:,:,:,:,z_c,:), [N_d2 * N_a1 * N_a2 * N_e_interp, 1]);
                else
                    RHS_L1 = RM_L1;
                end

                RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1, level1n, N_a2, 1, N_e]);
                [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a1) + 1;

                d_inner_idx = rem(maxindex1 - 1, N_d_inner) + 1;
                linear_p = d_inner_idx + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_idx_L1 - 1);

                V_fordouter(level1ii, :, z_c, :, d_outer_c) = reshape(v_tmp, [level1n, N_a2, 1, N_e]);
                P_fordouter(level1ii, :, z_c, :, d_outer_c) = reshape(linear_p, [level1n, N_a2, 1, N_e]);

                diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, 1, :) - a1p_idx_L1(1, 1:end-1, :, 1, :));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 5]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(level1n-1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                    loweredge = reshape(min(a1p_idx_L1(1, ii, :, 1, :), N_a1 - mg), [1, 1, 1, N_a2, 1, N_e]);
                    a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                    le_flat = repmat(loweredge, [1, 1, L, 1, 1, 1]);
                    a1p_idx = le_flat + a1p_offsets;
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                    RM_L2_DC = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals(curraindex), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_a2, 1, N_e]);

                    if has_cont
                        EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*e_off_L2;
                        RHS_L2_DC = RM_L2_DC + df * EV_coarse_flat_bz(EV_idx);
                    else
                        RHS_L2_DC = RM_L2_DC;
                    end

                    RHS_L2_DC_flat = reshape(RHS_L2_DC, [N_d_inner * (mg+1), L, N_a2, 1, N_e]);
                    [v_tmp_L2, maxindex2] = max(RHS_L2_DC_flat, [], 1);

                    d_inner_idx_L2 = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_offset = ceil(maxindex2 / N_d_inner);
                    a1p_abs = reshape(loweredge, [1, 1, N_a2, 1, N_e]) + reshape(a1p_offset, [1, L, N_a2, 1, N_e]) - 1;
                    linear_p_L2 = d_inner_idx_L2 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);

                    V_fordouter(curraindex, :, z_c, :, d_outer_c) = reshape(v_tmp_L2, [L, N_a2, 1, N_e]);
                    P_fordouter(curraindex, :, z_c, :, d_outer_c) = reshape(linear_p_L2, [L, N_a2, 1, N_e]);
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P_fordouter, N_d_outer);
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
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, 1, N_e]), 4); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                EV_coarse = reshape(EV_coarse, [N_d2, N_a1, 1, N_a2, N_semiz, N_z, N_e_interp]);
                DiscountedEV_L1_full = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :, :), [N_d_inner, N_a1, 1, N_a2, N_semiz, N_z, N_e_interp]);
            end

            for z_c = 1:N_z
                sb = (z_c-1)*N_semiz + (1:N_semiz);

                for e_c = 1:N_e
                    if isempty(RM_cache{z_c, e_c, d_outer_c})
                        bz_val = get_grid_slice(Grids.bothz_gridvals_J, sb, j, has_bothz);
                        e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);
                        RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                        RM_cache{z_c, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, level1n, N_a2, N_semiz, 1]);
                    end
                    RM_L1 = RM_cache{z_c, e_c, d_outer_c};

                    if has_cont
                        e_idx = min(e_c, N_e_interp);
                        DiscountedEV_L1 = reshape(DiscountedEV_L1_full(:, :, 1, :, :, z_c, e_idx), [N_d_inner, N_a1, 1, N_a2, N_semiz, 1]);
                        RHS_L1 = RM_L1 + DiscountedEV_L1;
                        EV_coarse_flat_ze = reshape(EV_coarse(:,:,:,:,:,z_c,e_idx), [N_d2 * N_a1 * N_a2 * N_semiz, 1]);
                    else
                        RHS_L1 = RM_L1;
                    end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1, level1n, N_a2, N_semiz, 1]);
                    [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a1) + 1;

                    d_inner_idx = rem(maxindex1 - 1, N_d_inner) + 1;
                    linear_p = d_inner_idx + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_idx_L1 - 1);

                    V_fordouter(level1ii, :, sb, e_c, d_outer_c) = reshape(v_tmp, [level1n, N_a2, N_semiz, 1]);
                    P_fordouter(level1ii, :, sb, e_c, d_outer_c) = reshape(linear_p, [level1n, N_a2, N_semiz, 1]);

                    diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, :, 1) - a1p_idx_L1(1, 1:end-1, :, :, 1));
                    maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 4]), [level1n - 1, 1]));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                        loweredge = reshape(min(a1p_idx_L1(1, ii, :, :, 1), N_a1 - mg), [1, 1, 1, N_a2, N_semiz, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                        le_flat = repmat(loweredge, [1, 1, L, 1, 1, 1]);
                        a1p_idx = le_flat + a1p_offsets;
                        a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                        RM_L2_DC = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, [Dims.special_n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals(curraindex), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_a2, N_semiz, 1]);

                        if has_cont
                            EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2 + N_d2*N_a1*N_a2*sb_off_L2;
                            RHS_L2_DC = RM_L2_DC + df * EV_coarse_flat_ze(EV_idx);
                        else
                            RHS_L2_DC = RM_L2_DC;
                        end

                        RHS_L2_DC_flat = reshape(RHS_L2_DC, [N_d_inner * (mg+1), L, N_a2, N_semiz, 1]);
                        [v_tmp_L2, maxindex2] = max(RHS_L2_DC_flat, [], 1);

                        d_inner_idx_L2 = rem(maxindex2 - 1, N_d_inner) + 1;
                        a1p_offset = ceil(maxindex2 / N_d_inner);
                        a1p_abs = reshape(loweredge, [1, 1, N_a2, N_semiz, 1]) + reshape(a1p_offset, [1, L, N_a2, N_semiz, 1]) - 1;
                        linear_p_L2 = d_inner_idx_L2 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);

                        V_fordouter(curraindex, :, sb, e_c, d_outer_c) = reshape(v_tmp_L2, [L, N_a2, N_semiz, 1]);
                        P_fordouter(curraindex, :, sb, e_c, d_outer_c) = reshape(linear_p_L2, [L, N_a2, N_semiz, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P_fordouter, N_d_outer);
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
            if has_e; EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, 1, N_e]), 4); else; EVpre = V_next; end
            EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);
        end

        V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
        P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');

        for d_outer_c = 1:N_d_outer
            if N_d_outer > 1; d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]); else; d_val = Grids.d12_gridvals; end

            if has_cont
                if has_semiz; pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j)); else; pi_bz = Probs.pi_z_J(:,:,j); end
                EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J);
                EV_coarse = reshape(EV_coarse, [N_d2, N_a1, 1, N_a2, N_bothz, N_e_interp]);
                DiscountedEV_L1_full = df * reshape(EV_coarse(d2_off_L1, :, :, :, :, :), [N_d_inner, N_a1, 1, N_a2, N_bothz, N_e_interp]);
            end

            for z_c = 1:N_bothz
                for e_c = 1:N_e
                    if isempty(RM_cache{z_c, e_c, d_outer_c})
                        bz_val = get_grid_slice(Grids.bothz_gridvals_J, z_c, j, has_bothz);
                        e_val = get_grid_slice(Grids.e_gridvals_J, e_c, j, has_e);
                        RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                        RM_cache{z_c, e_c, d_outer_c} = reshape(RM_L1, [N_d_inner, N_a1, level1n, N_a2, 1, 1]);
                    end
                    RM_L1 = RM_cache{z_c, e_c, d_outer_c};

                    if has_cont
                        e_idx = min(e_c, N_e_interp);
                        DiscountedEV_L1 = reshape(DiscountedEV_L1_full(:, :, 1, :, z_c, e_idx), [N_d_inner, N_a1, 1, N_a2, 1, 1]);
                        RHS_L1 = RM_L1 + DiscountedEV_L1;
                        EV_coarse_flat_ze = reshape(EV_coarse(:,:,:,:,z_c,e_idx), [N_d2 * N_a1 * N_a2, 1]);
                    else
                        RHS_L1 = RM_L1;
                    end

                    RHS_L1_flat = reshape(RHS_L1, [N_d_inner * N_a1, level1n, N_a2, 1, 1]);
                    [v_tmp, maxindex1] = max(RHS_L1_flat, [], 1);

                    a1p_a2p_ind = ceil(maxindex1 / N_d_inner);
                    a1p_idx_L1 = rem(a1p_a2p_ind - 1, N_a1) + 1;

                    d_inner_idx = rem(maxindex1 - 1, N_d_inner) + 1;
                    linear_p = d_inner_idx + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_idx_L1 - 1);

                    V_fordouter(level1ii, :, z_c, e_c, d_outer_c) = reshape(v_tmp, [level1n, N_a2, 1, 1]);
                    P_fordouter(level1ii, :, z_c, e_c, d_outer_c) = reshape(linear_p, [level1n, N_a2, 1, 1]);

                    diff_a1p = max(0, a1p_idx_L1(1, 2:end, :, 1, 1) - a1p_idx_L1(1, 1:end-1, :, 1, 1));
                    maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 4]), [level1n - 1, 1]));
                    if isempty(maxgap_vec); maxgap_vec = 0; end

                    for ii = 1:(level1n-1)
                        curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                        L = length(curraindex);
                        if L == 0; continue; end

                        mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                        loweredge = reshape(min(a1p_idx_L1(1, ii, :, 1, 1), N_a1 - mg), [1, 1, 1, N_a2, 1, 1]);
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                        le_flat = repmat(loweredge, [1, 1, L, 1, 1, 1]);
                        a1p_idx = le_flat + a1p_offsets;
                        a1p_grid_safe = reshape(a1_gridvals(a1p_idx), size(a1p_idx));

                        RM_L2_DC = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, Dims.special_n_bothz, Dims.special_n_e, d_val, a1p_grid_safe, a1_gridvals(curraindex), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_a2, 1, 1]);

                        if has_cont
                            EV_idx = (d2_off_L2 + 1) + N_d2*(a1p_idx - 1) + N_d2*N_a1*a2_off_L2;
                            RHS_L2_DC = RM_L2_DC + df * EV_coarse_flat_ze(EV_idx);
                        else
                            RHS_L2_DC = RM_L2_DC;
                        end

                        RHS_L2_DC_flat = reshape(RHS_L2_DC, [N_d_inner * (mg+1), L, N_a2, 1, 1]);
                        [v_tmp_L2, maxindex2] = max(RHS_L2_DC_flat, [], 1);

                        d_inner_idx_L2 = rem(maxindex2 - 1, N_d_inner) + 1;
                        a1p_offset = ceil(maxindex2 / N_d_inner);
                        a1p_abs = reshape(loweredge, [1, 1, N_a2, 1, 1]) + reshape(a1p_offset, [1, L, N_a2, 1, 1]) - 1;
                        linear_p_L2 = d_inner_idx_L2 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);

                        V_fordouter(curraindex, :, z_c, e_c, d_outer_c) = reshape(v_tmp_L2, [L, N_a2, 1, 1]);
                        P_fordouter(curraindex, :, z_c, e_c, d_outer_c) = reshape(linear_p_L2, [L, N_a2, 1, 1]);
                    end
                end
            end
        end
        [V_j, Pol_j] = extract_pol(V_fordouter, P_fordouter, N_d_outer);
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

    function [V_out, Pol_out] = extract_pol(V_douter, P_douter, N_d_outer)
        V_douter = reshape(V_douter, [N_a1, N_a2, N_bothz, N_e, N_d_outer]);
        if N_d_outer > 1
            [V_out, maxidx] = max(V_douter, [], 5);
            lin_idx = (1:(N_a1*N_a2*N_bothz*N_e))' + (N_a1*N_a2*N_bothz*N_e)*(maxidx(:) - 1);
            Pol_out = reshape(P_douter(lin_idx), [1, N_a1, N_a2, N_bothz, N_e]);
        else
            V_out = reshape(V_douter, [N_a1, N_a2, N_bothz, N_e]);
            Pol_out = reshape(P_douter, [1, N_a1, N_a2, N_bothz, N_e]);
        end
    end

    function EV_coarse = precalc_ev(EVpre_flat, pi_bz, N_e_interp, ExpAsset_J)
        EV_expected_base = compute_expected_value(EVpre_flat, pi_bz, N_a1 * N_a2, N_bothz);
        EV_expected_base = reshape(EV_expected_base, [N_a1, N_a2, N_bothz]);

        if N_e_interp > 1
            EV_expected_base = repmat(EV_expected_base, [1, 1, 1, N_e_interp]);
        end
        num_z_interp = N_bothz * N_e_interp;
        EV_expected_base = reshape(EV_expected_base, [N_a1, N_a2, num_z_interp]);

        a1_off_coarse = reshape(gpuArray(0:N_a1-1), [1, N_a1, 1, 1]);
        z_off = reshape(gpuArray(0:num_z_interp-1), [1, 1, 1, num_z_interp]);

        if Dims.exp_flag == 6
            N_u = prod(vfoptions.n_u);
            EV_coarse_sum = zeros(N_d2, N_a1, N_a2, num_z_interp, 'gpuArray');
            for u_c = 1:N_u
                exp_idx = ExpAsset_J.a2primeIndex(:,:,u_c);
                exp_prb = ExpAsset_J.a2primeProbs(:,:,u_c);
                if size(exp_idx, 3) < num_z_interp
                    exp_idx = repmat(exp_idx, [1, 1, num_z_interp / size(exp_idx, 3)]);
                    exp_prb = repmat(exp_prb, [1, 1, num_z_interp / size(exp_prb, 3)]);
                end

                a2p_idx = reshape(exp_idx, [N_d2, 1, N_a2, num_z_interp]);
                prob_flat = reshape(exp_prb, [N_d2, 1, N_a2, num_z_interp]);

                EV_idx_coarse = 1 + a1_off_coarse + N_a1 .* (a2p_idx - 1) + N_a1 .* N_a2 .* z_off;
                EV_coarse_sum = EV_coarse_sum + EV_expected_base(EV_idx_coarse) .* prob_flat;
            end
            EV_coarse = EV_coarse_sum;
        else
            exp_idx = ExpAsset_J.a2primeIndex;
            exp_prb = ExpAsset_J.a2primeProbs;
            if size(exp_idx, 3) < num_z_interp
                exp_idx = repmat(exp_idx, [1, 1, num_z_interp / size(exp_idx, 3)]);
                exp_prb = repmat(exp_prb, [1, 1, num_z_interp / size(exp_prb, 3)]);
            end

            a2p_idx = reshape(exp_idx, [N_d2, 1, N_a2, num_z_interp]);
            prob_flat = reshape(exp_prb, [N_d2, 1, N_a2, num_z_interp]);

            EV_idx_coarse = 1 + a1_off_coarse + N_a1 .* (a2p_idx - 1) + N_a1 .* N_a2 .* z_off;
            EV_coarse = EV_expected_base(EV_idx_coarse) .* prob_flat;
        end

        EV_coarse = permute(EV_coarse, [1, 2, 3, 4]); % [N_d2, N_a1, N_a2, num_z_interp]
    end

    function EV_out = compute_expected_value(EV, pi_bothz, N_a, N_bothz)
        EVinf = (EV == -Inf);
        EV_val = EV;
        EV_val(EVinf) = -1e250;
        EV_val = EV_val * pi_bothz';
        EV_val(EVinf * (pi_bothz' > 0) > 0) = -Inf;
        EV_out = reshape(EV_val, [N_a, 1, N_bothz]);
    end


end
