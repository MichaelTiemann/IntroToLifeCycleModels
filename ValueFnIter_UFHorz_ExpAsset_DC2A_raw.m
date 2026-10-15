function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---                                                                                                                                                                       
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;
has_z = Dims.has_z; has_semiz = Dims.has_semiz; has_e = Dims.has_e; has_bothz = Dims.has_bothz;
n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_e = Dims.n_e; n_semiz = Dims.n_semiz; n_d2 = Dims.n_d2;

N_z_iter = Dims.N_z_iter; N_z_slice = Dims.N_z_slice; z_mode = Dims.z_mode; spec_n_bz = Dims.spec_n_bz;
N_e_iter = Dims.N_e_iter; N_e_slice = Dims.N_e_slice; e_mode = Dims.e_mode; spec_n_e  = Dims.spec_n_e;

a1_gridvals = Grids.a1_gridvals;
a2_gridvals = Grids.a2_gridvals;
d2_gridvals = Grids.d2_gridvals;
aprimeFnParamNames = Dims.aprimeFnParamNames;
N_e_interp = Dims.N_e_interp;

% ExpAsset DC2A specific split: a1_c (DC choice), a2_c (discrete endogenous choice), exp (experience asset)
n_a1_c = Dims.n_a1(1);
n_a2_c_vec = Dims.n_a1(2:end);
n_a2_c = max(prod(n_a2_c_vec), 1);
n_exp  = Dims.n_a2;
N_exp  = Dims.N_a2;

Dims.N_d1 = N_d1; Dims.N_d2 = N_d2; Dims.N_d_inner = N_d_inner; Dims.N_d_outer = N_d_outer;
Dims.n_a1_c = n_a1_c; Dims.n_a2_c = n_a2_c; Dims.n_exp = n_exp; Dims.N_exp = N_exp;
Dims.has_z = has_z; Dims.has_semiz = has_semiz; Dims.has_e = has_e; Dims.has_bothz = has_bothz;
Dims.n_d = n_d;

% --- 2. Determine Iteration Strategy & Unified Slice Dimensions ---
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

switch IterStrategy
    case 'Full'
        N_z_iter = 1;       N_z_slice = N_bothz; z_mode = 0; spec_n_bz = Dims.special_n_bothz;
        N_e_iter = 1;       N_e_slice = N_e;     e_mode = 0;
    case 'Loop_e'
        N_z_iter = 1;       N_z_slice = N_bothz; z_mode = 0; spec_n_bz = Dims.special_n_bothz;
        N_e_iter = N_e;     N_e_slice = 1;       e_mode = 1;
    case 'Loop_z_parallel_semiz'
        N_z_iter = N_z;     N_z_slice = N_semiz; z_mode = 1; spec_n_bz = [Dims.n_semiz, Dims.special_n_z];
        N_e_iter = 1;       N_e_slice = N_e;     e_mode = 0;
    case 'Loop_z_parallel_semiz_e'
        N_z_iter = N_z;     N_z_slice = N_semiz; z_mode = 1; spec_n_bz = [Dims.n_semiz, Dims.special_n_z];
        N_e_iter = N_e;     N_e_slice = 1;       e_mode = 1;
    case 'Loop_bothz'
        N_z_iter = N_bothz; N_z_slice = 1;       z_mode = 2; spec_n_bz = Dims.special_n_bothz;
        N_e_iter = 1;       N_e_slice = N_e;     e_mode = 0;
    case 'Loop_bothz_e'
        N_z_iter = N_bothz; N_z_slice = 1;       z_mode = 2; spec_n_bz = Dims.special_n_bothz;
        N_e_iter = N_e;     N_e_slice = 1;       e_mode = 1;
end

% --- 3. Grid Extracts & Divide-and-Conquer Setup ---
a1_c_gridvals = CreateGridvals(n_a1_c, vfoptions.a1_grid(1:n_a1_c), 1);
a2_c_gridvals = CreateGridvals(n_a2_c_vec, vfoptions.a1_grid(n_a1_c+1:end), 1);
exp_gridvals  = CreateGridvals(n_exp, vfoptions.a2_grid, 1);
d2_gridvals   = CreateGridvals(n_d2, vfoptions.d2_grid, 1);

level1n  = Dims.level1n;
level1ii = round(linspace(1, n_a1_c, level1n));

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 4. Introspect ExpAsset Parameters ---
if Dims.exp_flag > 0
    l_d2_len  = length(n_d2);  if sum(n_d2) == 0;  l_d2_len = 0; end
    l_exp_len = length(n_exp); if sum(n_exp) == 0; l_exp_len = 0; end
    l_z_len   = length(n_z);   if sum(n_z) == 0;   l_z_len = 0; end
    l_e_len   = length(n_e);   if sum(n_e) == 0;   l_e_len = 0; end
    l_semiz_len = length(n_semiz); if sum(n_semiz) == 0; l_semiz_len = 0; end
    if isfield(vfoptions, 'n_u'); l_u_len = length(vfoptions.n_u); if sum(vfoptions.n_u) == 0; l_u_len = 0; end; else; l_u_len = 0; end

    if Dims.exp_flag == 3 || Dims.exp_flag == 4; num_state_args = l_d2_len + l_exp_len + l_z_len + l_e_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 2; num_state_args = l_d2_len + l_exp_len + l_z_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 5; num_state_args = l_d2_len + l_exp_len + l_semiz_len + (l_exp_len >= 2);
    elseif Dims.exp_flag == 6; num_state_args = l_d2_len + l_exp_len + l_u_len + (l_exp_len >= 2);
    else; num_state_args = l_d2_len + l_exp_len + (l_exp_len >= 2); end

    temp_args = getAnonymousFnInputNames(vfoptions.aprimeFn);
    if length(temp_args) > num_state_args; aprimeFnParamNames = temp_args(num_state_args + 1 : end);
    else; aprimeFnParamNames = {}; end
else
    aprimeFnParamNames = {};
end

if Dims.exp_flag == 6
    Grids.u_gridvals = CreateGridvals(vfoptions.n_u, vfoptions.u_grid, 1);
end

if Dims.exp_flag == 3 || Dims.exp_flag == 4; N_e_interp = N_e; else; N_e_interp = 1; end
N_e_SL_interp = min(N_e_slice, N_e_interp);

% --- 5. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:1:n_a1_c);

d2_off_L2  = reshape(repelem(gpuArray(1:1:N_d2)', N_d1, 1), [N_d_inner, 1, 1, 1, 1, 1, 1, 1]);
a2p_off_L2 = reshape(gpuArray(0:1:n_a2_c-1),                [1, 1, n_a2_c, 1, 1, 1, 1, 1]);
exp_off_L2 = reshape(gpuArray(0:1:N_exp-1),                 [1, 1, 1, 1, 1, N_exp, 1, 1]);
bz_off_L2  = reshape(gpuArray(0:1:N_z_slice-1),             [1, 1, 1, 1, 1, 1, N_z_slice, 1]);
e_off_L2   = reshape(gpuArray(0:1:N_e_SL_interp-1),         [1, 1, 1, 1, 1, 1, 1, N_e_SL_interp]);

% Linear offset over (a2_c, exp, z_slice, e_slice) for indexing loweredge of shape [N_d_inner, 1, n_a2_c, 1, n_a2_c, N_exp, N_z_slice, N_e_slice]
S_le1 = N_d_inner;
S_le2 = N_d_inner * n_a2_c;
le_state_off = reshape(gpuArray(0:1:(n_a2_c * N_exp * N_z_slice * N_e_slice - 1)), [1, 1, n_a2_c, N_exp, N_z_slice, N_e_slice]);

if N_d_outer > 1
    V_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');
    P_fordouter = zeros(n_a1_c, n_a2_c, N_exp, N_bothz, N_e, N_d_outer, 'gpuArray');
end

% --- 6. Backward Induction Loop (All 6 Strategies Inlined for Buffer Reuse) ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1')
            V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]);
        else
            V_next = [];
        end
    else
        V_next = reshape(V(:, :, :, :, :, j+1), [N_a, N_bothz, N_e]);
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
        EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);

        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, j);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, vfoptions.z_gridvals_J(:,:,j), n_e, get_grid_slice_local(vfoptions.e_gridvals_J, 'all', j, has_e));
        elseif Dims.exp_flag == 2
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, z_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 5
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_semiz, get_grid_slice_local(vfoptions.semiz_gridvals_J, 'all', j, has_semiz));
        elseif Dims.exp_flag == 6
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, vfoptions.n_u, Grids.u_gridvals);
        else
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2);
        end
        ExpAsset_J.a2primeIndex = expIndex;
        ExpAsset_J.a2primeProbs = expProbs;
    end

    for d_outer_c = 1:N_d_outer
        if N_d_outer > 1
            d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d_inner, size(Grids.d12_gridvals, 3)]);
        else
            d_val = Grids.d12_gridvals;
        end

        if has_cont
            if has_semiz
                pi_bz = kron(pi_z_J(:,:,j), vfoptions.pi_semiz_J(:,:,d_outer_c,j));
            else
                pi_bz = pi_z_J(:,:,j);
            end
            [DiscountedEV, DiscountedEV_d2] = precalc_ev_local(EVpre_flat, df, pi_bz, N_e_interp, ExpAsset_J, Dims, vfoptions);
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

                % --- Stage 1: Coarse Level 1 Divide-and-Conquer over a1_c(level1ii) ---
                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, Dims.special_n_d, n_a1_c, n_a2_c, n_exp, spec_n_bz, spec_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals(level1ii,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0 && (e_mode == 0 || N_e_interp == 1)
                        RHS_L1 = RM_L1 + DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV(:, :, :, :, :, :, z_idx, e_ev_idx);
                    end
                    ev_flat_offset = (z_c - 1)*(N_d2*n_a1_c*n_a2_c*N_exp*N_z_slice) + (e_ev_idx(1) - 1)*(N_d2*n_a1_c*n_a2_c*N_exp*N_bothz);
                else
                    RHS_L1 = RM_L1;
                end

                % 1. Optimal a1prime (dim 2) conditional on (d_inner, a2prime, a1_L1, a2, exp, z, e)
                [~, a1p_idx_L1] = max(RHS_L1, [], 2); % [N_d_inner, 1, n_a2_c, level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]

                % 2. Full (d_inner, a1prime, a2prime) optimum at Level 1 points
                [v_tmp_L1, p_tmp_L1] = max(reshape(RHS_L1, [N_d_inner * n_a1_c * n_a2_c, level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]), [], 1);

                if N_d_outer == 1
                    V(level1ii, :, :, z_idx, e_idx, j)         = reshape(v_tmp_L1, [level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                    Policy(1, level1ii, :, :, z_idx, e_idx, j) = reshape(p_tmp_L1, [1, level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                else
                    d_in_L1   = rem(p_tmp_L1 - 1, N_d_inner) + 1;
                    a12p_L1   = ceil(p_tmp_L1 / N_d_inner);
                    linear_L1 = d_in_L1 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a12p_L1 - 1);
                    V_fordouter(level1ii, :, :, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L1, [level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                    P_fordouter(level1ii, :, :, z_idx, e_idx, d_outer_c) = reshape(linear_L1, [level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                end

                % --- Stage 2: Level 2 Bounded Monotonicity Refinement ---
                diff_a1p = max(0, a1p_idx_L1(:, 1, :, 2:end, :, :, :, :) - a1p_idx_L1(:, 1, :, 1:end-1, :, :, :, :));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 5, 6, 7, 8]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(level1n - 1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), n_a1_c - 1), 0);
                    if mg > 0
                        loweredge = min(a1p_idx_L1(:, 1, :, ii, :, :, :, :), n_a1_c - mg); % [N_d_inner, 1, n_a2_c, 1, n_a2_c, N_exp, N_z_slice, N_e_slice]
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets; % [N_d_inner, mg+1, n_a2_c, 1, n_a2_c, N_exp, N_z_slice, N_e_slice]
                    else
                        loweredge = a1p_idx_L1(:, 1, :, ii, :, :, :, :);
                        a1p_idx = loweredge;
                    end
                    a1p_grid_safe = reshape(a1_c_gridvals(a1p_idx), [N_d_inner, mg+1, n_a2_c, 1, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, Dims.special_n_d, n_a1_c, n_a2_c, n_exp, spec_n_bz, spec_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals(curraindex,:), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, n_a2_c, L, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                    if has_cont
                        if N_e_SL_interp == 1 && N_e_slice > 1
                            EV_idx = d2_off_L2 + N_d2*(a1p_idx - 1) + N_d2*n_a1_c*a2p_off_L2 + N_d2*n_a1_c*n_a2_c*exp_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*bz_off_L2 + ev_flat_offset;
                        else
                            EV_idx = d2_off_L2 + N_d2*(a1p_idx - 1) + N_d2*n_a1_c*a2p_off_L2 + N_d2*n_a1_c*n_a2_c*exp_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*bz_off_L2 + N_d2*n_a1_c*n_a2_c*N_exp*N_bothz*e_off_L2 + ev_flat_offset;
                        end
                        RHS_L2 = RM_L2 + reshape(DiscountedEV_d2(EV_idx(:)), [N_d_inner, mg+1, n_a2_c, 1, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                    else
                        RHS_L2 = RM_L2;
                    end

                    [v_tmp_L2, maxindex2] = max(reshape(RHS_L2, [N_d_inner * (mg+1) * n_a2_c, L, n_a2_c, N_exp, N_z_slice, N_e_slice]), [], 1);

                    dind        = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_a2p_idx = ceil(maxindex2 / N_d_inner);
                    a1p_offset  = rem(a1p_a2p_idx - 1, mg+1) + 1;
                    a2p_ind     = ceil(a1p_a2p_idx / (mg+1));

                    % Extract loweredge corresponding to winning (dind, a2p_ind)
                    le_allind = dind + S_le1*(a2p_ind - 1) + S_le2*le_state_off;
                    a1p_abs   = loweredge(le_allind) + a1p_offset - 1;

                    if N_d_outer == 1
                        pol_L2 = dind + N_d_inner * (a1p_abs - 1) + (N_d_inner * n_a1_c) * (a2p_ind - 1);
                        V(curraindex, :, :, z_idx, e_idx, j)         = reshape(v_tmp_L2, [L, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                        Policy(1, curraindex, :, :, z_idx, e_idx, j) = reshape(pol_L2, [1, L, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                    else
                        a12p_abs  = a1p_abs + n_a1_c * (a2p_ind - 1);
                        linear_L2 = dind + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a12p_abs - 1);
                        V_fordouter(curraindex, :, :, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L2, [L, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                        P_fordouter(curraindex, :, :, z_idx, e_idx, d_outer_c) = reshape(linear_L2, [L, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                    end
                end
            end
        end
    end

    if N_d_outer > 1
        [V(:,:,:,:,:,j), Policy(:,:,:,:,:,:,j)] = extract_pol_local(V_fordouter, P_fordouter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e);
    end
end

V = reshape(V, [N_a, N_bothz, N_e, N_j]);
Policy = reshape(Policy, [1, N_a, N_bothz, N_e, N_j]);

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function [DiscountedEV, DiscountedEV_d2] = precalc_ev_local(EVpre_flat, df, pi_bz, N_e_interp, ExpAsset_J, Dims, vfoptions)
    N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_bothz = Dims.N_bothz; N_semiz = Dims.N_semiz;
    N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d_inner = Dims.N_d_inner;
    n_a1_c = Dims.n_a1_c; n_a2_c = Dims.n_a2_c; N_exp = Dims.N_exp; n_a2 = Dims.n_a2;

    EV_val = reshape(EVpre_flat, [N_a1*N_a2, N_bothz]);
    a2p_idx_base = ExpAsset_J.a2primeIndex;
    a2p_weight_base = ExpAsset_J.a2primeProbs;

    if length(n_a2) == 1
        N_d_exp = size(a2p_idx_base, 1);
        N_a2_exp = size(a2p_idx_base, 2);
        N_3rd = max(size(a2p_idx_base, 3), 1);

        if Dims.exp_flag ~= 6 && N_3rd > 1 && N_3rd ~= N_bothz
            a2p_idx_base = reshape(repmat(reshape(a2p_idx_base, [N_d_exp, N_a2_exp, 1, N_3rd]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
            a2p_weight_base = reshape(repmat(reshape(a2p_weight_base, [N_d_exp, N_a2_exp, 1, N_3rd]), 1, 1, N_semiz, 1), [N_d_exp, N_a2_exp, N_bothz]);
            N_3rd = N_bothz;
        end

        a1_base = repelem(gpuArray(1:1:N_a1)', N_d_exp, N_a2_exp, N_3rd);
        idx_lo = a1_base + N_a1 * repmat(a2p_idx_base - 1, N_a1, 1, 1);
        idx_hi = a1_base + N_a1 * repmat(a2p_idx_base, N_a1, 1, 1);
        prob_exp = repmat(a2p_weight_base, N_a1, 1, 1, N_bothz);

        Vlower = reshape(EV_val(idx_lo(:), :), [N_d_exp*N_a1, N_a2_exp, N_3rd, N_bothz]);
        Vupper = reshape(EV_val(idx_hi(:), :), [N_d_exp*N_a1, N_a2_exp, N_3rd, N_bothz]);

        skipinterp = (Vlower == Vupper);
        prob_exp(skipinterp) = 0;

        EV_u = prob_exp .* Vlower + (1 - prob_exp) .* Vupper;
        EV_u(prob_exp == 0) = Vupper(prob_exp == 0);
        EV_u(prob_exp == 1) = Vlower(prob_exp == 1);
    else
        N_d_exp = size(a2p_idx_base, 2);
        N_a2_exp = size(a2p_idx_base, 3);
        N_3rd = max(size(a2p_idx_base, 4), 1);

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
        n_a2_1 = n_a2(1);

        idx_ll = a1_base + N_a1 * repmat(loIdx_1 + n_a2_1*(loIdx_2-1) - 1, N_a1, 1, 1);
        idx_hl = a1_base + N_a1 * repmat((loIdx_1+1) + n_a2_1*(loIdx_2-1) - 1, N_a1, 1, 1);
        idx_lh = a1_base + N_a1 * repmat(loIdx_1 + n_a2_1*loIdx_2 - 1, N_a1, 1, 1);
        idx_hh = a1_base + N_a1 * repmat((loIdx_1+1) + n_a2_1*loIdx_2 - 1, N_a1, 1, 1);

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

    EV_coarse = reshape(EV_coarse, [N_d_exp, n_a1_c, n_a2_c, 1, 1, N_a2_exp, N_bothz, N_e_interp]);
    if N_d_exp == 1 && N_d2 > 1; EV_coarse = repmat(EV_coarse, [N_d2, 1, 1, 1, 1, 1, 1, 1]); end
    if N_a2_exp == 1 && N_exp > 1; EV_coarse = repmat(EV_coarse, [1, 1, 1, 1, 1, N_exp, 1, 1]); end

    DiscountedEV_d2 = df * EV_coarse;
    if N_d1 > 1
        DiscountedEV = repelem(DiscountedEV_d2, N_d1, 1, 1, 1, 1, 1, 1, 1);
    else
        DiscountedEV = DiscountedEV_d2;
    end
    DiscountedEV = reshape(DiscountedEV, [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_bothz, N_e_interp]);
end

function [V_j, Pol_j] = extract_pol_local(V_fordouter, P_fordouter, n_a1_c, n_a2_c, N_exp, N_bothz, N_e)
    N_a = n_a1_c * n_a2_c * N_exp;
    [V_j, max_d_out] = max(V_fordouter, [], 6);
    lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);
    Pol_j = reshape(P_fordouter(lin_idx), [1, n_a1_c, n_a2_c, N_exp, N_bothz, N_e]);
end

function slice = get_grid_slice_local(grid_J, idx, j, has_var)
    if ~has_var || isempty(grid_J); slice = []; return; end
    if strcmp(idx, 'all')
        if ndims(grid_J) >= 3; slice = grid_J(:, :, j); else; slice = grid_J; end
    else
        if ndims(grid_J) >= 3; slice = grid_J(idx, :, j); else; slice = grid_J(idx, :); end
    end
end
