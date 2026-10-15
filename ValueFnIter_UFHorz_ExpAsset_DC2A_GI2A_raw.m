function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC2A_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;

has_z     = Dims.special_n_z(1) > 0;
has_semiz = Dims.N_semiz_raw > 0;
has_e     = Dims.N_e_raw > 0;
has_bothz = has_semiz || has_z;

n_e = Dims.n_e; n_semiz = Dims.n_semiz;
n_d1 = vfoptions.n_d1; n_d2 = vfoptions.n_d2;
N_d1 = max(prod(n_d1), 1); N_d2 = max(prod(n_d2), 1);
if isfield(vfoptions, 'n_d3') && prod(vfoptions.n_d3) > 0
    n_d3 = vfoptions.n_d3; N_d3 = max(prod(n_d3), 1);
else
    n_d3 = 0; N_d3 = 1;
end
N_d_inner = N_d1 * N_d2;
N_d_outer = N_d3;

% ExpAsset DC2A_GI2A specific split: a1_c (interpolated choice), a2_c (discrete endogenous choice), exp (experience asset)
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

% --- 3. Grid Extracts, Interpolation & Divide-and-Conquer Setup ---
a1_c_gridvals = CreateGridvals(n_a1_c, vfoptions.a1_grid(1:n_a1_c), 1);
a2_c_gridvals = CreateGridvals(n_a2_c_vec, vfoptions.a1_grid(n_a1_c+1:end), 1);
exp_gridvals  = CreateGridvals(n_exp, vfoptions.a2_grid, 1);
d2_gridvals   = CreateGridvals(n_d2, vfoptions.d2_grid, 1);

n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
a1prime_grid_col = gpuArray(interp1(1:1:n_a1_c, gather(a1_c_gridvals), linspace(1, n_a1_c, n_a1_c+(n_a1_c-1)*n2short))');
a1prime_grid_col = reshape(a1prime_grid_col, [], 1);
N_a1fine = length(a1prime_grid_col);

level1n = Dims.level1n;
level1ii = round(linspace(1, n_a1_c, level1n));

Grids.a1_c_gridvals = a1_c_gridvals;
Grids.a2_c_gridvals = a2_c_gridvals;
Grids.exp_gridvals  = exp_gridvals;
Grids.a1prime_grid_col = a1prime_grid_col;
Dims.n2short = n2short; Dims.n2long = n2long; Dims.N_a1fine = N_a1fine;

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
    u_gridvals = CreateGridvals(vfoptions.n_u, vfoptions.u_grid, 1);
end

if Dims.exp_flag == 3 || Dims.exp_flag == 4; N_e_interp = N_e; else; N_e_interp = 1; end
N_e_SL_interp = min(N_e_slice, N_e_interp);

% --- 5. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:1:n_a1_c);
n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1, 1, 1, 1]);

d_off_L2   = reshape(gpuArray(1:1:N_d_inner),       [N_d_inner, 1, 1, 1, 1, 1, 1, 1]);
a2p_off_L2 = reshape(gpuArray(0:1:n_a2_c-1),        [1, 1, n_a2_c, 1, 1, 1, 1, 1]);
exp_off_L2 = reshape(gpuArray(0:1:N_exp-1),         [1, 1, 1, 1, 1, N_exp, 1, 1]);
bz_off_L2  = reshape(gpuArray(0:1:N_z_slice-1),     [1, 1, 1, 1, 1, 1, N_z_slice, 1]);
e_off_L2   = reshape(gpuArray(0:1:N_e_SL_interp-1), [1, 1, 1, 1, 1, 1, 1, N_e_SL_interp]);

S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * n_a2_c;
S_mid2 = N_d_inner; S_mid3 = S_mid2 * n_a2_c;

state_lin_off = reshape(gpuArray(0:1:(N_a * N_z_slice * N_e_slice - 1)), [1, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);

midpt_local = zeros(N_d_inner, 1, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice, 'gpuArray');

if N_d_outer > 1
    V_fordouter  = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
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
        V_next = V(:, :, :, j+1);
    end
    has_cont = ~isempty(V_next);

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);

    if has_cont
        df = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
        if has_e
            EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3);
        else
            EVpre = V_next;
        end
        EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);

        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, j);
        if Dims.exp_flag == 3 || Dims.exp_flag == 4
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, Grids.z_gridvals_J(:,:,j), n_e, get_grid_slice_local(Grids.e_gridvals_J, 'all', j, has_e));
        elseif Dims.exp_flag == 2
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_z, z_gridvals_J(:,:,j));
        elseif Dims.exp_flag == 5
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, n_semiz, get_grid_slice_local(vfoptions.semiz_gridvals_J, 'all', j, has_semiz));
        elseif Dims.exp_flag == 6
            [expIndex, expProbs] = CreateExperienceAssetFnMatrix_UDisc(vfoptions.aprimeFn, n_d2, n_exp, d2_gridvals, vfoptions.a2_grid, aprimeFnParamsVec, 2, vfoptions.n_u, u_gridvals);
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
                pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
            else
                pi_bz = Probs.pi_z_J(:,:,j);
            end
            [DiscountedEV, DiscountedEVinterp] = precalc_ev_local(EVpre_flat, df, pi_bz, N_e_interp, ExpAsset_J, Dims, Grids, vfoptions);
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
                    e_val = get_grid_slice_local(Grids.e_gridvals_J, 'all', j, has_e);
                    e_ev_idx = 1:N_e_interp;
                else
                    e_idx = e_c;
                    e_val = get_grid_slice_local(Grids.e_gridvals_J, e_c, j, has_e);
                    e_ev_idx = min(e_c, N_e_interp);
                end

                % --- Stage 1: Coarse Level 1 Divide-and-Conquer over a1_c(level1ii) ---
                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, Dims.special_n_d, n_a1_c, n_a2_c, n_exp, spec_n_bz, Dims.special_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals(level1ii), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, n_a1_c, n_a2_c, level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0 && (e_mode == 0 || N_e_interp == 1)
                        RHS_L1 = RM_L1 + DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV(:, :, :, :, :, :, z_idx, e_ev_idx);
                    end
                    ev_coarse_offset = (z_c - 1)*(N_d_inner*n_a1_c*n_a2_c*N_exp*N_z_slice)   + (e_ev_idx(1) - 1)*(N_d_inner*n_a1_c*n_a2_c*N_exp*N_bothz);
                    ev_fine_offset   = (z_c - 1)*(N_d_inner*N_a1fine*n_a2_c*N_exp*N_z_slice) + (e_ev_idx(1) - 1)*(N_d_inner*N_a1fine*n_a2_c*N_exp*N_bothz);
                else
                    RHS_L1 = RM_L1;
                end

                % Maximize over a1prime (dim 2) conditional on (d_inner, a2prime, a1_L1, a2, exp, z, e)
                [~, a1p_idx_L1] = max(RHS_L1, [], 2); % [N_d_inner, 1, n_a2_c, level1n, n_a2_c, N_exp, N_z_slice, N_e_slice]
                midpt_local(:, :, :, level1ii, :, :, :, :) = a1p_idx_L1;

                % --- Stage 2: Level 2 Bounded Monotonicity Midpoint Search ---
                diff_a1p = max(0, a1p_idx_L1(:, 1, :, 2:end, :, :, :, :) - a1p_idx_L1(:, 1, :, 1:end-1, :, :, :, :));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 5, 6, 7, 8]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

		for ii = 1:(level1n - 1)
		    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
		    L = length(curraindex);
		    if L == 0; continue; end

		    mg = max(min(maxgap_vec(ii), n_a1_c - 1), 0);
		    if mg > 0
			loweredge = min(a1p_idx_L1(:, 1, :, ii, :, :, :, :), n_a1_c - mg);
			a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1, 1]);
			a1p_idx = loweredge + a1p_offsets;
			a1p_grid_safe = reshape(a1_c_gridvals(a1p_idx), [N_d_inner, mg+1, n_a2_c, 1, n_a2_c, N_exp, N_z_slice, N_e_slice]);

			RM_L2_DC = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, Dims.special_n_d, n_a1_c, n_a2_c, n_exp, spec_n_bz, Dims.special_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals(curraindex), a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
			RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, n_a2_c, L, n_a2_c, N_exp, N_z_slice, N_e_slice]);

			if has_cont
			    if N_e_SL_interp == 1 && N_e_slice > 1
				da1p_coarse = d_off_L2 + N_d_inner*(a1p_idx - 1) + N_d_inner*n_a1_c*a2p_off_L2 + N_d_inner*n_a1_c*n_a2_c*exp_off_L2 + N_d_inner*n_a1_c*n_a2_c*N_exp*bz_off_L2 + ev_coarse_offset;
			    else
				da1p_coarse = d_off_L2 + N_d_inner*(a1p_idx - 1) + N_d_inner*n_a1_c*a2p_off_L2 + N_d_inner*n_a1_c*n_a2_c*exp_off_L2 + N_d_inner*n_a1_c*n_a2_c*N_exp*bz_off_L2 + N_d_inner*n_a1_c*n_a2_c*N_exp*N_bothz*e_off_L2 + ev_coarse_offset;
			    end
			    RHS_L2_DC = RM_L2_DC + reshape(DiscountedEV(da1p_coarse(:)), [N_d_inner, mg+1, n_a2_c, 1, n_a2_c, N_exp, N_z_slice, N_e_slice]);
			else
			    RHS_L2_DC = RM_L2_DC;
			end

			[~, a1p_offset] = max(RHS_L2_DC, [], 2);
			midpt_local(:, :, :, curraindex, :, :, :, :) = loweredge + a1p_offset - 1;
		    else
			midpt_local(:, :, :, curraindex, :, :, :, :) = repmat(a1p_idx_L1(:, 1, :, ii, :, :, :, :), [1, 1, 1, L, 1, 1, 1, 1]);
		    end
		end
                % --- Stage 3: Grid Interpolation (GI2A) Fine Pass around Midpoints ---
                midpoint = max(min(midpt_local, n_a1_c - 1), 2);
                a1primeindexes = (midpoint + (midpoint - 1)*n2short) + n2_offsets; % [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]
                a1p_grid_safe  = reshape(a1prime_grid_col(a1primeindexes), [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                RM_GI = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, Dims.special_n_d, n_a1_c, n_a2_c, n_exp, spec_n_bz, Dims.special_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_GI = reshape(RM_GI, [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                if has_cont
                    if N_e_SL_interp == 1 && N_e_slice > 1
                        da1p_fine = d_off_L2 + N_d_inner*(a1primeindexes - 1) + N_d_inner*N_a1fine*a2p_off_L2 + N_d_inner*N_a1fine*n_a2_c*exp_off_L2 + N_d_inner*N_a1fine*n_a2_c*N_exp*bz_off_L2 + ev_fine_offset;
                    else
                        da1p_fine = d_off_L2 + N_d_inner*(a1primeindexes - 1) + N_d_inner*N_a1fine*a2p_off_L2 + N_d_inner*N_a1fine*n_a2_c*exp_off_L2 + N_d_inner*N_a1fine*n_a2_c*N_exp*bz_off_L2 + N_d_inner*N_a1fine*n_a2_c*N_exp*N_bothz*e_off_L2 + ev_fine_offset;
                    end
                    RHS_GI = RM_GI + reshape(DiscountedEVinterp(da1p_fine(:)), [N_d_inner, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                else
                    RHS_GI = RM_GI;
                end

                [v_tmp, p_tmp] = max(reshape(RHS_GI, [N_d_inner * n2long * n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]), [], 1);

                d_ind          = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p     = ceil(p_tmp / N_d_inner);
                L2offset       = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                % Boundary -Inf checks & midpoint extraction using unified state_lin_off
                linidx_lower = d_ind + S2*(a2p_idx_chosen - 1) + S3*state_lin_off;
                linidx_upper = linidx_lower + S1*(n2long - 1);
                flag_val = 2 + ((L2offset >= 2 & L2offset <= n2short+1) & (RM_GI(linidx_lower) == -Inf)) ...
                             - ((L2offset >= n2short+3 & L2offset <= n2long-1) & (RM_GI(linidx_upper) == -Inf));

                midpt_idx    = d_ind + S_mid2*(a2p_idx_chosen - 1) + S_mid3*state_lin_off;
                chosen_midpt = midpoint(midpt_idx);
                joint_a_idx  = chosen_midpt + n_a1_c * (a2p_idx_chosen - 1);

                if N_d_outer == 1
                    V(:, z_idx, e_idx, j)         = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    Policy(1, :, z_idx, e_idx, j) = reshape(d_ind, [1, N_a, N_z_slice, N_e_slice]);
                    Policy(2, :, z_idx, e_idx, j) = reshape(joint_a_idx, [1, N_a, N_z_slice, N_e_slice]);
                    Policy(3, :, z_idx, e_idx, j) = reshape(L2offset, [1, N_a, N_z_slice, N_e_slice]);
                    Policy(4, :, z_idx, e_idx, j) = reshape(flag_val, [1, N_a, N_z_slice, N_e_slice]);
                else
                    V_fordouter(:, z_idx, e_idx, d_outer_c)  = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    P1_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(d_ind + N_d_inner*(d_outer_c - 1), [N_a, N_z_slice, N_e_slice]);
                    P2_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(joint_a_idx, [N_a, N_z_slice, N_e_slice]);
                    P3_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(L2offset, [N_a, N_z_slice, N_e_slice]);
                    P4_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(flag_val, [N_a, N_z_slice, N_e_slice]);
                end
            end
        end
    end

    if N_d_outer > 1
        [V(:,:,:,j), Policy(:,:,:,:,j)] = extract_pol_local(V_fordouter, P1_fordouter, P2_fordouter, P3_fordouter, P4_fordouter, N_a, N_bothz, N_e);
    end
end

% --- 7. Post-Loop Policy Adjustments (Unified over all J) ---
L2_adjust = (Policy(3,:,:,:,:) - 1 - n2short) .* (Policy(4,:,:,:,:) == 2) ...
          + (Policy(3,:,:,:,:) - 1)           .* (Policy(4,:,:,:,:) == 1) ...
          + (Policy(3,:,:,:,:) - 2*n2short - 1) .* (Policy(4,:,:,:,:) == 3);

Policy(2,:,:,:,:) = Policy(2,:,:,:,:) + (Policy(4,:,:,:,:) == 3) - (Policy(4,:,:,:,:) == 1);
Policy(3,:,:,:,:) = max(min(1 + n2short + L2_adjust, 1 + 2*n2short), 1);
Policy = Policy(1:3, :, :, :, :);

if Dims.n_d(1) == 0
    Policy = Policy(2:3, :, :, :, :);
end

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function [DiscountedEV, DiscountedEVinterp] = precalc_ev_local(EVpre_flat, df, pi_bz, N_e_interp, ExpAsset_J, Dims, Grids, vfoptions)
    N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_bothz = Dims.N_bothz; N_semiz = Dims.N_semiz;
    N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d_inner = Dims.N_d_inner; N_a1fine = Dims.N_a1fine;
    n_a1_c = Dims.n_a1_c; n_a2_c = Dims.n_a2_c; N_exp = Dims.N_exp; n_a2 = Dims.n_a2;
    a1_c_gridvals = Grids.a1_c_gridvals; a1prime_grid_col = Grids.a1prime_grid_col;

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
        EV_loy = p1_loy .* V_ll + (1-p1_loy) .* V_hl;
        EV_loy(p1_loy == 0) = V_hl(p1_loy == 0);
        EV_loy(p1_loy == 1) = V_ll(p1_loy == 1);

        p1_hiy = prb1_full; p1_hiy(V_lh == V_hh) = 0;
        EV_hiy = p1_hiy .* V_lh + (1-p1_hiy) .* V_hh;
        EV_hiy(p1_hiy == 0) = V_hh(p1_hiy == 0);
        EV_hiy(p1_hiy == 1) = V_lh(p1_hiy == 1);

        p2 = prb2_full; p2(EV_loy == EV_hiy) = 0;
        EV_u = p2 .* EV_loy + (1-p2) .* EV_hiy;
        EV_u(p2 == 0) = EV_hiy(p2 == 0);
        EV_u(p2 == 1) = EV_loy(p2 == 1);
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

    % Split N_a1 into (n_a1_c, n_a2_c) and interpolate strictly over n_a1_c
    EV_coarse = reshape(EV_coarse, [N_d_exp, n_a1_c, n_a2_c, 1, 1, N_a2_exp, N_bothz, N_e_interp]);
    if N_d_exp == 1 && N_d2 > 1; EV_coarse = repmat(EV_coarse, [N_d2, 1, 1, 1, 1, 1, 1, 1]); end
    if N_a2_exp == 1 && N_exp > 1; EV_coarse = repmat(EV_coarse, [1, 1, 1, 1, 1, N_exp, 1, 1]); end

    DiscountedEV_d2 = df * EV_coarse;
    DiscountedEV_perm = permute(DiscountedEV_d2, [2, 1, 3, 6, 7, 8, 4, 5]); % [n_a1_c, N_d2, n_a2_c, N_exp, N_bothz, N_e_interp]
    DiscountedEVinterp_perm = interp1(a1_c_gridvals, DiscountedEV_perm, a1prime_grid_col);
    DiscountedEVinterp_d2 = permute(DiscountedEVinterp_perm, [2, 1, 3, 7, 8, 4, 5, 6]); % [N_d2, N_a1fine, n_a2_c, 1, 1, N_exp, N_bothz, N_e_interp]

    if N_d1 > 1
        DiscountedEV       = repelem(DiscountedEV_d2, N_d1, 1, 1, 1, 1, 1, 1, 1);
        DiscountedEVinterp = repelem(DiscountedEVinterp_d2, N_d1, 1, 1, 1, 1, 1, 1, 1);
    else
        DiscountedEV       = DiscountedEV_d2;
        DiscountedEVinterp = DiscountedEVinterp_d2;
    end

    DiscountedEV       = reshape(DiscountedEV, [N_d_inner, n_a1_c, n_a2_c, 1, 1, N_exp, N_bothz, N_e_interp]);
    DiscountedEVinterp = reshape(DiscountedEVinterp, [N_d_inner, N_a1fine, n_a2_c, 1, 1, N_exp, N_bothz, N_e_interp]);
end

function [V_j, Pol_j] = extract_pol_local(V_fordouter, P1, P2, P3, P4, N_a, N_bothz, N_e)
    [V_j, max_d_out] = max(V_fordouter, [], 4);
    lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);

    Pol_j = zeros(4, N_a, N_bothz, N_e, 'gpuArray');
    Pol_j(1,:,:,:) = reshape(P1(lin_idx), [1, N_a, N_bothz, N_e]);
    Pol_j(2,:,:,:) = reshape(P2(lin_idx), [1, N_a, N_bothz, N_e]);
    Pol_j(3,:,:,:) = reshape(P3(lin_idx), [1, N_a, N_bothz, N_e]);
    Pol_j(4,:,:,:) = reshape(P4(lin_idx), [1, N_a, N_bothz, N_e]);
end

function slice = get_grid_slice_local(grid_J, idx, j, has_var)
    if ~has_var || isempty(grid_J); slice = []; return; end
    if strcmp(idx, 'all')
        if ndims(grid_J) >= 3; slice = grid_J(:, :, j); else; slice = grid_J; end
    else
        if ndims(grid_J) >= 3; slice = grid_J(idx, :, j); else; slice = grid_J(idx, :); end
    end
end
