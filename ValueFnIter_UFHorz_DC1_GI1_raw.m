function [V, Policy] = ValueFnIter_UFHorz_DC1_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a     = Dims.N_a;
N_bothz = Dims.N_bothz;
N_e     = Dims.N_e;
N_z     = Dims.N_z;
N_semiz = Dims.N_semiz;

has_semiz = Dims.N_semiz_raw > 0;
has_e     = Dims.N_e_raw > 0;
has_z     = Dims.special_n_z(1) > 0;
has_bothz = has_semiz || has_z;

N_d_inner = Dims.N_d_inner;
N_d_outer = Dims.N_d_outer;

a_gridvals = gpuArray(a_grid);
n2short    = vfoptions.ngridinterp;
n2long     = n2short*2 + 3;
a1prime_grid = gpuArray(interp1(1:1:N_a, gather(a_gridvals), linspace(1, N_a, N_a+(N_a-1)*n2short))');
N_a1fine   = length(a1prime_grid);

level1n  = Dims.level1n;
level1ii = round(linspace(1, N_a, level1n));

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
        N_z_iter = N_z;     N_z_slice = N_semiz; z_mode = 1; spec_n_bz = [Dims.special_n_semiz, Dims.special_n_z];
        N_e_iter = 1;       N_e_slice = N_e;     e_mode = 0;
    case 'Loop_z_parallel_semiz_e'
        N_z_iter = N_z;     N_z_slice = N_semiz; z_mode = 1; spec_n_bz = [Dims.special_n_semiz, Dims.special_n_z];
        N_e_iter = N_e;     N_e_slice = 1;       e_mode = 1;
    case 'Loop_bothz'
        N_z_iter = N_bothz; N_z_slice = 1;       z_mode = 2; spec_n_bz = Dims.special_n_bothz;
        N_e_iter = 1;       N_e_slice = N_e;     e_mode = 0;
    case 'Loop_bothz_e'
        N_z_iter = N_bothz; N_z_slice = 1;       z_mode = 2; spec_n_bz = Dims.special_n_bothz;
        N_e_iter = N_e;     N_e_slice = 1;       e_mode = 1;
end

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 3. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:1:N_a);
n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1]);

aind_SL  = reshape(gpuArray(0:1:N_a-1),       [1, N_a, 1, 1]);
bzind_SL = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, N_z_slice, 1]);
eind_SL  = reshape(gpuArray(0:1:N_e_slice-1), [1, 1, 1, N_e_slice]);
state_lin_off = aind_SL + N_a * bzind_SL + (N_a * N_z_slice) * eind_SL;

bz_off_L2 = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, 1, N_z_slice, 1]);

midpt_local = zeros(N_d_inner, 1, N_a, N_z_slice, N_e_slice, 'gpuArray');

if N_d_outer > 1
    V_fordouter  = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
end

% --- 4. Backward Induction Loop (All 6 Strategies Inlined for Buffer Reuse) ---
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
                    pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d_outer_c,j));
                else
                    pi_bz = Probs.pi_z_J(:,:,j);
                end
            else
                pi_bz = 1;
            end
            [DiscountedEV, DiscountedEVinterp] = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz, a_gridvals, a1prime_grid);
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
                else
                    e_idx = e_c;
                    e_val = get_grid_slice_local(Grids.e_gridvals_J, e_c, j, has_e);
                end

                % --- Stage 1: Coarse Level 1 Divide-and-Conquer (Conditional on d_inner) ---
                RM_L1 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, Dims.special_n_e, d_val, a_gridvals, a_gridvals(level1ii), bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a, level1n, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0
                        RHS_L1 = RM_L1 + DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV(:, :, :, z_idx, 1);
                    end
                    ev_coarse_offset = (z_c - 1) * (N_a * N_z_slice);
                    ev_fine_offset   = (z_c - 1) * (N_a1fine * N_z_slice);
                else
                    RHS_L1 = RM_L1;
                end

                % Maximize over aprime (dim 2) conditional on d_inner
                [~, maxindex1] = max(RHS_L1, [], 2); % [N_d_inner, 1, level1n, N_z_slice, N_e_slice]
                midpt_local(:, 1, level1ii, :, :) = maxindex1;

                % --- Stage 2: Level 2 Bounded Monotonicity Midpoint Search ---
                diff_a1p = max(0, maxindex1(:, 1, 2:end, :, :) - maxindex1(:, 1, 1:end-1, :, :));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 4, 5]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(level1n - 1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a - 1), 0);
                    if mg > 0
                        loweredge = min(maxindex1(:, 1, ii, :, :), N_a - mg); % [N_d_inner, 1, 1, N_z_slice, N_e_slice]
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets; % [N_d_inner, mg+1, 1, N_z_slice, N_e_slice]
                        a1p_grid_safe = reshape(a_gridvals(a1p_idx), [N_d_inner, mg+1, 1, N_z_slice, N_e_slice]);

                        RM_L2_DC = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                        RM_L2_DC = reshape(RM_L2_DC, [N_d_inner, mg+1, L, N_z_slice, N_e_slice]);

                        if has_cont
                            aprimez = a1p_idx + N_a * bz_off_L2 + ev_coarse_offset;
                            RHS_L2_DC = RM_L2_DC + reshape(DiscountedEV(aprimez(:)), [N_d_inner, mg+1, 1, N_z_slice, N_e_slice]);
                        else
                            RHS_L2_DC = RM_L2_DC;
                        end

                        [~, a1p_offset] = max(RHS_L2_DC, [], 2);
                        midpt_local(:, 1, curraindex, :, :) = loweredge + a1p_offset - 1;
                    else
                        midpt_local(:, 1, curraindex, :, :) = repmat(maxindex1(:, 1, ii, :, :), [1, 1, L, 1, 1]);
                    end
                end

                % --- Stage 3: Grid Interpolation (GI1) Fine Pass around Midpoints ---
                midpoint = max(min(midpt_local, N_a - 1), 2);
                a1primeindexes = (midpoint + (midpoint - 1)*n2short) + n2_offsets; % [N_d_inner, n2long, N_a, N_z_slice, N_e_slice]
                a1p_grid_safe  = reshape(a1prime_grid(a1primeindexes), [N_d_inner, n2long, N_a, N_z_slice, N_e_slice]);

                RM_GI = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_GI = reshape(RM_GI, [N_d_inner * n2long, N_a, N_z_slice, N_e_slice]);

                if has_cont
                    aprimez_GI = a1primeindexes + N_a1fine * bz_off_L2 + ev_fine_offset;
                    RHS_GI = RM_GI + reshape(DiscountedEVinterp(aprimez_GI(:)), [N_d_inner * n2long, N_a, N_z_slice, N_e_slice]);
                else
                    RHS_GI = RM_GI;
                end

                [v_tmp, p_tmp] = max(RHS_GI, [], 1);

                d_ind    = rem(p_tmp - 1, N_d_inner) + 1;
                L2offset = ceil(p_tmp / N_d_inner);

                linidx_lower = d_ind + (N_d_inner * n2long) * state_lin_off;
                linidx_upper = linidx_lower + N_d_inner * (n2long - 1);
                flag_val = 2 + ((L2offset >= 2 & L2offset <= n2short+1) & (RM_GI(linidx_lower) == -Inf)) ...
                             - ((L2offset >= n2short+3 & L2offset <= n2long-1) & (RM_GI(linidx_upper) == -Inf));

                allind     = d_ind + N_d_inner * state_lin_off;
                chosen_mid = midpoint(allind);

                if N_d_outer == 1
                    V(:, z_idx, e_idx, j)         = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    Policy(1, :, z_idx, e_idx, j) = d_ind;
                    Policy(2, :, z_idx, e_idx, j) = chosen_mid;
                    Policy(3, :, z_idx, e_idx, j) = L2offset;
                    Policy(4, :, z_idx, e_idx, j) = flag_val;
                else
                    V_fordouter(:, z_idx, e_idx, d_outer_c)  = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    P1_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(d_ind + N_d_inner*(d_outer_c - 1), [N_a, N_z_slice, N_e_slice]);
                    P2_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(chosen_mid, [N_a, N_z_slice, N_e_slice]);
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

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function [DiscountedEV, DiscountedEVinterp] = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz, a_gridvals, a1prime_grid)
    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat;
    EV_val(EVinf) = -1e250;
    EV_val = EV_val * pi_bz';
    EV_val(EVinf * (pi_bz' > 0) > 0) = -Inf; % Exact -Inf restoration

    DiscountedEV_2D    = df * EV_val; % [N_a, N_bothz]
    DiscountedEVinterp = interp1(a_gridvals, DiscountedEV_2D, a1prime_grid); % [N_a1fine, N_bothz]
    DiscountedEV       = reshape(DiscountedEV_2D, [1, N_a, 1, N_bothz, 1]);
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
