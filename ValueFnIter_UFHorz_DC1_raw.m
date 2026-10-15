function [V, Policy] = ValueFnIter_UFHorz_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;
has_z = Dims.has_z; has_semiz = Dims.has_semiz; has_e = Dims.has_e; has_bothz = Dims.has_bothz;
n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_e = Dims.n_e; n_semiz = Dims.n_semiz; n_d2 = Dims.n_d2;

N_z_iter = Dims.N_z_iter; N_z_slice = Dims.N_z_slice; z_mode = Dims.z_mode; spec_n_bz = Dims.spec_n_bz;
N_e_iter = Dims.N_e_iter; N_e_slice = Dims.N_e_slice; e_mode = Dims.e_mode; spec_n_e  = Dims.spec_n_e;
a_gridvals = gpuArray(a_grid);
level1n    = Dims.level1n;
level1ii   = round(linspace(1, N_a, level1n));

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
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:1:N_a);
bz_off_L2 = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, 1, N_z_slice, 1]);

zind_SL = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, N_z_slice, 1]);
eind_SL = reshape(gpuArray(0:1:N_e_slice-1), [1, 1, 1, N_e_slice]);
ze_lin_off = N_d_inner * zind_SL + (N_d_inner * N_z_slice) * eind_SL;

if N_d_outer > 1
    V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
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
            EVpre = sum(V_next .* reshape(vfoptions.pi_e_J(:, j+1), [1, 1, N_e]), 3);
        else
            EVpre = V_next;
        end
        EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
    end

    for d_outer_c = 1:N_d_outer
        if N_d_outer > 1
            d_val = Grids.d12_gridvals(:, :, d_outer_c);
        else
            d_val = Grids.d12_gridvals;
        end

        if has_cont
            if has_bothz
                if has_semiz
                    pi_bz = kron(pi_z_J(:,:,j), vfoptions.pi_semiz_J(:,:,d_outer_c,j));
                else
                    pi_bz = pi_z_J(:,:,j);
                end
            else
                pi_bz = 1;
            end
            DiscountedEV = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz);
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

                if has_cont
                    if z_mode == 0
                        RHS_L1 = RM_L1 + DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV(:, :, :, z_idx, 1);
                    end
                    ev_flat_offset = (z_c - 1) * (N_a * N_z_slice);
                else
                    RHS_L1 = RM_L1;
                end

                % 1. Optimal aprime conditional on (d_inner, 1, a_L1, z, e) for monotonicity bounds
                [~, maxindex1] = max(RHS_L1, [], 2); % [N_d_inner, 1, level1n, N_z_slice, N_e_slice]

                % 2. Full (d_inner, aprime) optimum at Level 1 points
                [v_tmp_L1, maxindex2] = max(reshape(RHS_L1, [N_d_inner * N_a, level1n, N_z_slice, N_e_slice]), [], 1);

                if N_d_outer == 1
                    V(level1ii, z_idx, e_idx, j)         = reshape(v_tmp_L1, [level1n, N_z_slice, N_e_slice]);
                    Policy(1, level1ii, z_idx, e_idx, j) = reshape(maxindex2, [1, level1n, N_z_slice, N_e_slice]);
                else
                    d_inner_L1 = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_L1     = ceil(maxindex2 / N_d_inner);
                    linear_L1  = d_inner_L1 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_L1 - 1);
                    V_fordouter(level1ii, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L1, [level1n, N_z_slice, N_e_slice]);
                    P_fordouter(level1ii, z_idx, e_idx, d_outer_c) = reshape(linear_L1, [level1n, N_z_slice, N_e_slice]);
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
                        loweredge = min(maxindex1(:, 1, ii, :, :), N_a - mg); % [N_d_inner, 1, 1, N_z_slice, N_e_slice]
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets; % [N_d_inner, mg+1, 1, N_z_slice, N_e_slice]
                    else
                        loweredge = maxindex1(:, 1, ii, :, :);
                        a1p_idx = loweredge;
                    end
                    a1p_grid_safe = reshape(a_gridvals(a1p_idx), [N_d_inner, mg+1, 1, N_z_slice, N_e_slice]);

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, spec_n_e, d_val, a1p_grid_safe, a_gridvals(curraindex), bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d_inner * (mg+1), L, N_z_slice, N_e_slice]);

                    if has_cont
                        aprimez = a1p_idx + N_a * bz_off_L2 + ev_flat_offset;
                        RHS_L2 = RM_L2 + reshape(DiscountedEV(aprimez(:)), [N_d_inner * (mg+1), 1, N_z_slice, N_e_slice]);
                    else
                        RHS_L2 = RM_L2;
                    end

                    [v_tmp_L2, maxindex] = max(RHS_L2, [], 1); % [1, L, N_z_slice, N_e_slice]

                    dind   = rem(maxindex - 1, N_d_inner) + 1;
                    allind = dind + ze_lin_off;
                    pol_L2 = maxindex + N_d_inner * (loweredge(allind) - 1);

                    if N_d_outer == 1
                        V(curraindex, z_idx, e_idx, j)         = reshape(v_tmp_L2, [L, N_z_slice, N_e_slice]);
                        Policy(1, curraindex, z_idx, e_idx, j) = reshape(pol_L2, [1, L, N_z_slice, N_e_slice]);
                    else
                        a1p_abs   = ceil(pol_L2 / N_d_inner);
                        linear_L2 = dind + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a1p_abs - 1);
                        V_fordouter(curraindex, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L2, [L, N_z_slice, N_e_slice]);
                        P_fordouter(curraindex, z_idx, e_idx, d_outer_c) = reshape(linear_L2, [L, N_z_slice, N_e_slice]);
                    end
                end
            end
        end
    end

    if N_d_outer > 1
        [V(:,:,:,j), Policy(:,:,:,:,j)] = extract_pol_local(V_fordouter, P_fordouter, N_a, N_bothz, N_e);
    end
end

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function DiscountedEV = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz)
    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat;
    EV_val(EVinf) = -1e250;
    EV_val = EV_val * pi_bz';
    EV_val(EVinf * (pi_bz' > 0) > 0) = -Inf; % Exact -Inf restoration

    % Shaped [1, N_a, 1, N_bothz, 1] for Stage 1 implicit expansion, and linear-indexed in Stage 2
    DiscountedEV = df * reshape(EV_val, [1, N_a, 1, N_bothz, 1]);
end

function [V_j, Pol_j] = extract_pol_local(V_fordouter, P_fordouter, N_a, N_bothz, N_e)
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
