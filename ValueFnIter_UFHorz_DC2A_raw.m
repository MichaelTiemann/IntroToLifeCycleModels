function [V, Policy] = ValueFnIter_UFHorz_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

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

n_a1 = n_a(1);
n_a2 = n_a(2:end);
N_a1 = n_a1;
N_a2 = max(prod(n_a2), 1);
N_a  = N_a1 * N_a2;

a1_gridvals = gpuArray(a_grid(1:N_a1));
a2_gridvals = CreateGridvals(n_a2, gpuArray(a_grid(N_a1+1:end)), 1);

if ~isfield(Dims, 'level1n') || isempty(Dims.level1n)
    Dims.level1n = min(floor(sqrt(N_a1)), N_a1);
end
level1n  = Dims.level1n(1);
level1ii = round(linspace(1, N_a1, level1n));

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

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 3. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:1:N_a1);

a2p_off_L2 = reshape(gpuArray(0:1:N_a2-1),      [1, 1, N_a2, 1, 1, 1, 1]);
bz_off_L2  = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, 1, 1, 1, N_z_slice, 1]);

% Linear offset over (a2, z_slice, e_slice) for indexing loweredge of shape [N_d_inner, 1, N_a2, 1, N_a2, N_z_slice, N_e_slice]
S_le1 = N_d_inner;
S_le2 = N_d_inner * N_a2;
le_state_off = reshape(gpuArray(0:1:(N_a2 * N_z_slice * N_e_slice - 1)), [1, 1, N_a2, N_z_slice, N_e_slice]);

if N_d_outer > 1
    V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
    P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
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
        V_next = reshape(V(:, :, :, :, j+1), [N_a, N_bothz, N_e]);
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
            DiscountedEV = precalc_ev_local(EVpre_flat, df, pi_bz, N_a1, N_a2, N_bothz);
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

                % --- Stage 1: Coarse Level 1 Divide-and-Conquer over a1_gridvals(level1ii) ---
                RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, spec_n_bz, Dims.special_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals(level1ii,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, level1n, N_a2, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0
                        RHS_L1 = RM_L1 + DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV(:, :, :, :, :, z_idx, 1);
                    end
                    ev_flat_offset = (z_c - 1) * (N_a1 * N_a2 * N_z_slice);
                else
                    RHS_L1 = RM_L1;
                end

                % 1. Optimal a1prime (dim 2) conditional on (d_inner, 1, a2prime, a1_L1, a2, z, e)
                [~, maxindex1] = max(RHS_L1, [], 2); % [N_d_inner, 1, N_a2, level1n, N_a2, N_z_slice, N_e_slice]

                % 2. Full (d_inner, a1prime, a2prime) optimum at Level 1 points
                [v_tmp_L1, p_tmp_L1] = max(reshape(RHS_L1, [N_d_inner * N_a1 * N_a2, level1n, N_a2, N_z_slice, N_e_slice]), [], 1);

                if N_d_outer == 1
                    V(level1ii, :, z_idx, e_idx, j)         = reshape(v_tmp_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    Policy(1, level1ii, :, z_idx, e_idx, j) = reshape(p_tmp_L1, [1, level1n, N_a2, N_z_slice, N_e_slice]);
                else
                    d_in_L1   = rem(p_tmp_L1 - 1, N_d_inner) + 1;
                    a12p_L1   = ceil(p_tmp_L1 / N_d_inner);
                    linear_L1 = d_in_L1 + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a12p_L1 - 1);
                    V_fordouter(level1ii, :, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    P_fordouter(level1ii, :, z_idx, e_idx, d_outer_c) = reshape(linear_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                end

                % --- Stage 2: Level 2 Bounded Monotonicity Refinement ---
                diff_a1p = max(0, maxindex1(:, 1, :, 2:end, :, :, :) - maxindex1(:, 1, :, 1:end-1, :, :, :));
                maxgap_vec = gather(reshape(max(diff_a1p, [], [1, 3, 5, 6, 7]), [level1n - 1, 1]));
                if isempty(maxgap_vec); maxgap_vec = 0; end

                for ii = 1:(level1n - 1)
                    curraindex = level1ii(ii)+1 : level1ii(ii+1)-1;
                    L = length(curraindex);
                    if L == 0; continue; end

                    mg = max(min(maxgap_vec(ii), N_a1 - 1), 0);
                    if mg > 0
                        loweredge = min(maxindex1(:, 1, :, ii, :, :, :), N_a1 - mg); % [N_d_inner, 1, N_a2, 1, N_a2, N_z_slice, N_e_slice]
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets; % [N_d_inner, mg+1, N_a2, 1, N_a2, N_z_slice, N_e_slice]
                    else
                        loweredge = maxindex1(:, 1, :, ii, :, :, :);
                        a1p_idx = loweredge;
                    end
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), [N_d_inner, mg+1, N_a2, 1, N_a2, N_z_slice, N_e_slice]);

                    RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, spec_n_bz, Dims.special_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals(curraindex,:), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);
                    RM_L2 = reshape(RM_L2, [N_d_inner, mg+1, N_a2, L, N_a2, N_z_slice, N_e_slice]);

                    if has_cont
                        EV_idx = a1p_idx + N_a1 * a2p_off_L2 + (N_a1 * N_a2) * bz_off_L2 + ev_flat_offset;
                        RHS_L2 = RM_L2 + DiscountedEV(EV_idx);
                    else
                        RHS_L2 = RM_L2;
                    end

                    [v_tmp_L2, maxindex2] = max(reshape(RHS_L2, [N_d_inner * (mg+1) * N_a2, L, N_a2, N_z_slice, N_e_slice]), [], 1);

                    dind        = rem(maxindex2 - 1, N_d_inner) + 1;
                    a1p_a2p_idx = ceil(maxindex2 / N_d_inner);
                    a1p_offset  = rem(a1p_a2p_idx - 1, mg+1) + 1;
                    a2p_ind     = ceil(a1p_a2p_idx / (mg+1));

                    % Extract loweredge corresponding to winning (dind, a2p_ind)
                    le_allind = dind + S_le1*(a2p_ind - 1) + S_le2*le_state_off;
                    a1p_abs   = loweredge(le_allind) + a1p_offset - 1;

                    if N_d_outer == 1
                        pol_L2 = dind + N_d_inner * (a1p_abs - 1) + (N_d_inner * N_a1) * (a2p_ind - 1);
                        V(curraindex, :, z_idx, e_idx, j)         = reshape(v_tmp_L2, [L, N_a2, N_z_slice, N_e_slice]);
                        Policy(1, curraindex, :, z_idx, e_idx, j) = reshape(pol_L2, [1, L, N_a2, N_z_slice, N_e_slice]);
                    else
                        a12p_abs  = a1p_abs + N_a1 * (a2p_ind - 1);
                        linear_L2 = dind + N_d_inner * (d_outer_c - 1) + (N_d_inner * N_d_outer) * (a12p_abs - 1);
                        V_fordouter(curraindex, :, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L2, [L, N_a2, N_z_slice, N_e_slice]);
                        P_fordouter(curraindex, :, z_idx, e_idx, d_outer_c) = reshape(linear_L2, [L, N_a2, N_z_slice, N_e_slice]);
                    end
                end
            end
        end
    end

    if N_d_outer > 1
        [V(:,:,:,:,j), Policy(:,:,:,:,:,j)] = extract_pol_local(V_fordouter, P_fordouter, N_a1, N_a2, N_bothz, N_e);
    end
end

V = reshape(V, [N_a, N_bothz, N_e, N_j]);
Policy = reshape(Policy, [1, N_a, N_bothz, N_e, N_j]);

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function DiscountedEV = precalc_ev_local(EVpre_flat, df, pi_bz, N_a1, N_a2, N_bothz)
    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat;
    EV_val(EVinf) = -1e250;
    EV_expected = EV_val * pi_bz';
    EV_expected((EVinf * (pi_bz' > 0)) > 0) = -Inf;

    % Shaped [1, N_a1, N_a2, 1, 1, N_bothz, 1] for Stage 1 implicit expansion, and linear-indexed in Stage 2
    DiscountedEV = df * reshape(EV_expected, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
end

function [V_j, Pol_j] = extract_pol_local(V_fordouter, P_fordouter, N_a1, N_a2, N_bothz, N_e)
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
