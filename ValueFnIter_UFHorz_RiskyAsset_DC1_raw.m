function [V, Policy] = ValueFnIter_UFHorz_RiskyAsset_DC1_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_grid, z_gridvals_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d3 = Dims.N_d3;
N_d13 = Dims.N_d13; N_d_outer = Dims.N_d_outer; N_d = Dims.N_d;
has_z = Dims.has_z; has_semiz = Dims.has_semiz; has_e = Dims.has_e; has_bothz = Dims.has_bothz;
n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_u = vfoptions.n_u;

N_z_iter = Dims.N_z_iter; N_z_slice = Dims.N_z_slice; z_mode = Dims.z_mode; spec_n_bz = Dims.spec_n_bz;
N_e_iter = Dims.N_e_iter; N_e_slice = Dims.N_e_slice; e_mode = Dims.e_mode; spec_n_e  = Dims.spec_n_e;

a1_gridvals = Grids.a1_gridvals;
a2_gridvals = Grids.a2_gridvals;
u_gridvals  = Grids.u_gridvals;
pi_u        = vfoptions.pi_u;
aprimeFnParamNames = Dims.aprimeFnParamNames;

level1n  = Dims.level1n;
level1ii = round(linspace(1, N_a1, level1n));

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 2. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a1, N_a2, N_bothz, N_e, N_j, 'gpuArray');

a1p_offsets_base = gpuArray(0:1:N_a1);

d3_off_L2 = reshape(repelem(gpuArray(1:1:N_d3)', N_d1, 1), [N_d13, 1, 1, 1, 1, 1]);
bz_off_L2 = reshape(gpuArray(0:1:N_z_slice-1),             [1, 1, 1, 1, N_z_slice, 1]);

bz_SL_4D  = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, 1, N_z_slice, 1]);
d2_bz_off = (N_d3 * N_a1) * bz_SL_4D;

% Offset into loweredge [N_d13, 1, 1, N_a2, N_z_slice, N_e_slice] for Stage 2
le_state_off = N_d13 * reshape(gpuArray(0:1:(N_a2 * N_z_slice * N_e_slice - 1)), [1, 1, N_a2, N_z_slice, N_e_slice]);

if N_d_outer > 1
    V_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
    P_fordouter = zeros(N_a1, N_a2, N_bothz, N_e, N_d_outer, 'gpuArray');
end

% --- 3. Backward Induction Loop (All 6 Strategies Inlined for Buffer Reuse) ---
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
            EVpre = sum(V_next .* reshape(vfoptions.pi_e_J(:, j+1), [1, 1, N_e]), 3);
        else
            EVpre = V_next;
        end
        EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);

        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, j);
        [a2primeIndex, a2primeProbs] = CreateRiskyAssetFnMatrix(vfoptions.aprimeFn, Dims.n_d23, n_a2, n_u, Grids.d23_grid, Grids.a2_grid, u_gridvals, aprimeFnParamsVec, 2);
        Risky_J.a2primeIndex = a2primeIndex;
        Risky_J.a2primeProbs = a2primeProbs;
    end

    for d_outer_c = 1:N_d_outer
        if N_d_outer > 1
            d_val = reshape(Grids.d12_gridvals(:, d_outer_c, :), [N_d13, size(Grids.d12_gridvals, 3)]);
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
            [DiscountedEV_d13, DiscountedEV_d3, d2index] = precalc_ev_risky_local(EVpre_flat, df, pi_bz, pi_u, Risky_J, Dims);
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
                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, spec_n_bz, spec_n_e, d_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d13, N_a1, level1n, N_a2, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0
                        RHS_L1 = RM_L1 + DiscountedEV_d13;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV_d13(:, :, :, :, z_idx, 1);
                    end
                    ev_flat_offset = (z_c - 1) * (N_d3 * N_a1 * N_z_slice);
                else
                    RHS_L1 = RM_L1;
                end

                % 1. Optimal a1prime conditional on d13 for monotonicity bounds
                [~, a1p_idx_L1] = max(RHS_L1, [], 2); % [N_d13, 1, level1n, N_a2, N_z_slice, N_e_slice]

                % 2. Full (d13, a1prime) optimum at Level 1 points
                [v_tmp_L1, maxindex1] = max(reshape(RHS_L1, [N_d13 * N_a1, level1n, N_a2, N_z_slice, N_e_slice]), [], 1);

                d13_L1 = rem(maxindex1 - 1, N_d13) + 1;
                a1p_L1 = ceil(maxindex1 / N_d13);
                d1_L1  = rem(d13_L1 - 1, N_d1) + 1;
                d3_L1  = ceil(d13_L1 / N_d1);

                if has_cont && N_d2 > 1
                    d2_L1 = d2index(d3_L1 + N_d3*(a1p_L1 - 1) + d2_bz_off + ev_flat_offset);
                else
                    d2_L1 = 1;
                end

                linear_L1 = d1_L1 + N_d1*(d2_L1 - 1) + (N_d1*N_d2)*(d3_L1 - 1) + (N_d1*N_d2*N_d3)*(d_outer_c - 1) + N_d*(a1p_L1 - 1);

                if N_d_outer == 1
                    V(level1ii, :, z_idx, e_idx, j)         = reshape(v_tmp_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    Policy(1, level1ii, :, z_idx, e_idx, j) = reshape(linear_L1, [1, level1n, N_a2, N_z_slice, N_e_slice]);
                else
                    V_fordouter(level1ii, :, z_idx, e_idx, d_outer_c) = reshape(v_tmp_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
                    P_fordouter(level1ii, :, z_idx, e_idx, d_outer_c) = reshape(linear_L1, [level1n, N_a2, N_z_slice, N_e_slice]);
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
                        loweredge = min(a1p_idx_L1(:, 1, ii, :, :, :), N_a1 - mg); % [N_d13, 1, 1, N_a2, N_z_slice, N_e_slice]
                        a1p_offsets = reshape(a1p_offsets_base(1:mg+1), [1, mg+1, 1, 1, 1, 1]);
                        a1p_idx = loweredge + a1p_offsets; % [N_d13, mg+1, 1, N_a2, N_z_slice, N_e_slice]
                    else
                        loweredge = a1p_idx_L1(:, 1, ii, :, :, :);
                        a1p_idx = loweredge;
                    end
                    a1p_grid_safe = reshape(a1_gridvals(a1p_idx), [N_d13, mg+1, 1, N_a2, N_z_slice, N_e_slice]);

                    RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a1, n_a2, spec_n_bz, spec_n_e, d_val, a1p_grid_safe, a1_gridvals(curraindex), a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                    RM_L2 = reshape(RM_L2, [N_d13, mg+1, L, N_a2, N_z_slice, N_e_slice]);

                    if has_cont
                        d3aprimez = d3_off_L2 + N_d3*(a1p_idx - 1) + (N_d3 * N_a1) * bz_off_L2 + ev_flat_offset;
                        RHS_L2 = RM_L2 + reshape(DiscountedEV_d3(d3aprimez(:)), [N_d13, mg+1, 1, N_a2, N_z_slice, N_e_slice]);
                    else
                        RHS_L2 = RM_L2;
                    end

                    [v_tmp_L2, maxindex2] = max(reshape(RHS_L2, [N_d13 * (mg+1), L, N_a2, N_z_slice, N_e_slice]), [], 1);

                    d13_L2     = rem(maxindex2 - 1, N_d13) + 1;
                    a1p_offset = ceil(maxindex2 / N_d13);
                    d1_L2      = rem(d13_L2 - 1, N_d1) + 1;
                    d3_L2      = ceil(d13_L2 / N_d1);

                    le_allind = d13_L2 + le_state_off;
                    a1p_abs   = loweredge(le_allind) + a1p_offset - 1;

                    if has_cont && N_d2 > 1
                        d2_L2 = d2index(d3_L2 + N_d3*(a1p_abs - 1) + d2_bz_off + ev_flat_offset);
                    else
                        d2_L2 = 1;
                    end

                    linear_L2 = d1_L2 + N_d1*(d2_L2 - 1) + (N_d1*N_d2)*(d3_L2 - 1) + (N_d1*N_d2*N_d3)*(d_outer_c - 1) + N_d*(a1p_abs - 1);

                    if N_d_outer == 1
                        V(curraindex, :, z_idx, e_idx, j)         = reshape(v_tmp_L2, [L, N_a2, N_z_slice, N_e_slice]);
                        Policy(1, curraindex, :, z_idx, e_idx, j) = reshape(linear_L2, [1, L, N_a2, N_z_slice, N_e_slice]);
                    else
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

function [DiscountedEV_d13, DiscountedEV_d3, d2index] = precalc_ev_risky_local(EVpre_flat, df, pi_bz, pi_u, Risky_J, Dims)
    N_a1 = Dims.N_a1; N_bothz = Dims.N_bothz;
    N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d3 = Dims.N_d3; N_d23 = Dims.N_d23; N_u = Dims.N_u;

    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat;
    EV_val(EVinf) = -1e250;
    EV_z = EV_val * pi_bz';
    EV_z((EVinf * (pi_bz' > 0)) > 0) = -Inf;

    a2p_idx   = Risky_J.a2primeIndex;
    a2p_probs = Risky_J.a2primeProbs;

    a1_base = repelem(gpuArray(1:1:N_a1)', N_d23, N_u);
    idx_lo  = a1_base + N_a1 * repmat(a2p_idx - 1, N_a1, 1);
    idx_hi  = a1_base + N_a1 * repmat(a2p_idx,     N_a1, 1);

    Vlower = reshape(EV_z(idx_lo(:), :), [N_d23 * N_a1, N_u, N_bothz]);
    Vupper = reshape(EV_z(idx_hi(:), :), [N_d23 * N_a1, N_u, N_bothz]);

    prob_u = repmat(reshape(a2p_probs, [N_d23, 1, N_u]), [1, N_a1, 1, N_bothz]);
    prob_u = reshape(prob_u, [N_d23 * N_a1, N_u, N_bothz]);
    prob_u(Vlower == Vupper) = 0;

    c_lo = prob_u .* Vlower;       c_lo(isnan(c_lo)) = 0;
    c_hi = (1 - prob_u) .* Vupper; c_hi(isnan(c_hi)) = 0;

    EV_integrated = sum((c_lo + c_hi) .* reshape(pi_u, [1, N_u, 1]), 2);
    [EV_onlyd3, d2index] = max(reshape(df * EV_integrated, [N_d2, N_d3 * N_a1, N_bothz]), [], 1);

    DiscountedEV_d3 = reshape(EV_onlyd3, [N_d3, N_a1, 1, 1, N_bothz, 1]);
    if N_d1 > 1
        DiscountedEV_d13 = repelem(DiscountedEV_d3, N_d1, 1, 1, 1, 1, 1);
    else
        DiscountedEV_d13 = DiscountedEV_d3;
    end
    d2index = reshape(d2index, [N_d3 * N_a1, N_bothz]);
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
