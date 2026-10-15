function [V, Policy] = ValueFnIter_UFHorz_RiskyAsset_GI2A_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_grid, z_gridvals_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d3 = Dims.N_d3;
N_d13 = Dims.N_d13; N_d_outer = Dims.N_d_outer;
has_z = Dims.has_z; has_semiz = Dims.has_semiz; has_e = Dims.has_e; has_bothz = Dims.has_bothz;
n_u = vfoptions.n_u;

N_z_iter = Dims.N_z_iter; N_z_slice = Dims.N_z_slice; z_mode = Dims.z_mode; spec_n_bz = Dims.spec_n_bz;
N_e_iter = Dims.N_e_iter; N_e_slice = Dims.N_e_slice; e_mode = Dims.e_mode; spec_n_e  = Dims.spec_n_e;

% RiskyAsset GI2A specific split: a1_c (interpolated standard asset), a2_c (discrete standard asset), exp (risky asset)
n_a1_c     = Dims.n_a1(1);
n_a2_c_vec = Dims.n_a1(2:end);
n_a2_c     = max(prod(n_a2_c_vec), 1);
n_exp      = Dims.n_a2;
N_exp      = Dims.N_a2;

a1_c_gridvals = CreateGridvals(n_a1_c, gpuArray(vfoptions.a1_grid(1:n_a1_c)), 1);
a2_c_gridvals = CreateGridvals(n_a2_c_vec, gpuArray(vfoptions.a1_grid(n_a1_c+1:end)), 1);
exp_gridvals  = Grids.a2_gridvals;
u_gridvals    = Grids.u_gridvals;
pi_u          = vfoptions.pi_u;
aprimeFnParamNames = Dims.aprimeFnParamNames;

% --- 2. Interpolation Setup ---
n2short = vfoptions.ngridinterp;
n2long  = n2short*2 + 3;
a1prime_grid_col = gpuArray(interp1(1:1:n_a1_c, gather(a1_c_gridvals), linspace(1, n_a1_c, n_a1_c+(n_a1_c-1)*n2short))');
a1prime_grid_col = reshape(a1prime_grid_col, [], 1);
N_a1fine = length(a1prime_grid_col);

Grids.a1_c_gridvals    = a1_c_gridvals;
Grids.a1prime_grid_col = a1prime_grid_col;
Dims.n_a1_c = n_a1_c; Dims.n_a2_c = n_a2_c; Dims.N_a1fine = N_a1fine;

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 3. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');

n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1, 1, 1, 1]);

d3_off_L2  = reshape(repelem(gpuArray(1:1:N_d3)', N_d1, 1), [N_d13, 1, 1, 1, 1, 1, 1, 1]);
a2p_off_L2 = reshape(gpuArray(0:1:n_a2_c-1),                [1, 1, n_a2_c, 1, 1, 1, 1, 1]);
bz_off_L2  = reshape(gpuArray(0:1:N_z_slice-1),             [1, 1, 1, 1, 1, 1, N_z_slice, 1]);

S1 = N_d13; S2 = S1 * n2long; S3 = S2 * n_a2_c;
S_mid2 = N_d13; S_mid3 = S_mid2 * n_a2_c;

state_lin_off = reshape(gpuArray(0:1:(N_a * N_z_slice * N_e_slice - 1)), [1, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);
d2_bz_off     = (N_d3 * N_a1) * reshape(gpuArray(0:1:N_z_slice-1), [1, 1, 1, 1, N_z_slice, 1]);

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
            EVpre = sum(V_next .* reshape(vfoptions.pi_e_J(:, j+1), [1, 1, N_e]), 3);
        else
            EVpre = V_next;
        end
        EVpre_flat = reshape(EVpre, [N_a1 * N_a2, N_bothz]);

        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, j);
        [a2primeIndex, a2primeProbs] = CreateRiskyAssetFnMatrix(vfoptions.aprimeFn, Dims.n_d23, n_exp, n_u, Grids.d23_grid, Grids.a2_grid, u_gridvals, aprimeFnParamsVec, 2);
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
            [DiscountedEV_d13, DiscountedEVinterp_d3, d2index] = precalc_ev_risky_local(EVpre_flat, df, pi_bz, pi_u, Risky_J, Dims, Grids);
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

                % --- Stage 1: Coarse Grid Evaluation over a1prime (dim 2) ---
                RM_L1 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, Dims.special_n_d, n_a1_c, n_a2_c, n_exp, spec_n_bz, spec_n_e, d_val, a1_c_gridvals, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d13, n_a1_c, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0
                        RHS_L1 = RM_L1 + DiscountedEV_d13;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV_d13(:, :, :, :, :, :, z_idx, 1);
                    end
                    ev_fine_offset = (z_c - 1) * (N_d3 * N_a1fine * n_a2_c * N_z_slice);
                    d2_offset_z    = (z_c - 1) * (N_d3 * N_a1 * N_z_slice);
                else
                    RHS_L1 = RM_L1;
                end

                % Maximize strictly over a1prime (dimension 2)
                [~, maxindex1] = max(RHS_L1, [], 2);
                midpoint = max(min(maxindex1, n_a1_c - 1), 2);
                a1primeindexes = (midpoint + (midpoint - 1)*n2short) + n2_offsets; % [N_d13, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]
                a1p_grid_safe  = reshape(a1prime_grid_col(a1primeindexes), [N_d13, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                % --- Stage 2: Fine Grid Evaluation over (d13, L2, a2prime) ---
                RM_L2 = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, Dims.special_n_d, n_a1_c, n_a2_c, n_exp, spec_n_bz, spec_n_e, d_val, a1p_grid_safe, a2_c_gridvals, a1_c_gridvals, a2_c_gridvals, exp_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d13, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);

                if has_cont
                    d3a1p_fine = d3_off_L2 + N_d3*(a1primeindexes - 1) + (N_d3 * N_a1fine)*a2p_off_L2 + (N_d3 * N_a1fine * n_a2_c)*bz_off_L2 + ev_fine_offset;
                    RHS_L2 = RM_L2 + reshape(DiscountedEVinterp_d3(d3a1p_fine(:)), [N_d13, n2long, n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]);
                else
                    RHS_L2 = RM_L2;
                end

                [v_tmp, p_tmp] = max(reshape(RHS_L2, [N_d13 * n2long * n_a2_c, n_a1_c, n_a2_c, N_exp, N_z_slice, N_e_slice]), [], 1);

                d13_ind        = rem(p_tmp - 1, N_d13) + 1;
                L2_and_a2p     = ceil(p_tmp / N_d13);
                L2offset       = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                d1_ind = rem(d13_ind - 1, N_d1) + 1;
                d3_ind = ceil(d13_ind / N_d1);

                % Boundary -Inf checks & midpoint extraction using unified state_lin_off
                linidx_lower = d13_ind + S2*(a2p_idx_chosen - 1) + S3*state_lin_off;
                linidx_upper = linidx_lower + S1*(n2long - 1);
                flag_val = 2 + ((L2offset >= 2 & L2offset <= n2short+1) & (RM_L2(linidx_lower) == -Inf)) ...
                             - ((L2offset >= n2short+3 & L2offset <= n2long-1) & (RM_L2(linidx_upper) == -Inf));

                midpt_idx    = d13_ind + S_mid2*(a2p_idx_chosen - 1) + S_mid3*state_lin_off;
                chosen_midpt = midpoint(midpt_idx);
                joint_a_idx  = chosen_midpt + n_a1_c * (a2p_idx_chosen - 1);

                if has_cont && N_d2 > 1
                    a12p_coarse = joint_a_idx + (flag_val == 3) - (flag_val == 1);
                    d2_ind = d2index(d3_ind + N_d3*(a12p_coarse - 1) + d2_bz_off + d2_offset_z);
                else
                    d2_ind = 1;
                end

                d_joint_ind = d1_ind + N_d1*(d2_ind - 1) + (N_d1*N_d2)*(d3_ind - 1) + (N_d1*N_d2*N_d3)*(d_outer_c - 1);

                if N_d_outer == 1
                    V(:, z_idx, e_idx, j)         = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    Policy(1, :, z_idx, e_idx, j) = reshape(d_joint_ind, [1, N_a, N_z_slice, N_e_slice]);
                    Policy(2, :, z_idx, e_idx, j) = reshape(joint_a_idx, [1, N_a, N_z_slice, N_e_slice]);
                    Policy(3, :, z_idx, e_idx, j) = reshape(L2offset, [1, N_a, N_z_slice, N_e_slice]);
                    Policy(4, :, z_idx, e_idx, j) = reshape(flag_val, [1, N_a, N_z_slice, N_e_slice]);
                else
                    V_fordouter(:, z_idx, e_idx, d_outer_c)  = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    P1_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(d_joint_ind, [N_a, N_z_slice, N_e_slice]);
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

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function [DiscountedEV_d13, DiscountedEVinterp_d3, d2index] = precalc_ev_risky_local(EVpre_flat, df, pi_bz, pi_u, Risky_J, Dims, Grids)
    N_a1 = Dims.N_a1; N_bothz = Dims.N_bothz; N_a1fine = Dims.N_a1fine;
    N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d3 = Dims.N_d3; N_d23 = Dims.N_d23; N_u = Dims.N_u;
    n_a1_c = Dims.n_a1_c; n_a2_c = Dims.n_a2_c;
    a1_c_gridvals = Grids.a1_c_gridvals; a1prime_grid_col = Grids.a1prime_grid_col;

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

    DiscountedEV_d3 = reshape(EV_onlyd3, [N_d3, n_a1_c, n_a2_c, N_bothz]);
    DiscountedEVinterp_perm = interp1(a1_c_gridvals, permute(DiscountedEV_d3, [2, 1, 3, 4]), a1prime_grid_col);
    DiscountedEVinterp_d3   = reshape(permute(DiscountedEVinterp_perm, [2, 1, 3, 4]), [N_d3, N_a1fine, n_a2_c, N_bothz]);

    DiscountedEV_d3 = reshape(DiscountedEV_d3, [N_d3, n_a1_c, n_a2_c, 1, 1, 1, N_bothz, 1]);
    if N_d1 > 1
        DiscountedEV_d13 = repelem(DiscountedEV_d3, N_d1, 1, 1, 1, 1, 1, 1, 1);
    else
        DiscountedEV_d13 = DiscountedEV_d3;
    end
    d2index = reshape(d2index, [N_d3 * N_a1, N_bothz]);
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
