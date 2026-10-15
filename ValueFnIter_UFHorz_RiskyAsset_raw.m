function [V, Policy] = ValueFnIter_UFHorz_RiskyAsset_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_grid, z_gridvals_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d3 = Dims.N_d3; N_d4 = Dims.N_d4;
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

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 2. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');

aind_SL  = reshape(gpuArray(0:1:N_a-1),       [1, N_a, 1, 1]);
bzind_SL = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, N_z_slice, 1]);
eind_SL  = reshape(gpuArray(0:1:N_e_slice-1), [1, 1, 1, N_e_slice]);

d1_state_off = (N_d3 * N_a1) * (aind_SL + N_a * bzind_SL + (N_a * N_z_slice) * eind_SL);
d2_bz_off    = (N_d3 * N_a1) * bzind_SL;

if N_d_outer > 1
    V_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
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
            [DiscountedEV_onlyd3, d2index] = precalc_ev_risky_local(EVpre_flat, df, pi_bz, pi_u, Risky_J, Dims);
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

                % 1. Evaluate ReturnFn over (d1, d3, d4_c) and a1prime
                RM = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, Dims.special_n_d, n_a1, n_a2, spec_n_bz, spec_n_e, d_val, a1_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);

                % 2. Refine out d1 immediately
                if N_d1 > 1
                    [RM_onlyd3, d1index] = max(reshape(RM, [N_d1, N_d3 * N_a1, N_a, N_z_slice, N_e_slice]), [], 1);
                    RM_onlyd3 = reshape(RM_onlyd3, [N_d3 * N_a1, N_a, N_z_slice, N_e_slice]);
                else
                    RM_onlyd3 = reshape(RM, [N_d3 * N_a1, N_a, N_z_slice, N_e_slice]);
                end

                % 3. Add Continuation Value (already refined over d2)
                if has_cont
                    if z_mode == 0
                        entireRHS = RM_onlyd3 + DiscountedEV_onlyd3;
                    else
                        entireRHS = RM_onlyd3 + DiscountedEV_onlyd3(:, :, z_idx, 1);
                    end
                    d2_offset_z = (z_c - 1) * (N_d3 * N_a1 * N_z_slice);
                else
                    entireRHS = RM_onlyd3;
                end

                % 4. Maximize over (d3, a1prime)
                [v_tmp, d3a1p_ind] = max(entireRHS, [], 1); % [1, N_a, N_z_slice, N_e_slice]

                d3_ind  = rem(d3a1p_ind - 1, N_d3) + 1;
                a1p_ind = ceil(d3a1p_ind / N_d3);

                if N_d1 > 1
                    d1_ind = d1index(d3a1p_ind + d1_state_off);
                else
                    d1_ind = 1;
                end

                if has_cont && N_d2 > 1
                    d2_ind = d2index(d3a1p_ind + d2_bz_off + d2_offset_z);
                else
                    d2_ind = 1;
                end

                % Combine (d1, d2, d3, d4_c, a1prime) into standard 1-row Kronecker index
                linear_p = d1_ind + N_d1*(d2_ind - 1) + (N_d1*N_d2)*(d3_ind - 1) + (N_d1*N_d2*N_d3)*(d_outer_c - 1) + N_d*(a1p_ind - 1);

                if N_d_outer == 1
                    V(:, z_idx, e_idx, j)         = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    Policy(1, :, z_idx, e_idx, j) = reshape(linear_p, [1, N_a, N_z_slice, N_e_slice]);
                else
                    V_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(v_tmp, [N_a, N_z_slice, N_e_slice]);
                    P_fordouter(:, z_idx, e_idx, d_outer_c) = reshape(linear_p, [N_a, N_z_slice, N_e_slice]);
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

function [DiscountedEV_onlyd3, d2index] = precalc_ev_risky_local(EVpre_flat, df, pi_bz, pi_u, Risky_J, Dims)
    N_a1 = Dims.N_a1; N_bothz = Dims.N_bothz;
    N_d2 = Dims.N_d2; N_d3 = Dims.N_d3; N_d23 = Dims.N_d23; N_u = Dims.N_u;

    % 1. Conditional expectation over next-period z/semiz with exact -Inf safeguard
    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat;
    EV_val(EVinf) = -1e250;
    EV_z = EV_val * pi_bz';
    EV_z((EVinf * (pi_bz' > 0)) > 0) = -Inf; % [N_a, N_bothz]

    % 2. Interpolate risky asset a2' across (d2, d3, u)
    a2p_idx   = Risky_J.a2primeIndex; % [N_d23, N_u]
    a2p_probs = Risky_J.a2primeProbs; % [N_d23, N_u]

    a1_base = repelem(gpuArray(1:1:N_a1)', N_d23, N_u);
    idx_lo  = a1_base + N_a1 * repmat(a2p_idx - 1, N_a1, 1); % [N_d23 * N_a1, N_u]
    idx_hi  = a1_base + N_a1 * repmat(a2p_idx,     N_a1, 1);

    Vlower = reshape(EV_z(idx_lo(:), :), [N_d23 * N_a1, N_u, N_bothz]);
    Vupper = reshape(EV_z(idx_hi(:), :), [N_d23 * N_a1, N_u, N_bothz]);

    prob_u = repmat(reshape(a2p_probs, [N_d23, 1, N_u]), [1, N_a1, 1, N_bothz]);
    prob_u = reshape(prob_u, [N_d23 * N_a1, N_u, N_bothz]);
    prob_u(Vlower == Vupper) = 0;

    c_lo = prob_u .* Vlower;       c_lo(isnan(c_lo)) = 0;
    c_hi = (1 - prob_u) .* Vupper; c_hi(isnan(c_hi)) = 0;

    % 3. Integrate over u shock and refine out d2
    EV_integrated = sum((c_lo + c_hi) .* reshape(pi_u, [1, N_u, 1]), 2); % [N_d23 * N_a1, 1, N_bothz]
    [EV_onlyd3, d2index] = max(reshape(df * EV_integrated, [N_d2, N_d3 * N_a1, N_bothz]), [], 1);

    DiscountedEV_onlyd3 = reshape(EV_onlyd3, [N_d3 * N_a1, 1, N_bothz, 1]);
    d2index             = reshape(d2index,   [N_d3 * N_a1, N_bothz]);
end

function [V_j, Pol_j] = extract_pol_local(V_fordouter, P1, N_a, N_bothz, N_e)
    [V_j, max_d_out] = max(V_fordouter, [], 4);
    lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d_out(:) - 1);
    Pol_j = reshape(P1(lin_idx), [1, N_a, N_bothz, N_e]);
end

function slice = get_grid_slice_local(grid_J, idx, j, has_var)
    if ~has_var || isempty(grid_J); slice = []; return; end
    if strcmp(idx, 'all')
        if ndims(grid_J) >= 3; slice = grid_J(:, :, j); else; slice = grid_J; end
    else
        if ndims(grid_J) >= 3; slice = grid_J(idx, :, j); else; slice = grid_J(idx, :); end
    end
end
