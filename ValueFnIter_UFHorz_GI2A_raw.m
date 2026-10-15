function [V, Policy] = ValueFnIter_UFHorz_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, vfoptions);

N_a1 = Dims.N_a1; N_a2 = Dims.N_a2; N_a = Dims.N_a;
N_bothz = Dims.N_bothz; N_e = Dims.N_e; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_d_inner = Dims.N_d_inner; N_d_outer = Dims.N_d_outer;
has_z = Dims.has_z; has_semiz = Dims.has_semiz; has_e = Dims.has_e; has_bothz = Dims.has_bothz;
n_a1 = Dims.n_a1; n_a2 = Dims.n_a2; n_e = Dims.n_e; n_semiz = Dims.n_semiz; n_d2 = Dims.n_d2;

N_z_iter = Dims.N_z_iter; N_z_slice = Dims.N_z_slice; z_mode = Dims.z_mode; spec_n_bz = Dims.spec_n_bz;
N_e_iter = Dims.N_e_iter; N_e_slice = Dims.N_e_slice; e_mode = Dims.e_mode; spec_n_e  = Dims.spec_n_e;

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

% --- 3. Grid Extracts & Interpolation Setup ---
a1_gridvals = gpuArray(a_grid(1:N_a1));
a2_gridvals = CreateGridvals(n_a2, gpuArray(a_grid(N_a1+1:end)), 1);

n2short = vfoptions.ngridinterp;
n2long  = n2short*2 + 3;
a1prime_grid = gpuArray(interp1(1:1:N_a1, gather(a1_gridvals), linspace(1, N_a1, N_a1+(N_a1-1)*n2short))');
N_a1fine = length(a1prime_grid);

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 4. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');

n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1, 1, 1]);

a2p_off_L2 = reshape(gpuArray(0:1:N_a2-1),      [1, 1, N_a2, 1, 1, 1, 1]);
bz_off_L2  = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, 1, 1, 1, N_z_slice, 1]);

S1 = N_d_inner; S2 = S1 * n2long; S3 = S2 * N_a2;
S_mid1 = N_d_inner; S_mid2 = S_mid1 * N_a2;

state_lin_off = reshape(gpuArray(0:1:(N_a * N_z_slice * N_e_slice - 1)), [1, N_a1, N_a2, N_z_slice, N_e_slice]);

if N_d_outer > 1
    V_fordouter  = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P1_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P2_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P3_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
    P4_fordouter = zeros(N_a, N_bothz, N_e, N_d_outer, 'gpuArray');
end

% --- 5. Backward Induction Loop (All 6 Strategies Inlined for Buffer Reuse) ---
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
                    pi_bz = kron(pi_z_J(:,:,j), vfoptions.pi_semiz_J(:,:,d_outer_c,j));
                else
                    pi_bz = pi_z_J(:,:,j);
                end
            else
                pi_bz = 1;
            end
            [DiscountedEV, DiscountedEVinterp] = precalc_ev_local(EVpre_flat, df, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid);
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
                RM_L1 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, spec_n_bz, spec_n_e, d_val, a1_gridvals, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 1, 0);
                RM_L1 = reshape(RM_L1, [N_d_inner, N_a1, N_a2, N_a1, N_a2, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0
                        RHS_L1 = RM_L1 + DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV(:, :, :, :, :, z_idx, 1);
                    end
                    ev_flat_offset = (z_c - 1) * (N_a1fine * N_a2 * N_z_slice);
                else
                    RHS_L1 = RM_L1;
                end

                % Maximize strictly over a1prime (dimension 2)
                [~, maxindex1] = max(RHS_L1, [], 2);
                midpoint = max(min(maxindex1, N_a1 - 1), 2);
                a1primeindexes = (midpoint + (midpoint - 1)*n2short) + n2_offsets; % [N_d_inner, n2long, N_a2, N_a1, N_a2, N_z_slice, N_e_slice]
                a1p_grid_safe  = reshape(a1prime_grid(a1primeindexes), [N_d_inner, n2long, N_a2, N_a1, N_a2, N_z_slice, N_e_slice]);

                % --- Stage 2: Fine Grid Evaluation over (d_inner, L2, a2prime) ---
                RM_L2 = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, Dims.special_n_d, spec_n_bz, spec_n_e, d_val, a1p_grid_safe, a2_gridvals, a1_gridvals, a2_gridvals, bz_val, e_val, ReturnFnParamsVec, 2, 0);
                RM_L2 = reshape(RM_L2, [N_d_inner, n2long, N_a2, N_a1, N_a2, N_z_slice, N_e_slice]);

                if has_cont
                    EV_idx = a1primeindexes + N_a1fine * a2p_off_L2 + N_a1fine * N_a2 * bz_off_L2 + ev_flat_offset;
                    RHS_L2 = RM_L2 + DiscountedEVinterp(EV_idx);
                else
                    RHS_L2 = RM_L2;
                end

                [v_tmp, p_tmp] = max(reshape(RHS_L2, [N_d_inner * n2long * N_a2, N_a1, N_a2, N_z_slice, N_e_slice]), [], 1);

                d_ind          = rem(p_tmp - 1, N_d_inner) + 1;
                L2_and_a2p     = ceil(p_tmp / N_d_inner);
                L2offset       = rem(L2_and_a2p - 1, n2long) + 1;
                a2p_idx_chosen = ceil(L2_and_a2p / n2long);

                % Boundary -Inf checks & midpoint extraction using unified state_lin_off
                linidx_lower = d_ind + S2*(a2p_idx_chosen - 1) + S3*state_lin_off;
                linidx_upper = linidx_lower + S1*(n2long - 1);
                flag_val = 2 + ((L2offset >= 2 & L2offset <= n2short+1) & (RM_L2(linidx_lower) == -Inf)) ...
                             - ((L2offset >= n2short+3 & L2offset <= n2long-1) & (RM_L2(linidx_upper) == -Inf));

                midpt_idx    = d_ind + S_mid1*(a2p_idx_chosen - 1) + S_mid2*state_lin_off;
                chosen_midpt = midpoint(midpt_idx);
                joint_a_idx  = chosen_midpt + N_a1 * (a2p_idx_chosen - 1);

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

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function [DiscountedEV, DiscountedEVinterp] = precalc_ev_local(EVpre_flat, df, pi_bz, N_a1, N_a2, N_bothz, a1_gridvals, a1prime_grid)
    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat;
    EV_val(EVinf) = -1e250;
    EV_expected = EV_val * pi_bz';
    EV_expected((EVinf * (pi_bz' > 0)) > 0) = -Inf;

    DiscountedEV_3D    = df * reshape(EV_expected, [N_a1, N_a2, N_bothz]);
    DiscountedEVinterp = interp1(a1_gridvals, DiscountedEV_3D, a1prime_grid); % [N_a1fine, N_a2, N_bothz]
    DiscountedEV       = reshape(DiscountedEV_3D, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
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
