function [V, Policy] = ValueFnIter_UFHorz_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a     = Dims.N_a;
N_bothz = Dims.N_bothz;
N_e     = Dims.N_e;
N_d1    = Dims.N_d1;
N_d2    = Dims.N_d2;
N_z     = Dims.N_z;
N_semiz = Dims.N_semiz;

has_d1    = Dims.has_d1;
has_d2    = Dims.has_d2;
has_semiz = Dims.N_semiz_raw > 0;
has_e     = Dims.N_e_raw > 0;
has_z     = Dims.special_n_z(1) > 0;
has_bothz = has_semiz || has_z;

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

% --- 3. Grid Extracts & Interpolation Setup ---
a_gridvals = gpuArray(a_grid);
n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
aprime_grid = gpuArray(interp1(1:1:N_a, gather(a_gridvals), linspace(1, N_a, N_a+(N_a-1)*n2short))');
n2aprime = length(aprime_grid);

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

% --- 4. Preallocate Outputs & Persistent Slice Offsets ---
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');

n2_offsets = reshape(gpuArray(-n2short-1:1:n2short+1), [1, n2long, 1, 1, 1]);

aind_SL  = reshape(gpuArray(0:1:N_a-1),       [1, N_a, 1, 1]);
bzind_SL = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, N_z_slice, 1]);
eind_SL  = reshape(gpuArray(0:1:N_e_slice-1), [1, 1, 1, N_e_slice]);
state_lin_off = aind_SL + N_a * bzind_SL + N_a * N_z_slice * eind_SL;

bz_off_L2 = reshape(gpuArray(0:1:N_z_slice-1), [1, 1, 1, N_z_slice, 1]);

if N_d2 > 1
    V_ford2  = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
    P1_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
    P2_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
    P3_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
    P4_ford2 = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
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
            EVpre = sum(V_next .* reshape(Probs.pi_e_J(:, j+1), [1, 1, N_e]), 3);
        else
            EVpre = V_next;
        end
        EVpre_flat = reshape(EVpre, [N_a, N_bothz]);
    end

    for d2_c = 1:N_d2
        if has_d2 && has_d1
            d_val = Grids.d12_gridvals(:, :, d2_c);
        else
            d_val = Grids.d12_gridvals;
        end

        if has_cont
            if has_bothz
                if has_semiz
                    pi_bz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                else
                    pi_bz = Probs.pi_z_J(:,:,j);
                end
            else
                pi_bz = 1;
            end
            [DiscountedEV, DiscountedEVinterp] = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz, a_gridvals, aprime_grid);
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

                % --- Stage 1: Coarse Grid Evaluation ---
                RM_L1 = CreateReturnFnMatrix_UDisc(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, Dims.special_n_e, d_val, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 1);
                RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, N_z_slice, N_e_slice]);

                if has_cont
                    if z_mode == 0
                        RHS_L1 = RM_L1 + DiscountedEV;
                    else
                        RHS_L1 = RM_L1 + DiscountedEV(:, :, :, z_idx, 1);
                    end
                    ev_flat_offset = (z_c - 1) * (n2aprime * N_z_slice);
                else
                    RHS_L1 = RM_L1;
                end

                [~, maxindex1] = max(RHS_L1, [], 2);
                midpoint = max(min(maxindex1, N_a - 1), 2);
                aprimeindexes = (midpoint + (midpoint - 1)*n2short) + n2_offsets; % [N_d1, n2long, N_a, N_z_slice, N_e_slice]
                a1p_grid_safe = reshape(aprime_grid(aprimeindexes), [N_d1, n2long, N_a, N_z_slice, N_e_slice]);

                % --- Stage 2: Fine Grid Evaluation ---
                RM_L2 = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, Dims.special_n_d, n_a, spec_n_bz, Dims.special_n_e, d_val, a1p_grid_safe, a_gridvals, bz_val, e_val, ReturnFnParamsVec, 2);
                RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, N_z_slice, N_e_slice]);

                if has_cont
                    aprimez = aprimeindexes + n2aprime * bz_off_L2 + ev_flat_offset;
                    RHS_L2 = RM_L2 + reshape(DiscountedEVinterp(aprimez(:)), [N_d1 * n2long, N_a, N_z_slice, N_e_slice]);
                else
                    RHS_L2 = RM_L2;
                end

                [Vtempii, maxindexL2] = max(RHS_L2, [], 1);

                d1_ind   = rem(maxindexL2 - 1, N_d1) + 1;
                L2offset = ceil(maxindexL2 / N_d1);

                linidx_lower = d1_ind + N_d1 * n2long * state_lin_off;
                linidx_upper = linidx_lower + N_d1 * (n2long - 1);
                flag_val = 2 + ((L2offset >= 2 & L2offset <= n2short+1) & (RM_L2(linidx_lower) == -Inf)) ...
                             - ((L2offset >= n2short+3 & L2offset <= n2long-1) & (RM_L2(linidx_upper) == -Inf));

                allind     = d1_ind + N_d1 * state_lin_off;
                chosen_mid = midpoint(allind);

                if N_d2 == 1
                    V(:, z_idx, e_idx, j)         = reshape(Vtempii, [N_a, N_z_slice, N_e_slice]);
                    Policy(1, :, z_idx, e_idx, j) = d1_ind;
                    Policy(2, :, z_idx, e_idx, j) = chosen_mid;
                    Policy(3, :, z_idx, e_idx, j) = L2offset;
                    Policy(4, :, z_idx, e_idx, j) = flag_val;
                else
                    V_ford2(:, z_idx, e_idx, d2_c)  = reshape(Vtempii, [N_a, N_z_slice, N_e_slice]);
                    P1_ford2(:, z_idx, e_idx, d2_c) = reshape(d1_ind + N_d1*(d2_c - 1), [N_a, N_z_slice, N_e_slice]);
                    P2_ford2(:, z_idx, e_idx, d2_c) = reshape(chosen_mid, [N_a, N_z_slice, N_e_slice]);
                    P3_ford2(:, z_idx, e_idx, d2_c) = reshape(L2offset, [N_a, N_z_slice, N_e_slice]);
                    P4_ford2(:, z_idx, e_idx, d2_c) = reshape(flag_val, [N_a, N_z_slice, N_e_slice]);
                end
            end
        end
    end

    if N_d2 > 1
        [V(:,:,:,j), Policy(:,:,:,:,j)] = extract_pol_local(V_ford2, P1_ford2, P2_ford2, P3_ford2, P4_ford2, N_a, N_bothz, N_e);
    end
end

end % <--- MAIN FUNCTION ENDS HERE!

% =========================================================================
% LOCAL SUBFUNCTIONS (Small Array Outputs Only)
% =========================================================================

function [DiscountedEV, DiscountedEVinterp] = precalc_ev_local(EVpre_flat, df, pi_bz, N_a, N_bothz, a_gridvals, aprime_grid)
    EVinf = (EVpre_flat == -Inf);
    EV_val = EVpre_flat;
    EV_val(EVinf) = -1e250;
    EV_val = EV_val * pi_bz';
    EV_val(EVinf * (pi_bz' > 0) > 0) = -Inf; % Exact -Inf restoration

    DiscountedEV_2D    = df * EV_val; % [N_a, N_bothz]
    DiscountedEVinterp = interp1(a_gridvals, DiscountedEV_2D, aprime_grid); % [n2aprime, N_bothz]
    DiscountedEV       = reshape(DiscountedEV_2D, [1, N_a, 1, N_bothz, 1]);
end

function [V_j, Pol_j] = extract_pol_local(V_ford2, P1, P2, P3, P4, N_a, N_bothz, N_e)
    [V_j, max_d2] = max(V_ford2, [], 4);
    lin_idx = gpuArray(1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (max_d2(:) - 1);

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
