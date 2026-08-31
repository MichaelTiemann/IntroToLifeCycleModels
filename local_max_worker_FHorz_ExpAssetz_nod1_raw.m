function [v_max, v_idx] = local_max_worker_FHorz_ExpAssetz_nod1_raw(ReturnFn_str, n_d2, n_a1, n_a2, d2_gridvals, a1_gridvals, a2_gridvals, z_slice_aligned, DiscountedEV_slice, ReturnFnParamNames, ReturnFnParamsVec, vfoptions)

% 1. PURE HANDLE RECONSTRUCTION
% Because the function is self-contained, str2func works perfectly.
ReturnFn = str2func(ReturnFn_str);

% 2. INJECT VFOPTIONS (The Obi-Wan Trick)
% Forces your custom arrayfun.m to stay on the vectorized fast path.
vfoptions_worker = vfoptions;
vfoptions_worker.vectorizedarrayfunnames = {'FORCE_VECTORIZED'};
vfoptions_worker.n_proc = 1;
vfoptions_worker.parallel = 1;
assignin('base', 'vfoptions', vfoptions_worker);

% 3. DYNAMIC Z-DIMENSION RESTORATION
z_val = shiftdim(z_slice_aligned, 2);

% Adapt n_a2 to match the incoming a2_gridvals slice
n_a2_slice = ones(1, size(a2_gridvals, 2), vfoptions.precision);
n_a2_slice(1) = size(a2_gridvals, 1);

% Adapt n_z to match the incoming z_val slice
n_z_slice = ones(1, size(z_val, 2), vfoptions.precision);
n_z_slice(1) = size(z_val, 1);

N_d2 = prod(n_d2);
N_a1 = prod(n_a1);
N_a2_slice = prod(n_a2_slice);
N_z_slice = prod(n_z_slice);

if vfoptions.lowmemory == 1
    % --- LOWMEMORY == 1: ITERATE ON z VALUES (NATIVE VFI CONVENTION) ---
    special_n_z = ones(1, length(n_z_slice), vfoptions.precision);
    special_n_z(1) = 1;

    v_max = zeros(1, N_a1 * N_a2_slice, N_z_slice, vfoptions.precision);
    v_idx = zeros(1, N_a1 * N_a2_slice, N_z_slice);

    for z_val_counter = 1:N_z_slice
        z_sub = z_val(z_val_counter, :);

        R_sub = CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1, ...
            n_a2_slice, special_n_z, d2_gridvals, a1_gridvals, a1_gridvals, ...
            a2_gridvals, z_sub, ReturnFnParamsVec, 0, 0);

        if ~isempty(DiscountedEV_slice)
            EV_sub = DiscountedEV_slice(:, :, z_val_counter);
            EV_expanded = repelem(EV_sub, 1, N_a1, 1);
            R_sub = R_sub + EV_expanded;
        end

        [v_max(1, :, z_val_counter), v_idx(1, :, z_val_counter)] = max(R_sub, [], 1);
    end

elseif vfoptions.lowmemory == 5
    % --- LOWMEMORY == 5: ITERATE ON a2 VALUES (CUSTOM ExpAsset CONVENTION) ---
    special_n_a2 = ones(1, length(n_a2_slice), vfoptions.precision);
    special_n_a2(1) = 1;

    v_max = zeros(1, N_a1 * N_a2_slice, N_z_slice, vfoptions.precision);
    v_idx = zeros(1, N_a1 * N_a2_slice, N_z_slice);

    for a2_val_counter = 1:N_a2_slice
        a2_sub = a2_gridvals(a2_val_counter, :);

        R_sub = CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1, ...
            special_n_a2, n_z_slice, d2_gridvals, a1_gridvals, a1_gridvals, ...
            a2_sub, z_val, ReturnFnParamsVec, 0, 0);

        if ~isempty(DiscountedEV_slice)
            EV_sub = DiscountedEV_slice(:, a2_val_counter, :);
            EV_expanded = repelem(EV_sub, 1, N_a1, 1);
            R_sub = R_sub + EV_expanded;
        end

        % Slice insertion along flattened a1*a2 dimension:
        % a1 cycles fast, so the a2_val_counter block occupies a contiguous slice of length N_a1
        start_col = (a2_val_counter - 1) * N_a1 + 1;
        end_col   = a2_val_counter * N_a1;

        [v_max(1, start_col:end_col, :), v_idx(1, start_col:end_col, :)] = max(R_sub, [], 1);
    end
else
    % --- LOWMEMORY == 0: FULLY VECTORIZED CHUNK ---
    R_slice = CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1, ...
        n_a2_slice, n_z_slice, d2_gridvals, a1_gridvals, a1_gridvals, ...
        a2_gridvals, z_val, ReturnFnParamsVec, 0, 0);

    if ~isempty(DiscountedEV_slice)
        EV_expanded = repelem(DiscountedEV_slice, 1, N_a1, 1);
        R_slice = R_slice + EV_expanded;
    end

    [v_max, v_idx] = max(R_slice, [], 1);
end


end
