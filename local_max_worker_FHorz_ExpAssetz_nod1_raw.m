function [v_max, v_idx] = local_max_worker_FHorz_ExpAssetz_nod1_raw(ReturnFn_str, n_d2, n_a1, n_a2, d2_gridvals, a1_gridvals, a2_gridvals, z_slice_aligned, DiscountedEV_slice, ReturnFnParamNames, ReturnFnParamsVec)

% 1. PURE HANDLE RECONSTRUCTION
% Because the function is self-contained, str2func works perfectly.
ReturnFn = str2func(ReturnFn_str);

% 2. INJECT VFOPTIONS (The Obi-Wan Trick)
% Forces your custom arrayfun.m to stay on the vectorized fast path.
vfoptions_worker = struct();
vfoptions_worker.vectorizedarrayfunnames = {'FORCE_VECTORIZED'};
vfoptions_worker.n_proc = 1;
vfoptions_worker.parallel = 1;
assignin('base', 'vfoptions', vfoptions_worker);

% 3. DYNAMIC Z-DIMENSION RESTORATION
z_val = shiftdim(z_slice_aligned, 2);
K = size(z_val, 1);
num_z_vars = size(z_val, 2);

% Creates an n_z_slice vector matching the number of z-variables
% so VFIToolkit's variadic router passes the correct number of arguments!
n_z_slice = ones(1, num_z_vars);
n_z_slice(1) = K;

% 4. NATIVE TOOLKIT CALL
R_slice = CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1, n_a2, n_z_slice, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec, 0, 0);

% 5. EXPECTED VALUE ADDITION (Robust Skinny Broadcasting)
if ~isempty(DiscountedEV_slice)

    sz_R = size(R_slice);
    dim1 = sz_R(1);

    % Get the exact sub-dimensions
    N_a1_total = prod(n_a1);
    N_a2_total = prod(n_a2);

    % Dynamically determine N_z_local based on actual dimensions of R_slice
    if length(sz_R) >= 3
        N_z_local = sz_R(3);
    else
        N_z_local = 1;
    end

    % 1. Reshape R_slice to split the flattened [N_a1 * N_a2] into [N_a1, N_a2]
    if N_z_local > 1
        R_slice = reshape(R_slice, [dim1, N_a1_total, N_a2_total, N_z_local]);
        EV_aligned = reshape(DiscountedEV_slice, [dim1, 1, N_a2_total, N_z_local]);
    else
        R_slice = reshape(R_slice, [dim1, N_a1_total, N_a2_total]);
        EV_aligned = reshape(DiscountedEV_slice, [dim1, 1, N_a2_total]);
    end

    % 2. Broadcast Add!
    R_slice = R_slice + EV_aligned;

    % 3. Flatten R_slice back so max() works correctly
    if N_z_local > 1
        R_slice = reshape(R_slice, [dim1, N_a1_total * N_a2_total, N_z_local]);
    else
        R_slice = reshape(R_slice, [dim1, N_a1_total * N_a2_total]);
    end
end

% 6. REDUCE
[v_max, v_idx] = max(R_slice, [], 1);

% FORCE GARBAGE COLLECTION
clear R_slice ReturnFn z_val DiscountedEV_slice;

end
