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

% 5. EXPECTED VALUE ADDITION
if ~isempty(DiscountedEV_slice)
    R_slice = R_slice + DiscountedEV_slice;
end

% 6. REDUCE
[v_max, v_idx] = max(R_slice, [], 1);

% FORCE GARBAGE COLLECTION
clear R_slice ReturnFn z_val DiscountedEV_slice;

end
