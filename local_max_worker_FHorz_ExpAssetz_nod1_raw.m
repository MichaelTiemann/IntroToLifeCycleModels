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

% Adapt n_a2 to match the incoming a2_gridvals slice
n_a2_slice = ones(1, size(a2_gridvals, 2));
n_a2_slice(1) = size(a2_gridvals, 1);

% Adapt n_z to match the incoming z_val slice
n_z_slice = ones(1, size(z_val, 2));
n_z_slice(1) = size(z_val, 1);

% 4. NATIVE TOOLKIT CALL
R_slice = CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, n_a1, ...
        n_a2_slice, n_z_slice, d2_gridvals, a1_gridvals, a1_gridvals, ...
        a2_gridvals, z_val, ReturnFnParamsVec, 0, 0);

% 5. EXPECTED VALUE ADDITION (Robust Skinny Broadcasting)
if ~isempty(DiscountedEV_slice)
    N_a1_total = prod(n_a1);
    % Because a1 cycles fast, repeating by N_a1 flawlessly aligns
    % the EV array to R_slice regardless of whether a2 or z was sliced!
    EV_expanded = repelem(DiscountedEV_slice, 1, N_a1_total, 1);
    R_slice = R_slice + EV_expanded;
end

% 6. REDUCE
[v_max, v_idx] = max(R_slice, [], 1);

% FORCE GARBAGE COLLECTION
clear R_slice ReturnFn z_val DiscountedEV_slice;

end
