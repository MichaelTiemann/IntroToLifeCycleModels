function [V_out, Policy_out] = ComputeMaxRHS_FHorz_ExpAssetz_nod1_raw(n_proc, ReturnFn, n_d2, n_a1, n_a2, d2_gridvals, a1_gridvals, a2_gridvals, z_gridvals, DiscountedEV, ReturnFnParamNames, ReturnFnParamsVec)

% Flatten the self-contained anonymous function to a pure string
ReturnFn_str = func2str(ReturnFn);

% Align the dimension we want to slice
z_grid_aligned = shiftdim(z_gridvals, -2);
is_vectorized = true;

% Dispatch to the workers
[V_out, Policy_out] = pararrayfun_vfi(n_proc, 'local_max_worker_FHorz_ExpAssetz_nod1_raw', is_vectorized, ...
    ReturnFn_str, n_d2, n_a1, n_a2, d2_gridvals, a1_gridvals, a2_gridvals, z_grid_aligned, DiscountedEV, ReturnFnParamNames, ReturnFnParamsVec);

end
