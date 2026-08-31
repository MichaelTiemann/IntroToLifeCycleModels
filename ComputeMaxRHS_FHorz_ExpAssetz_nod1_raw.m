function [V_out, Policy_out] = ComputeMaxRHS_FHorz_ExpAssetz_nod1_raw(ReturnFn, ...
    n_d2, n_a1, n_a2, d2_gridvals, a1_gridvals, a2_gridvals, z_gridvals, ...
    DiscountedEV, ReturnFnParamNames, ReturnFnParamsVec, vfoptions)

ReturnFn_str = func2str(ReturnFn);
z_grid_aligned = shiftdim(z_gridvals, -2);
is_vectorized = true;

[V_out, Policy_out] = pararrayfun_vfi('local_max_worker_FHorz_ExpAssetz_nod1_raw', is_vectorized, ...
    ReturnFn_str, n_d2, n_a1, n_a2, d2_gridvals, a1_gridvals, a2_gridvals, ...
    z_grid_aligned, DiscountedEV, ReturnFnParamNames, ReturnFnParamsVec, vfoptions);


end
