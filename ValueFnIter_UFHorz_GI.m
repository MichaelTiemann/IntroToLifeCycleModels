function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

if isscalar(n_a)
    [VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a) >= 2
    [VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

[V, Policy, Valt, Policyalt] = UnKronUFHorz(VKron, PolicyKron, ValtKron, PolicyaltKron, n_d, n_a, n_a, n_z, N_j, vfoptions, 1);

end
