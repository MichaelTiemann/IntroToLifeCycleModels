function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_RiskyAsset_GI(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_a1 = vfoptions.n_a1;
if sum(n_a1) == 0
    error('Cannot use vfoptions.gridinterplayer=1 with riskyasset when there is no standard endogenous state (n_a1==0).');
end

if isscalar(n_a1)
    [VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_RiskyAsset_GI1_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a1) == 2
    [VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_RiskyAsset_GI2A_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    error('Cannot use vfoptions.gridinterplayer with more than two standard endogenous states.');
end

[V, Policy, Valt, Policyalt] = UnKronUFHorz(VKron, PolicyKron, ValtKron, PolicyaltKron, n_d, n_a1, n_a, n_z, N_j, vfoptions, 1);

end
