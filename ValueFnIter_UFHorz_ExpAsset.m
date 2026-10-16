function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_ExpAsset(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Dispatch to Grid Interpolation / Divide-and-Conquer Bridges
if vfoptions.divideandconquer == 1 && vfoptions.gridinterplayer == 1
    [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_ExpAsset_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
elseif vfoptions.gridinterplayer == 1
    [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_ExpAsset_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
elseif vfoptions.divideandconquer == 1
    [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_ExpAsset_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
end

% 2. Base ExpAsset Solver (Returns 1-row Kron index)
[VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_ExpAsset_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

[V, Policy, Valt, Policyalt] = UnKronUFHorz(VKron, PolicyKron, ValtKron, PolicyaltKron, n_d, vfoptions.n_a1, n_a, n_z, N_j, vfoptions, 0);

end
