function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_RiskyAsset(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

if length(vfoptions.n_a2) > 1
    error('Have not yet implemented riskyasset for more than one riskyasset.');
end
if ~isfield(vfoptions, 'refine_d')
    error('When using vfoptions.riskyasset you should also set vfoptions.refine_d.');
end
if sum(vfoptions.refine_d) ~= length(n_d)
    error('vfoptions.refine_d seems to be set up wrong, it is inconsistent with n_d.');
end

vfoptions.d_grid = d_grid;
vfoptions.pi_u   = gpuArray(vfoptions.pi_u);
vfoptions.u_grid = gpuArray(vfoptions.u_grid);

if vfoptions.divideandconquer == 1 && vfoptions.gridinterplayer == 1
    [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_RiskyAsset_DC_GI(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
elseif vfoptions.gridinterplayer == 1
    [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_RiskyAsset_GI(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
elseif vfoptions.divideandconquer == 1
    [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_RiskyAsset_DC(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
end

[VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_RiskyAsset_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

[V, Policy, Valt, Policyalt] = UnKronUFHorz(VKron, PolicyKron, ValtKron, PolicyaltKron, n_d, vfoptions.n_a1, n_a, n_z, N_j, vfoptions, 0);

end
