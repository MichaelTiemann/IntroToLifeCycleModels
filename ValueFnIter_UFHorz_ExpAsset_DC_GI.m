function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_ExpAsset_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_a1 = vfoptions.n_a1;
if sum(n_a1) == 0 || n_a1(1) < 5
    error('Cannot use vfoptions.divideandconquer=1 with less than 5 points in the first standard endogenous state (n_a1(1)).');
end
if ~isfield(vfoptions, 'level1n') || isempty(vfoptions.level1n)
    vfoptions.level1n = floor(sqrt(n_a1(1)));
elseif ~isscalar(n_a1) && length(vfoptions.level1n) > 1
    if vfoptions.level1n(2) >= n_a1(2)
        vfoptions.level1n = vfoptions.level1n(1);
    else
        error('With two standard endogenous states, can only do divide-and-conquer in the first endogenous state.');
    end
end

if isscalar(n_a1)
    [VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_ExpAsset_DC1_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a1) == 2
    [VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_ExpAsset_DC2A_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    error('Cannot use vfoptions.gridinterplayer with more than two standard endogenous states.');
end

[V, Policy, Valt, Policyalt] = UnKronUFHorz(VKron, PolicyKron, ValtKron, PolicyaltKron, n_d, n_a1, n_a, n_z, N_j, vfoptions, 1);

end
