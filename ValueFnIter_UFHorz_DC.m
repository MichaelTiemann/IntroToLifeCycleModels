function [V, Policy, Valt, Policyalt] = ValueFnIter_UFHorz_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

if n_a(1) < 5
    error('Cannot use vfoptions.divideandconquer=1 with less than 5 points in the first a variable.');
end
if ~isfield(vfoptions, 'level1n') || isempty(vfoptions.level1n)
    vfoptions.level1n = floor(sqrt(n_a(1)));
elseif ~isscalar(n_a) && length(vfoptions.level1n) > 1
    if vfoptions.level1n(2) >= n_a(2)
        vfoptions.level1n = vfoptions.level1n(1);
    else
        error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both).');
    end
end

if isscalar(n_a)
    [VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [VKron, PolicyKron, ValtKron, PolicyaltKron] = ValueFnIter_UFHorz_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

[V, Policy, Valt, Policyalt] = UnKronUFHorz(VKron, PolicyKron, ValtKron, PolicyaltKron, n_d, n_a, n_a, n_z, N_j, vfoptions, 0);

end
