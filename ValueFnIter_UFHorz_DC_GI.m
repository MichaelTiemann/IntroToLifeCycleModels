function [V, Policy] = ValueFnIter_UFHorz_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

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
    [VKron, PolicyKron] = ValueFnIter_UFHorz_DC1_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a) == 2
    if isscalar(vfoptions.ngridinterp)
        [VKron, PolicyKron] = ValueFnIter_UFHorz_DC2A_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        error('vfoptions.gridinterplayer=1 with two endogenous states can only be applied to the first of the two endo states.');
    end
else
    error('Cannot use vfoptions.gridinterplayer with more than two endogenous states (you have length(n_a)>2).');
end

% Resolve boundary flag (row 4) into coarse midpoint (row 2) and fine L2 index (row 3)
n2short = vfoptions.ngridinterp;
L2_adjust = (PolicyKron(3,:,:,:,:) - 1 - n2short)   .* (PolicyKron(4,:,:,:,:) == 2) ...
          + (PolicyKron(3,:,:,:,:) - 1)             .* (PolicyKron(4,:,:,:,:) == 1) ...
          + (PolicyKron(3,:,:,:,:) - 2*n2short - 1) .* (PolicyKron(4,:,:,:,:) == 3);

PolicyKron(2,:,:,:,:) = PolicyKron(2,:,:,:,:) + (PolicyKron(4,:,:,:,:) == 3) - (PolicyKron(4,:,:,:,:) == 1);
PolicyKron(3,:,:,:,:) = max(min(1 + n2short + L2_adjust, 1 + 2*n2short), 1);
PolicyKron = PolicyKron(1:3, :, :, :, :);

if vfoptions.outputkron == 1
    if isempty(n_d) || n_d(1) == 0
        Policy = PolicyKron(2:3, :, :, :, :);
    else
        Policy = PolicyKron;
    end
    V = VKron;
    return;
end

N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end
l_a = length(n_a);

s2 = size(PolicyKron, 2); s3 = size(PolicyKron, 3); s4 = size(PolicyKron, 4); s5 = size(PolicyKron, 5);
n_rows = l_d + l_a + 1; % [d's, a's, L2_index]
Policy = zeros(n_rows, s2, s3, s4, s5, 'gpuArray');

if l_d > 0
    d_idx = PolicyKron(1,:,:,:,:);
    d_cum = 1;
    for i = 1:l_d
        Policy(i,:,:,:,:) = rem(ceil(d_idx / d_cum) - 1, n_d(i)) + 1;
        d_cum = d_cum * n_d(i);
    end
end

a_idx = PolicyKron(2,:,:,:,:);
a_cum = 1;
for i = 1:l_a
    Policy(l_d + i,:,:,:,:) = rem(ceil(a_idx / a_cum) - 1, n_a(i)) + 1;
    a_cum = a_cum * n_a(i);
end

Policy(n_rows,:,:,:,:) = PolicyKron(3,:,:,:,:);

state_dims = n_a;
if prod(vfoptions.n_semiz) > 0; state_dims = [state_dims, vfoptions.n_semiz]; end
if prod(n_z) > 0;               state_dims = [state_dims, n_z];               end
if prod(vfoptions.n_e) > 0;     state_dims = [state_dims, vfoptions.n_e];     end

V      = reshape(VKron,  [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);

end
