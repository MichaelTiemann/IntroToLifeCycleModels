function [V, Policy] = ValueFnIter_UFHorz_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Validate & Default level1n
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

% 2. Dispatch to DC1 or DC2A
if isscalar(n_a)
    [VKron, PolicyKron] = ValueFnIter_UFHorz_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [VKron, PolicyKron] = ValueFnIter_UFHorz_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

if vfoptions.outputkron == 1
    V = VKron;
    Policy = PolicyKron;
    return;
end

% 3. Decode 1-row Kron Policy into [l_d + l_a] choice rows and UnKron dimensions
N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end
l_a = length(n_a);

s2 = size(PolicyKron, 2); s3 = size(PolicyKron, 3); s4 = size(PolicyKron, 4); s5 = size(PolicyKron, 5);
n_rows = l_d + l_a;
Policy = zeros(n_rows, s2, s3, s4, s5, 'gpuArray');

if l_d > 0
    d_idx = rem(PolicyKron - 1, N_d) + 1;
    d_cum = 1;
    for i = 1:l_d
        Policy(i,:,:,:,:) = rem(ceil(d_idx / d_cum) - 1, n_d(i)) + 1;
        d_cum = d_cum * n_d(i);
    end
end

a_idx = ceil(PolicyKron / N_d);
a_cum = 1;
for i = 1:l_a
    Policy(l_d + i,:,:,:,:) = rem(ceil(a_idx / a_cum) - 1, n_a(i)) + 1;
    a_cum = a_cum * n_a(i);
end

state_dims = n_a;
if prod(vfoptions.n_semiz) > 0; state_dims = [state_dims, vfoptions.n_semiz]; end
if prod(n_z) > 0;               state_dims = [state_dims, n_z];               end
if prod(vfoptions.n_e) > 0;     state_dims = [state_dims, vfoptions.n_e];     end

V      = reshape(VKron,  [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);

end
