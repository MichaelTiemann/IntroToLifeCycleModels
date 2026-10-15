function [V, Policy] = ValueFnIter_UFHorz_RiskyAsset_DC(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

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
        error('With RiskyAsset DC2A, can only do divide-and-conquer on the first standard endogenous state.');
    end
end

% 1. Dispatch to DC1_raw (1 standard endo state) or DC2A_raw (2 standard endo states)
if isscalar(n_a1)
    [VKron, PolicyKron] = ValueFnIter_UFHorz_RiskyAsset_DC1_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a1) == 2
    [VKron, PolicyKron] = ValueFnIter_UFHorz_RiskyAsset_DC2A_raw(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    error('Cannot use vfoptions.divideandconquer with more than two standard endogenous states.');
end

if vfoptions.outputkron == 1
    V = VKron;
    Policy = PolicyKron;
    return;
end

% 2. Decode 1-row Kron Policy into [l_d + l_a1] choice rows and UnKron dimensions
N_d = max(prod(n_d), 1);
l_d = length(n_d); if n_d(1) == 0; l_d = 0; end
l_a1 = length(n_a1);

s2 = size(PolicyKron, 2); s3 = size(PolicyKron, 3); s4 = size(PolicyKron, 4); s5 = size(PolicyKron, 5);
n_rows = l_d + l_a1;
Policy = zeros(n_rows, s2, s3, s4, s5, 'gpuArray');

if l_d > 0
    d_idx = rem(PolicyKron - 1, N_d) + 1;
    d_cum = 1;
    for i = 1:l_d
        Policy(i,:,:,:,:) = rem(ceil(d_idx / d_cum) - 1, n_d(i)) + 1;
        d_cum = d_cum * n_d(i);
    end
end

a1_idx = ceil(PolicyKron / N_d);
a1_cum = 1;
for i = 1:l_a1
    Policy(l_d + i,:,:,:,:) = rem(ceil(a1_idx / a1_cum) - 1, n_a1(i)) + 1;
    a1_cum = a1_cum * n_a1(i);
end

% 3. Reshape V and Policy to match canonical VFI Toolkit output dimensions
state_dims = n_a;
if prod(vfoptions.n_semiz) > 0; state_dims = [state_dims, vfoptions.n_semiz]; end
if prod(n_z) > 0;               state_dims = [state_dims, n_z];               end
if prod(vfoptions.n_e) > 0;     state_dims = [state_dims, vfoptions.n_e];     end

V      = reshape(VKron,  [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);

end
