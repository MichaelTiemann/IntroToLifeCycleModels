function [V, Policy] = ValueFnIter_UFHorz_ExpAsset(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Call the core bare-metal engine
if vfoptions.divideandconquer == 1
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_raw(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

% 2. Extract dimensions
N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && n_d(1) == 0; l_d = 0; end

s2 = size(PolicyKron, 2);
s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4);
s5 = size(PolicyKron, 5);

% 3. Decode 1-row linear index into [l_d + 1] choice rows
n_rows = l_d + 1;
Policy = zeros(n_rows, s2, s3, s4, s5, 'gpuArray');

d_idx = rem(PolicyKron - 1, N_d) + 1;

if l_d == 1
    Policy(1,:,:,:,:) = d_idx;
elseif l_d > 1
    d_cum = 1;
    for i = 1:l_d
        Policy(i,:,:,:,:) = rem(ceil(d_idx / d_cum) - 1, n_d(i)) + 1;
        d_cum = d_cum * n_d(i);
    end
end

Policy(n_rows,:,:,:,:) = ceil(PolicyKron / N_d); % aprime_idx

% 4. Dynamically shape the final state dimensions
state_dims = n_a; % n_a is inherently [n_a1, n_a2] in ExpAsset context
if prod(n_z) > 0
    state_dims = [state_dims, n_z];
end
if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0
    state_dims = [state_dims, vfoptions.n_e];
end

% 5. Apply unkronned shapes
V = reshape(VKron, [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);


end
