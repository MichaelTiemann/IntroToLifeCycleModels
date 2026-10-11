function [V, Policy] = ValueFnIter_UFHorz_ExpAssetSemiExo(n_d1, n_d2, n_d3, n_a1, n_a2, n_z, n_semiz, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Reconstruct unified arguments for the underlying core engine
if isempty(n_d1) || (length(n_d1)==1 && n_d1==0)
    n_d = [n_d2, n_d3];
    d_grid = [d2_grid; d3_grid];
else
    n_d = [n_d1, n_d2, n_d3];
    d_grid = [d1_grid; d2_grid; d3_grid];
end

% Generate the full joint gridvals internally
d_gridvals = CreateGridvals(n_d, d_grid, 1);
n_a = [n_a1, n_a2];

% 2. Call the unified ExpAsset raw engine
if vfoptions.divideandconquer == 1
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_raw(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

% 3. Extract dimensions for decoding
N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && n_d(1) == 0; l_d = 0; end

s2 = size(PolicyKron, 2);
s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4);
s5 = size(PolicyKron, 5);

% 4. Decode the 1-row linear index into [l_d + 1] choice rows
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

% 5. Dynamically shape the final state dimensions
state_dims = n_a; % ExpAsset always naturally splits into [n_a1, n_a2]
if prod(n_semiz) > 0; state_dims = [state_dims, n_semiz]; end
if prod(n_z) > 0; state_dims = [state_dims, n_z]; end
if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0
    state_dims = [state_dims, vfoptions.n_e];
end

% 6. Reshape into fully unkronned state space
V = reshape(VKron, [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);


end
