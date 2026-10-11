function [V, Policy] = ValueFnIter_FHorz_ExpAssetSemiExo_uraw(n_d1, n_d2, n_d3, n_a1, n_a2, n_z, n_semiz, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Reconstruct unified arguments for the underlying core engine
N_d1 = max(prod(n_d1), 1);
N_d2 = max(prod(n_d2), 1);
N_d3 = max(prod(n_d3), 1);

if isempty(n_d1) || (length(n_d1)==1 && n_d1==0)
    n_d = [n_d2, n_d3];
    d_grid = [d2_grid; d3_grid];
    N_d1 = 1;
else
    n_d = [n_d1, n_d2, n_d3];
    d_grid = [d1_grid; d2_grid; d3_grid];
end

d_gridvals = CreateGridvals(n_d, d_grid, 1);
n_a = [n_a1, n_a2];
a_grid = [a1_grid; a2_grid];

% 2. Call the unified ExpAsset raw engine
[VKron, PolicyKron] = ValueFnIter_FHorz_ExpAsset_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

% 3. Extract dimensions for decoding
N_d_comb = N_d1 * N_d2 * N_d3;
s2 = size(PolicyKron, 2);
s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4);
s5 = size(PolicyKron, 5);

% 4. Decode the 1-row linear index into the explicit 4-row format required by ExpAssetSemiExo references
Policy = zeros(4, s2, s3, s4, s5, 'gpuArray');

d_ind = rem(PolicyKron - 1, N_d_comb) + 1;
d12_ind = rem(d_ind - 1, N_d1 * N_d2) + 1;

Policy(1,:,:,:,:) = rem(d12_ind - 1, N_d1) + 1;       % Row 1: d1
Policy(2,:,:,:,:) = ceil(d12_ind / N_d1);             % Row 2: d2 (ExpAsset Driver)
Policy(3,:,:,:,:) = ceil(d_ind / (N_d1 * N_d2));      % Row 3: d3 (SemiExo Driver)
Policy(4,:,:,:,:) = ceil(PolicyKron / N_d_comb);      % Row 4: a1prime

% 5. Dynamically shape the state dimensions
state_dims = [prod(n_a1) * prod(n_a2)];
if prod(n_semiz) > 0 && prod(n_z) > 0
    state_dims = [state_dims, prod(n_semiz) * prod(n_z)];
elseif prod(n_semiz) > 0
    state_dims = [state_dims, prod(n_semiz)];
elseif prod(n_z) > 0
    state_dims = [state_dims, prod(n_z)];
end
if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0
    state_dims = [state_dims, prod(vfoptions.n_e)];
end

V = reshape(VKron, [state_dims, N_j]);
Policy = reshape(Policy, [4, state_dims, N_j]);


end
