function [V, Policy] = ValueFnIter_UFHorz_SemiExo(n_d1, n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Inject the necessary SemiExo dimensions into vfoptions so Setup_UFHorz can find them
if sum(n_d1) > 0
    vfoptions.n_d1 = n_d1;
    l_d1 = length(n_d1);
else
    vfoptions.n_d1 = 0;
    l_d1 = 0;
end

vfoptions.n_d2 = n_d2;
l_d2 = length(n_d2); if sum(n_d2) == 0; l_d2 = 0; end

vfoptions.n_semiz = n_semiz;
vfoptions.semiz_gridvals_J = semiz_gridvals_J;
vfoptions.pi_semiz_J = pi_semiz_J;

if sum(n_d1) > 0
    n_d = [n_d1, n_d2];
else
    n_d = n_d2;
end

N_d1 = max(prod(n_d1), 1);
N_d2 = max(prod(n_d2), 1);

% 2. Build the exact [N_d1, l_d, N_d2] combined tensor grid
% CreateGridvals strictly outputs grids as [N_points, l_vars]
if sum(n_d1) > 0
    d1_clean = reshape(d1_gridvals, [N_d1, l_d1]); % Ensure column format
    d2_clean = reshape(d2_gridvals, [N_d2, l_d2]); % Ensure column format

    d1_rep = repmat(d1_clean, [1, 1, N_d2]); % [N_d1, l_d1, N_d2]

    d2_reshaped = reshape(d2_clean, [1, l_d2, N_d2]);
    d2_rep = repmat(d2_reshaped, [N_d1, 1, 1]); % [N_d1, l_d2, N_d2]

    d_gridvals = cat(2, d1_rep, d2_rep); % [N_d1, l_d1 + l_d2, N_d2]
else
    d2_clean = reshape(d2_gridvals, [N_d2, l_d2]);
    d_gridvals = reshape(d2_clean, [1, l_d2, N_d2]);
end

% 3. Pass to the universal raw engine
[V, PolicyKron] = ValueFnIter_UFHorz_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

% 4. UnKron the Policy
n_actions = [n_d, n_a];
l_actions = length(n_actions);

s2 = size(PolicyKron, 2);
s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4);
s5 = size(PolicyKron, 5);

Policy = zeros(l_actions, s2, s3, s4, s5, 'gpuArray');

cum_dims = 1;
for i = 1:l_actions
    Policy(i, :, :, :, :) = rem(ceil(PolicyKron / cum_dims) - 1, n_actions(i)) + 1;
    cum_dims = cum_dims * n_actions(i);
end

% 5. Shape the final output to strictly match legacy API
sz_V = n_a;
if prod(n_semiz) > 0; sz_V = [sz_V, n_semiz]; end
if prod(n_z) > 0; sz_V = [sz_V, n_z]; end
if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0; sz_V = [sz_V, vfoptions.n_e]; end
sz_V = [sz_V, N_j];

sz_P = [l_actions, sz_V];

V = reshape(V, sz_V);
Policy = reshape(Policy, sz_P);


end
