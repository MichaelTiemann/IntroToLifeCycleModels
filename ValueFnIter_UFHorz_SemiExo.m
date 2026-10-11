function [V, Policy] = ValueFnIter_UFHorz_SemiExo(n_d1, n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Reconstruct unified arguments for uraw
N_d1 = max(prod(n_d1), 1);
N_d2 = max(prod(n_d2), 1);

if isempty(n_d1) || (length(n_d1)==1 && n_d1==0)
    n_d = n_d2;
    d_gridvals = d2_gridvals;
    has_d1 = false;
else
    n_d = [n_d1, n_d2];
    % Fully expand the joint grid (CreateGridvals outputs tall N_d x l_d arrays)
    % d1 cycles fast (repmat vertically), d2 cycles slow (kron vertically)
    d_gridvals = [repmat(d1_gridvals, N_d2, 1), kron(d2_gridvals, ones(N_d1, 1))];
    has_d1 = true;
end

if isempty(n_d2) || (length(n_d2)==1 && n_d2==0)
    has_d2 = false;
else
    has_d2 = true;
end

% 2. Call the unified raw engine
[VKron, PolicyKron] = ValueFnIter_UFHorz_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

% 3. Extract dimensions for decoding
N_d = N_d1 * N_d2;
s2 = size(PolicyKron, 2);
s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4);
s5 = size(PolicyKron, 5);

% 4. Decode the 1-row linear index into 2 or 3 distinct choice rows
if has_d2 && has_d1
    n_rows = 3;
    Pol_decoded = zeros(3, s2, s3, s4, s5, 'gpuArray');
    d12_idx = rem(PolicyKron - 1, N_d) + 1;
    Pol_decoded(1,:,:,:,:) = rem(d12_idx - 1, N_d1) + 1; % d1_idx
    Pol_decoded(2,:,:,:,:) = ceil(d12_idx / N_d1);       % d2_idx
    Pol_decoded(3,:,:,:,:) = ceil(PolicyKron / N_d);     % aprime_idx
elseif has_d2 && ~has_d1
    n_rows = 2;
    Pol_decoded = zeros(2, s2, s3, s4, s5, 'gpuArray');
    Pol_decoded(1,:,:,:,:) = rem(PolicyKron - 1, N_d2) + 1; % d2_idx
    Pol_decoded(2,:,:,:,:) = ceil(PolicyKron / N_d2);       % aprime_idx
else
    n_rows = 1;
    Pol_decoded = PolicyKron;
end

% 5. Dynamically determine state dimensions for final Unkronned Reshape
state_dims = n_a;
if prod(n_semiz) > 0; state_dims = [state_dims, n_semiz]; end
if prod(n_z) > 0; state_dims = [state_dims, n_z]; end
if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0
    state_dims = [state_dims, vfoptions.n_e];
end

% 6. Reshape into fully unkronned state space matching legacy SemiExo output
V = reshape(VKron, [state_dims, N_j]);
Policy = reshape(Pol_decoded, [n_rows, state_dims, N_j]);


end
