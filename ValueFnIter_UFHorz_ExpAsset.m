function [V, Policy] = ValueFnIter_UFHorz_ExpAsset(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Dispatch to Grid Interpolation / Divide-and-Conquer Bridges
if vfoptions.divideandconquer == 1 && vfoptions.gridinterplayer == 1
    [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
elseif vfoptions.gridinterplayer == 1
    [V, Policy] = ValueFnIter_UFHorz_ExpAsset_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
elseif vfoptions.divideandconquer == 1
    [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
end

% 2. Base ExpAsset Solver (Returns 1-row Kron index)
[VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

if vfoptions.outputkron == 1
    V = VKron;
    Policy = PolicyKron;
    return;
end

% 3. Decode 1-row Kron Policy into [l_d + l_a1] choice rows and UnKron dimensions
N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end

n_a1 = vfoptions.n_a1;
l_a1 = length(n_a1); if isempty(n_a1) || n_a1(1) == 0; l_a1 = 0; end

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

if l_a1 > 0
    a1_idx = ceil(PolicyKron / N_d);
    a1_cum = 1;
    for i = 1:l_a1
        Policy(l_d + i,:,:,:,:) = rem(ceil(a1_idx / a1_cum) - 1, n_a1(i)) + 1;
        a1_cum = a1_cum * n_a1(i);
    end
end

% 4. Reshape V and Policy to match canonical VFI Toolkit output dimensions
state_dims = n_a;
if prod(vfoptions.n_semiz) > 0; state_dims = [state_dims, vfoptions.n_semiz]; end
if prod(n_z) > 0;               state_dims = [state_dims, n_z];               end
if prod(vfoptions.n_e) > 0;     state_dims = [state_dims, vfoptions.n_e];     end

V      = reshape(VKron,  [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);

end
