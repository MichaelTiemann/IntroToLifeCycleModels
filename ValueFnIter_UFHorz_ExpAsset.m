function [V, Policy] = ValueFnIter_UFHorz_ExpAsset(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Grid Interpolation (Has its own bridge to handle decoding and shrink-wrapping)
if vfoptions.gridinterplayer == 1
    a_grid = [vfoptions.a1_grid(:); vfoptions.a2_grid(:)];
    [V, Policy] = ValueFnIter_UFHorz_ExpAsset_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
end

% 2. Divide and Conquer / Raw (Returns 1-row Kron index to be decoded)
if vfoptions.divideandconquer == 1
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

% 3. Extract dimensions for 1-row encoded policies
N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end

s2 = size(PolicyKron, 2);
s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4);
s5 = size(PolicyKron, 5);

% 4. Decode 1-row linear index into [l_d + 1] choice rows
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
V = VKron;


end
