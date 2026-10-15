function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

n_a1 = vfoptions.n_a1;

if isscalar(n_a1)
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a1) == 2
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    error('Cannot use vfoptions.gridinterplayer with more than two standard endogenous states.');
end

% If _raw returned 4 rows (unadjusted), resolve boundary flag (row 4) into rows 2 and 3
if size(PolicyKron, 1) == 4
    n2short = vfoptions.ngridinterp;
    L2_adjust = (PolicyKron(3,:,:,:,:) - 1 - n2short)   .* (PolicyKron(4,:,:,:,:) == 2) ...
              + (PolicyKron(3,:,:,:,:) - 1)             .* (PolicyKron(4,:,:,:,:) == 1) ...
              + (PolicyKron(3,:,:,:,:) - 2*n2short - 1) .* (PolicyKron(4,:,:,:,:) == 3);

    PolicyKron(2,:,:,:,:) = PolicyKron(2,:,:,:,:) + (PolicyKron(4,:,:,:,:) == 3) - (PolicyKron(4,:,:,:,:) == 1);
    PolicyKron(3,:,:,:,:) = max(min(1 + n2short + L2_adjust, 1 + 2*n2short), 1);
    PolicyKron = PolicyKron(1:3, :, :, :, :);
end

if vfoptions.outputkron == 1
    if n_d(1) == 0 && size(PolicyKron, 1) == 3
        Policy = PolicyKron(2:3, :, :, :, :);
    else
        Policy = PolicyKron;
    end
    V = VKron;
    return;
end

N_d = max(prod(n_d), 1);
l_d = length(n_d); if n_d(1) == 0; l_d = 0; end
l_a1 = length(n_a1);

% Handle case where _raw already stripped row 1 when n_d(1) == 0
if size(PolicyKron, 1) == 2
    a_kron_row = 1;
    L2_kron_row = 2;
else
    a_kron_row = 2;
    L2_kron_row = 3;
end

s2 = size(PolicyKron, 2); s3 = size(PolicyKron, 3); s4 = size(PolicyKron, 4); s5 = size(PolicyKron, 5);
n_rows = l_d + l_a1 + 1; % [d's, a1's, L2_index]
Policy = zeros(n_rows, s2, s3, s4, s5, 'gpuArray');

if l_d > 0
    d_idx = PolicyKron(1,:,:,:,:);
    d_cum = 1;
    for i = 1:l_d
        Policy(i,:,:,:,:) = rem(ceil(d_idx / d_cum) - 1, n_d(i)) + 1;
        d_cum = d_cum * n_d(i);
    end
end

a1_idx = PolicyKron(a_kron_row,:,:,:,:);
a1_cum = 1;
for i = 1:l_a1
    Policy(l_d + i,:,:,:,:) = rem(ceil(a1_idx / a1_cum) - 1, n_a1(i)) + 1;
    a1_cum = a1_cum * n_a1(i);
end

Policy(n_rows,:,:,:,:) = PolicyKron(L2_kron_row,:,:,:,:);

state_dims = n_a;
if prod(vfoptions.n_semiz) > 0; state_dims = [state_dims, vfoptions.n_semiz]; end
if prod(n_z) > 0;               state_dims = [state_dims, n_z];               end
if prod(vfoptions.n_e) > 0;     state_dims = [state_dims, vfoptions.n_e];     end

V      = reshape(VKron,  [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);

end
