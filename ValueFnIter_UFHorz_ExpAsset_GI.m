function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

if length(n_a) == 2
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a) >= 3
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    error('Experience Asset requires at least 2 endogenous states.');
end

N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end

s2 = size(PolicyKron, 2); s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4); s5 = size(PolicyKron, 5);

l_a_choice = length(n_a) - 1;
n_rows = l_d + l_a_choice + 1; % [d's, a's, flag]
Policy = zeros(n_rows, s2, s3, s4, s5, 'gpuArray');

d_idx = PolicyKron(1,:,:,:,:);
if l_d == 1
    Policy(1,:,:,:,:) = d_idx;
elseif l_d > 1
    d_cum = 1;
    for i = 1:l_d
        Policy(i,:,:,:,:) = rem(ceil(d_idx / d_cum) - 1, n_d(i)) + 1;
        d_cum = d_cum * n_d(i);
    end
end

n2short = vfoptions.ngridinterp;

if length(n_a) == 2
    % PolicyKron is [d_idx; a1fine_idx; flag_val]
    a1fine_idx = PolicyKron(2,:,:,:,:);
    Policy(l_d + 1,:,:,:,:) = (a1fine_idx - 1) / (n2short + 1) + 1;
    Policy(l_d + 2,:,:,:,:) = PolicyKron(3,:,:,:,:);
else
    % PolicyKron is [d_idx; joint_midpt; L2off; flag_val]
    a_idx = PolicyKron(2,:,:,:,:);
    a_cum = 1;
    for i = 1:l_a_choice
        Policy(l_d+i,:,:,:,:) = rem(ceil(a_idx / a_cum) - 1, n_a(i)) + 1;
        a_cum = a_cum * n_a(i);
    end
    L2off = PolicyKron(3,:,:,:,:);
    Policy(l_d + 1,:,:,:,:) = Policy(l_d + 1,:,:,:,:) + (L2off - (n2short + 2)) / (n2short + 1);
    Policy(l_d + l_a_choice + 1,:,:,:,:) = PolicyKron(4,:,:,:,:);
end

V = VKron;


end
