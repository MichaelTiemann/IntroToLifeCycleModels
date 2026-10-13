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
n_rows = l_d + l_a_choice + 2; % [d's, a's, L2, flag]
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
n_a1_c = n_a(1);

if length(n_a) == 2
    % PolicyKron is [d_idx; joint_midpt; L2off; flag_val]
    joint_midpt = PolicyKron(2,:,:,:,:);
    L2off = PolicyKron(3,:,:,:,:);
    flag_val = PolicyKron(4,:,:,:,:);

    a1_midpt = rem(joint_midpt - 1, n_a1_c) + 1;
    a1fine_idx = (a1_midpt - 1)*(n2short + 1) + 1 + (L2off - (n2short + 2));
else
    % PolicyKron is [d_idx; joint_midpt; L2off; flag_val]
    joint_midpt = PolicyKron(2,:,:,:,:);
    L2off = PolicyKron(3,:,:,:,:);
    flag_val = PolicyKron(4,:,:,:,:);

    a1_midpt = rem(joint_midpt - 1, n_a1_c) + 1;
    a1fine_idx = (a1_midpt - 1)*(n2short + 1) + 1 + (L2off - (n2short + 2));

    a_idx = ceil(joint_midpt / n_a1_c);
    a_cum = 1;
    for i = 2:l_a_choice
        Policy(l_d+i,:,:,:,:) = rem(ceil(a_idx / a_cum) - 1, n_a(i)) + 1;
        a_cum = a_cum * n_a(i);
    end
end

% Convert absolute fine index into coarse lower-bound and L2 offset
a1_idx = floor((a1fine_idx - 1) ./ (n2short + 1)) + 1;
a1_idx = min(a1_idx, n_a1_c - 1); % Ensure it doesn't index out of bounds at the top
L2 = a1fine_idx - 1 - (a1_idx - 1).*(n2short + 1);

Policy(l_d + 1,:,:,:,:) = a1_idx;
Policy(l_d + l_a_choice + 1,:,:,:,:) = L2;
Policy(l_d + l_a_choice + 2,:,:,:,:) = flag_val;

V = VKron;


end
