function [V, Policy] = ValueFnIter_UFHorz_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

if isscalar(n_a)
    [VKron, PolicyKron] = ValueFnIter_UFHorz_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a) >= 2
    [VKron, PolicyKron] = ValueFnIter_UFHorz_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end

s2 = size(PolicyKron, 2); s3 = size(PolicyKron, 3); s4 = size(PolicyKron, 4); s5 = size(PolicyKron, 5);

l_a_choice = length(n_a);
n_rows = l_d + l_a_choice + 2; % [d's, a's, L2off, flag]
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

a_idx = PolicyKron(2,:,:,:,:);
a_cum = 1;
for i = 1:l_a_choice
    Policy(l_d+i,:,:,:,:) = rem(ceil(a_idx / a_cum) - 1, n_a(i)) + 1;
    a_cum = a_cum * n_a(i);
end

n2short = vfoptions.ngridinterp;
L2off = PolicyKron(3,:,:,:,:);
flag  = PolicyKron(4,:,:,:,:);

adjust = (L2off < 1 + n2short + 1);
Policy(l_d+1,:,:,:,:) = Policy(l_d+1,:,:,:,:) - adjust;
L2off = L2off - (n2short + 1) .* (~adjust);

Policy(n_rows-1,:,:,:,:) = L2off;
Policy(n_rows,:,:,:,:) = flag;
V = VKron;


end
