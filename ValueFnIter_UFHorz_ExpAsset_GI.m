function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Call the raw engine
[VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

% 2. Extract dimensions
N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end

s2 = size(PolicyKron, 2);
s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4);
s5 = size(PolicyKron, 5);

% 3. Decode 1-row linear index into [l_d] choice rows, then append GI rows
n_rows = l_d + 3;
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

% 4. Adjust Policy Flags and Shrink-Wrap Continuous Choices
n2short = vfoptions.ngridinterp;
midpt = PolicyKron(2,:,:,:,:);
L2off = PolicyKron(3,:,:,:,:);
flag  = PolicyKron(4,:,:,:,:);

adjust = (L2off < 1 + n2short + 1);
midpt = midpt - adjust;
L2off = L2off - (n2short + 1) .* (~adjust);

Policy(l_d+1,:,:,:,:) = midpt;
Policy(l_d+2,:,:,:,:) = L2off;
Policy(l_d+3,:,:,:,:) = flag;

V = VKron;


end
