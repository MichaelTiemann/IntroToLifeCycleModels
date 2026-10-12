function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Route based on the number of endogenous states
% Experience assets require at least 2 endogenous states (1 standard, 1 exp)
if length(n_a) == 2
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
elseif length(n_a) >= 3
    [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_GI2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    error('Experience Asset requires at least 2 endogenous states.');
end

% 2. Extract dimensions
N_d = max(prod(n_d), 1);
l_d = length(n_d); if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end

s2 = size(PolicyKron, 2);
s3 = size(PolicyKron, 3);
s4 = size(PolicyKron, 4);
s5 = size(PolicyKron, 5);

% 3. Decode 1-row linear index into [l_d] choice rows, then append GI rows
% PolicyKron from raw always has 4 rows: [d_idx; midpt; L2off; flag]
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

% 4. Append the remaining continuous GI rows.
% CRITICAL: Legacy GI does NOT shrink-wrap continuous boundary offsets.
% It leaves L2offset in [1, n2long] and handles boundaries via the flag.
Policy(l_d+1,:,:,:,:) = PolicyKron(2,:,:,:,:); % midpt (or joint a_midpt)
Policy(l_d+2,:,:,:,:) = PolicyKron(3,:,:,:,:); % L2off
Policy(l_d+3,:,:,:,:) = PolicyKron(4,:,:,:,:); % flag

V = VKron;


end
