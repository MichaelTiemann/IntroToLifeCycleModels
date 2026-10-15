function [V, Policy] = ValueFnIter_UFHorz_SemiExo(n_d1, n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Inject SemiExo dimensions into vfoptions for Setup_UFHorz
if sum(n_d1) > 0
    vfoptions.n_d1 = n_d1;
    l_d1 = length(n_d1);
    n_d = [n_d1, n_d2];
else
    vfoptions.n_d1 = 0;
    l_d1 = 0;
    n_d = n_d2;
end
vfoptions.n_d2 = n_d2;
l_d2 = length(n_d2); if sum(n_d2) == 0; l_d2 = 0; end
vfoptions.n_semiz = n_semiz;
vfoptions.semiz_gridvals_J = semiz_gridvals_J;
vfoptions.pi_semiz_J = pi_semiz_J;

N_d1 = max(prod(n_d1), 1);
N_d2 = max(prod(n_d2), 1);

% 2. Build standard 2D [N_d1 * N_d2, l_d] joint d_gridvals for Setup_UFHorz
if l_d1 > 0
    d1_clean = reshape(d1_gridvals, [N_d1, l_d1]);
    d2_clean = reshape(d2_gridvals, [N_d2, l_d2]);
    d_gridvals = [repmat(d1_clean, N_d2, 1), repelem(d2_clean, N_d1, 1)];
else
    d_gridvals = reshape(d2_gridvals, [N_d2, l_d2]);
end

% 3. Dispatch to DC_GI, GI, DC, or Base Raw Engine
if vfoptions.divideandconquer == 1 && vfoptions.gridinterplayer == 1
    [V, Policy] = ValueFnIter_UFHorz_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
elseif vfoptions.gridinterplayer == 1
    [V, Policy] = ValueFnIter_UFHorz_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
elseif vfoptions.divideandconquer == 1
    [V, Policy] = ValueFnIter_UFHorz_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return;
end

[VKron, PolicyKron] = ValueFnIter_UFHorz_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

if vfoptions.outputkron == 1
    V = VKron;
    Policy = PolicyKron;
    return;
end

% 4. Decode 1-row Kron Policy into [l_d + l_a] rows and UnKron dimensions
N_d = N_d1 * N_d2;
l_d = l_d1 + l_d2;
l_a = length(n_a);

s2 = size(PolicyKron, 2); s3 = size(PolicyKron, 3); s4 = size(PolicyKron, 4); s5 = size(PolicyKron, 5);
n_rows = l_d + l_a;
Policy = zeros(n_rows, s2, s3, s4, s5, 'gpuArray');

if l_d > 0
    d_idx = rem(PolicyKron - 1, N_d) + 1;
    d_cum = 1;
    for i = 1:l_d
        Policy(i,:,:,:,:) = rem(ceil(d_idx / d_cum) - 1, n_d(i)) + 1;
        d_cum = d_cum * n_d(i);
    end
end

a_idx = ceil(PolicyKron / N_d);
a_cum = 1;
for i = 1:l_a
    Policy(l_d + i,:,:,:,:) = rem(ceil(a_idx / a_cum) - 1, n_a(i)) + 1;
    a_cum = a_cum * n_a(i);
end

state_dims = n_a;
if prod(n_semiz) > 0;       state_dims = [state_dims, n_semiz];       end
if prod(n_z) > 0;           state_dims = [state_dims, n_z];           end
if prod(vfoptions.n_e) > 0; state_dims = [state_dims, vfoptions.n_e]; end

V      = reshape(VKron,  [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);

end
