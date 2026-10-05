function [V,Policy]=ValueFnIter_FHorz_ExpAsset(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% --- Parse aprimeFnParamNames positionally based on explicit user configuration ---
l_d2 = length(n_d2);
l_a2 = length(n_a2);

% Check explicit vfoptions flags to see which exogenous states are passed to aprimeFn
has_exp_z = vfoptions.experienceassetz == 1 || vfoptions.experienceassetze == 1;
has_exp_e = vfoptions.experienceassete == 1 || vfoptions.experienceassetze == 1;
has_exp_u = vfoptions.experienceassetu == 1;

% Count the active state variables that aprimeFn expects
l_z_active = 0; if has_exp_z; l_z_active = length(n_z); end
l_e_active = 0; if has_exp_e; l_e_active = length(vfoptions.n_e); end
l_u_active = 0; if has_exp_u; l_u_active = length(vfoptions.n_u); end % (assuming n_u is in vfoptions)

% Total positional inputs = d2 + a2 + 'whicha' selector (if l_a2 >= 2) + active exogenous states
num_state_args = l_d2 + l_a2 + (l_a2 >= 2) + l_z_active + l_e_active + l_u_active;

temp = getAnonymousFnInputNames(aprimeFn);
if length(temp) > num_state_args
    aprimeFnParamNames = {temp{num_state_args + 1 : end}}; % The parameters start immediately after the state args
else
    aprimeFnParamNames = {};
end

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_a1 > 0
    a1_gridvals = CreateGridvals(n_a1, a1_grid, 1);
else
    a1_gridvals = []; 
end
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
if N_d1>0
    d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
else
    d_gridvals=[]; % not used
end


%% Dispatch
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    % Solve by doing Divide-and-Conquer, and then a grid interpolation layer
    [V,Policy]=ValueFnIter_FHorz_ExpAsset_DC_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.divideandconquer==1
    % Solve using Divide-and-Conquer algorithm
    [V,Policy]=ValueFnIter_FHorz_ExpAsset_DC(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.gridinterplayer==1
    % Solve using grid interpolation layer
    [V,Policy]=ValueFnIter_FHorz_ExpAsset_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
end


%% Plain case: no divide-and-conquer, no grid interpolation layer
% Core Dispatcher (Handles all permutations of d1, d2, and a1 dynamically)
if N_e == 0
    [VKron, PolicyKron] = ValueFnIter_FHorz_ExpAsset_raw(n_d1, n_d2, n_a1, n_a2, n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
else % N_e > 0
    [VKron, PolicyKron] = ValueFnIter_FHorz_ExpAsset_e_raw(n_d1, n_d2, n_a1, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

if N_d1>0
    n_d=[n_d1,n_d2];
else
    n_d=n_d2;
end
if N_a1>0
    n_d=[n_d,n_a1];
    n_a=[n_a1,n_a2];
else
    % n_d=n_d;
    n_a=n_a2;
end

% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if N_e==0
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
    end
else
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
    end
end


end


