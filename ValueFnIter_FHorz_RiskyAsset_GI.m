function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==0, vfoptions.gridinterplayer==1 (Plain GI)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%%
if N_a1==0
    error('Cannot use grid interpolation layer with riskyasset if there is no standard endogenous state (N_a1==0)')
end
if ~isfield(vfoptions,'ngridinterp')
    vfoptions.ngridinterp=9;
end

% Two standard endogenous assets -> the GI2A raws.
if length(n_a1)>1
    if length(n_a1)>2
        error('riskyasset gridinterplayer supports at most two standard endogenous assets')
    end
    % a1_grid holds both standard endogenous states (stacked); n_a2/a2_grid hold the riskyasset
    n_a3=n_a2;
    a3_grid=a2_grid;
    a2_grid=a1_grid(n_a1(1)+1:end);
    a1_grid=a1_grid(1:n_a1(1));
    n_a2=n_a1(2);
    n_a1=n_a1(1);
    % a is divided into a1 (first standard endogenous state, the one the grid interpolation layer refines), a2 (second standard endogenous state, folded) and a3 (the riskyasset)
    if N_e==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    % Policy channels: nod1 is 4 (d2, d3, a1prime, a2prime), with d1 it is 5 (d1, d2, d3, a1prime, a2prime).
    % The grid interpolation layer appends the L2 and L2flag rows, which UnKronPolicyIndexes*
    % passes through unchanged when vfoptions.gridinterplayer==1.
elseif N_e==0 % no e variable
    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
else % N_e
    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

%% Reconstruct full n_a array
if exist('n_a3', 'var')
    n_a = [n_a1, n_a2, n_a3]; % GI2A: [Safe, Future, Risky]
else
    n_a = [n_a1, n_a2];       % GI1: [Safe, Risky]
end

%% Transform Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% 1. Dynamically reconstruct the state-space dimensions
target_sz = n_a;
if N_z > 1, target_sz = [target_sz, n_z]; end
if N_e > 1, target_sz = [target_sz, vfoptions.n_e]; end
target_sz = [target_sz, N_j];

% 2. Reshape the Value Function
V = reshape(VKron, target_sz);

% 3. Un-Kronecker the Policy Function
has_z = (N_z > 1);
has_e = (N_e > 1);
has_d1 = (sum(n_d1) > 0);
has_d2 = (sum(n_d2) > 0);
has_d3 = (sum(n_d3) > 0);
has_a1 = (sum(n_a1) > 0); 
has_a2 = (length(n_a) > 2); % 3rd asset is risky, so > 2 means 2 standard assets

% Dynamically count the number of active choice variables
num_channels = has_d1 + has_d2 + has_d3 + has_a1 + has_a2; 

if has_e && has_z
    suffix = '_z_e';
elseif has_z || has_e
    suffix = '_z';
else
    suffix = '_noz';
end

base_fn = sprintf('UnKronPolicyIndexes%d_FHorz', num_channels);
UnKronFn = str2func([base_fn, suffix]);

% 4. Dynamically build the arguments list 
% (PolicyKron is ALREADY shrink-wrapped by the _raw helpers!)
PolicyKronSliced = PolicyKron; 

% Build the argument list based ONLY on active dimensions
args = {PolicyKronSliced};
if has_d1, args{end+1} = n_d1; end
if has_d2, args{end+1} = n_d2; end
if has_d3, args{end+1} = n_d3; end
if has_a1, args{end+1} = n_a1; end
if has_a2, args{end+1} = n_a2; end

args{end+1} = n_a; % The full combined asset grid size

if has_z && has_e
    args = [args, {n_z, vfoptions.n_e}];
elseif has_z
    args{end+1} = n_z;
elseif has_e
    args{end+1} = vfoptions.n_e;
end

args = [args, {N_j, vfoptions}];

% 5. Execute
Policy = UnKronFn(args{:});


end
