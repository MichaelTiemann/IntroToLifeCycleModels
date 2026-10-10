function [V, Policy]=ValueFnIter_FHorz_RiskyAsset(n_d,n_a1,n_a2,n_z,n_u,N_j,d_grid, a1_grid,a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a1=prod(n_a1);
N_z=prod(n_z);

%% Get aprimeFnParamNames
l_d=length(n_d); % because it is a risky asset there must be some decision variables
if isfield(vfoptions,'refine_d')
    l_d=l_d-vfoptions.refine_d(1);
    if length(vfoptions.refine_d)==4 % only relevant if using semiz
        l_d=l_d-vfoptions.refine_d(4);
    end
end
l_u=length(n_u);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d+l_u)
    aprimeFnParamNames={temp{l_d+l_u+1:end}}; % the first inputs will always be (d,u)
else
    aprimeFnParamNames={};
end

%%
% Make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_u=gpuArray(pi_u);
u_grid=gpuArray(u_grid);
% Check pi_u and u_grid are the right size
if all(size(pi_u)==[prod(n_u),1])
    % good
elseif all(size(pi_u)==[1,prod(n_u)])
    error('pi_u should be a column vector (it is a row vector, you need to transpose it')
else
    error('pi_u is the wrong size (it should be a column vector of size prod(n_u)-by-1)')
end
if all(size(u_grid)==[prod(n_u),1])
    % good
elseif all(size(u_grid)==[1,prod(n_u)])
    error('u_grid should be a column vector (it is a row vector, you need to transpose it')
else
    error('u_grid is the wrong size (it should be a column vector of size prod(n_u)-by-1)')
end


%% Solve based on which setup we have
if length(n_a2)>1
    error('Have not yet implemented riskyasset for more than one riskyasset')
end
N_e=prod(vfoptions.n_e);

if isfield(vfoptions,'refine_d')
    if vfoptions.refine_d(1)==0
        N_d1=0;
    else
        N_d1=prod(n_d(1:vfoptions.refine_d(1)));
    end
else
    error('When using vfoptions.riskyasset you should also set vfoptions.refine_d')
end

if sum(vfoptions.refine_d)~=length(n_d)
    error('vfoptions.refine_d seems to be set up wrong, it is inconsistent with n_d')
end

% Note: We have explicitly removed the legacy checks that delete refine_d 
% or throw errors if d2==0. refine_d = [0,0,X] is completely valid.

if vfoptions.refine_d(1)>0
    n_d1=n_d(1:vfoptions.refine_d(1));
else
    n_d1=0;
end
if vfoptions.refine_d(2)>0
    n_d2=n_d(vfoptions.refine_d(1)+1:vfoptions.refine_d(1)+vfoptions.refine_d(2));
else
    n_d2=0;
end
if vfoptions.refine_d(3)>0
    n_d3=n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1:end);
else
    n_d3=0;
end
d1_grid=d_grid(1:sum(n_d1));
d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:end);

%% Dispatch
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.divideandconquer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
end

if N_e==0
    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
else
    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

if prod(n_a1)>0
    n_a=[n_a1,n_a2];
else
    n_a=n_a2;
end

%% Transform Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% 1. Dynamically reconstruct the state-space dimensions
% Filter out empty/zero dimensions to build the exact target size
target_sz = n_a; % a1 and a2 combined
if N_z > 1, target_sz = [target_sz, n_z]; end
if N_e > 1, target_sz = [target_sz, vfoptions.n_e]; end
target_sz = [target_sz, N_j]; % Time dimension is always last

% 2. Reshape the Value Function
V = reshape(VKron, target_sz);

% 3. Extract Policy dynamically
has_z = (N_z > 1);
has_e = (N_e > 1);
has_d1 = (sum(n_d1) > 0);
has_d2 = (sum(n_d2) > 0);
has_d3 = (sum(n_d3) > 0);
has_a1 = (sum(n_a1) > 0); 

% Pure raw (no DC, no GI) ALWAYS Kroneckers all standard assets into 'a1prime'.
% There is NO separate 'a2prime' choice passed to UnKron here!
num_channels = has_d1 + has_d2 + has_d3 + has_a1;

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
% Note: PolicyKron is already perfectly shrink-wrapped by the _raw functions!
% We do NOT slice it again here.
args = {PolicyKron};
if has_d1, args{end+1} = n_d1; end
if has_d2, args{end+1} = n_d2; end
if has_d3, args{end+1} = n_d3; end
if has_a1, args{end+1} = n_a1; end

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
