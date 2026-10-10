function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==1
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% Divide-and-conquer level1n setup (divide-and-conquer requires the standard endogenous state)
if N_a1==0
    error('Cannot use vfoptions.divideandconquer with riskyasset DC_GI if there is no standard endogenous state (N_a1==0)')
end

if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=floor(sqrt(n_a1(1)));
    if n_a1(1)<5
        error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a1(1)); % n_a1(1): level1n is scalar, and with two standard assets it is a1_1 that is divide-conquered

% Two standard endogenous assets -> the DC2A_GI2A raws.
if length(n_a1)>1
    if length(n_a1)>2
        error('riskyasset divideandconquer supports at most two standard endogenous assets')
    end
    % a1_grid holds both standard endogenous states (stacked); n_a2/a2_grid hold the riskyasset
    n_a3=n_a2;
    a3_grid=a2_grid;
    a2_grid=a1_grid(n_a1(1)+1:end);
    a1_grid=a1_grid(1:n_a1(1));
    n_a2=n_a1(2);
    n_a1=n_a1(1);
    if ~isfield(vfoptions,'level1n')
        vfoptions.level1n=floor(sqrt(n_a1));
    end
    vfoptions.level1n=min(vfoptions.level1n,n_a1); % level1n is scalar, and with two standard assets it is a1 that is divide-conquered
    % a is divided into a1 (first standard endogenous state, divide-conquered), a2 (second standard endogenous state, folded) and a3 (the riskyasset)
    if N_e==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end

    % Policy channels: d2, d3, a1prime (divide-conquered), a2prime (folded) [plus d1 at the
    % front when there is a d1], and then the two grid-interp-layer rows (L2 and L2flag) which
    % UnKronPolicyIndexes*_FHorz_* passes through unchanged when vfoptions.gridinterplayer==1.
else
    %% Dispatch
    if N_e == 0
        [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end

%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

%% Reconstruct full n_a array
if exist('n_a3', 'var')
    n_a = [n_a1, n_a2, n_a3]; % DC2A: [Safe, Future, Risky]
else
    n_a = [n_a1, n_a2];       % DC1: [Safe, Risky]
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
has_a2 = (length(n_a) >= 2); % Detect second standard asset

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

% 4. Dynamically build the arguments list and shrink-wrap PolicyKron
active_rows = [];
if has_d1, active_rows(end+1) = 1; end
if has_d2, active_rows(end+1) = 2; end
if has_d3, active_rows(end+1) = 3; end
if has_a1, active_rows(end+1) = 4; end
if has_a2, active_rows(end+1) = 5; end

if vfoptions.gridinterplayer == 1
    if has_a2
        active_rows = [active_rows, 6, 7]; % DC2A flags
    else
        active_rows = [active_rows, 5, 6]; % DC1 flags
    end
end

% Slice out only the active rows
slice_idx = repmat({':'}, 1, ndims(PolicyKron));
slice_idx{1} = active_rows;
PolicyKronSliced = PolicyKron(slice_idx{:});

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
