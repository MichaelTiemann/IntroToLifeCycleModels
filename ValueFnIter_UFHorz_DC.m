function [V,Policy]=ValueFnIter_UFHorz_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'level1n')
    if isscalar(n_a)
        vfoptions.level1n=floor(sqrt(n_a(1)));
        if n_a(1)<5; error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable'); end
    elseif length(n_a)==2
        vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch
        if n_a(1)<5; error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable'); end
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
else
    if ~isscalar(n_a) && isscalar(vfoptions.level1n)
        vfoptions.level1n=[vfoptions.level1n,n_a(2:end)]; % user only needs to declare level1n for first dimension.
    end
end

%% 1 endogenous state
if isscalar(n_a)
    [VKron, PolicyKron] = ValueFnIter_UFHorz_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

%% 2 endogenous states
elseif length(n_a)>=2
    if length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a(2)
            vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2A
        else
            error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
        end
    end
    [VKron, PolicyKron] = ValueFnIter_UFHorz_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V=VKron; Policy=PolicyKron; return
end

if N_d==0; n_daprime=n_a; else; n_daprime=[n_d,n_a]; end

if N_e==0
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_daprime,n_a,N_j,vfoptions);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    end
else
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    end
end


end