function varargout=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Typically, varargout=[V, Policy]

V=nan;
Policy=nan;


%% Check which vfoptions have been used, set all others to defaults
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    % If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0; % =1 print out feedback on what is happening internally
    vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    vfoptions.gridinterplayer=0; % Interpolate between grid points (not yet implemented for alternative preferences)
    vfoptions.lowmemory=0; % use more loops and less parallelization, reduce memory use but at the cost of slower runtimes
    % Alternative model setups
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    vfoptions.exoticpreferences='None';
    vfoptions.dynasty=0;
    vfoptions.experienceasset=0;
    vfoptions.experienceassetu=0;
    vfoptions.experienceassete=0;
    vfoptions.experienceassetz=0;
    vfoptions.experienceassetze=0;
    vfoptions.experienceassetsemiz=0;
    vfoptions.riskyasset=0;
    vfoptions.residualasset=0;
    vfoptions.n_ambiguity=0;
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
    % Largely just for internal use only
    vfoptions.parallel=1+(gpuDeviceCount>0);
    % When calling as a subcommand, the following are used internally
    vfoptions.outputkron=0; % If 1 then leave output in Kron form
    vfoptions.alreadygridvals=0; % =1 when calling as a subcommand
    vfoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
else
    % Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0; % =1 Interpolate between grid points (not yet implemented for most cases)
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1*0 you must set vfoptions.ngridinterp (number of points to interpolate for aprime between each consecutive pair of points in a_grid)')
        end
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    % Alternative model setups
    if ~isfield(vfoptions,'incrementaltype')
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if ~isfield(vfoptions,'dynasty')
        vfoptions.dynasty=0;
    end
    if ~isfield(vfoptions,'experienceasset')
        vfoptions.experienceasset=0;
    end
    if ~isfield(vfoptions,'experienceassetu')
        vfoptions.experienceassetu=0;
    end
    if ~isfield(vfoptions,'experienceassete')
        vfoptions.experienceassete=0;
    end
    if ~isfield(vfoptions,'experienceassetz')
        vfoptions.experienceassetz=0;
    end
    if ~isfield(vfoptions,'experienceassetze')
        vfoptions.experienceassetze=0;
    end
    if ~isfield(vfoptions,'experienceassetsemiz')
        vfoptions.experienceassetsemiz=0;
    end
    if ~isfield(vfoptions,'riskyasset')
        vfoptions.riskyasset=0;
    end
    if ~isfield(vfoptions,'residualasset')
        vfoptions.residualasset=0;
    end
    if ~isfield(vfoptions,'n_ambiguity')
        vfoptions.n_ambiguity=0;
    end
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
    % Largely just for internal use only
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    % When calling as a subcommand, the following are used internally
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(vfoptions,'alreadygridvals')
        vfoptions.alreadygridvals=0; % =1 when calling as a subcommand
    end
    if ~isfield(vfoptions,'alreadygridvals_semiexo')
        vfoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    end
end

if isempty(n_d)
    error('If you have no d (decision) variables, set n_d=0;')
end
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~all(size(d_grid)==[sum(n_d), 1])
    if ~isempty(n_d) % Make sure d is being used before complaining about size of d_grid
        if n_d~=0
            error('d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
        end
    end
end
if ~all(size(a_grid)==[sum(n_a), 1])
    error('a_grid is not the correct shape (should be of size sum(n_a)-by-1; a (stacked) column vector)')
end

%% z_grid/pi_z/e_grid/pi_e shape validation is performed inside ExogShockSetup_FHorz (called below).

if vfoptions.parallel<2
    if N_e>0
        error('Sorry but e (i.i.d) variables are not implemented for cpu, you will need a gpu to use them')
    end
    if prod(vfoptions.n_semiz)>0
        error('Sorry but Semi-Exogenous states are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.divideandconquer==0
        error('Sorry but divideandconquer is not implemented for cpu, you will need a gpu to use this algorithm')
    end
    if ~strcmp(vfoptions.exoticpreferences,'None')
        error('Sorry but exoticpreferences are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.experienceasset==0 || ~vfoptions.experienceassetu==0  || ~vfoptions.experienceassetz==0  || ~vfoptions.experienceassete==0  || ~vfoptions.experienceassetze==0  || ~vfoptions.experienceassetsemiz==0
        error('Sorry but experience assets are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.riskyasset==0
        error('Sorry but riskyasset are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.residualasset==0
        error('Sorry but residualasset are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.dynasty==0
        error('Sorry but dynasty are not implemented for cpu, you will need a gpu to use them')
    end
end


%%
if vfoptions.parallel==2
    % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    z_grid=gpuArray(z_grid);
    pi_z=gpuArray(pi_z);
    if size(d_grid,2)==1
        d_gridvals=CreateGridvals(n_d,d_grid,1);
    else % already d_gridvals
        d_gridvals=d_grid;
    end
else
    % CPU can be used, but only for the basics. Is kept separate here so that the rest of the codes can just assume you have GPU and work with it.
    [V,Policy]=ValueFnIter_FHorz_CPU(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V,Policy};
    return
end


%% Exogenous shock gridvals and pi
if vfoptions.alreadygridvals==0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3,0);
    % output: z_gridvals_J, pi_z_J, vfoptions.e_gridvals_J, vfoptions.pi_e_J
    %
    % size(z_gridvals_J)=[prod(n_z),length(n_z),N_j]
    % size(pi_z_J)=[prod(n_z),prod(n_z),N_j-1], last dim is N_j if vfoptions.V_Jplus1 is used
    % size(e_gridvals_J)=[prod(n_e),length(n_e),N_j]
    % size(pi_e_J)=[prod(n_e),N_j], last dim is N_j+1 if vfoptions.V_Jplus1 is used
    % If no z, then z_gridvals_J=[] and pi_z_J=[]
    % If no e, then e_gridvals_J=[] and pi_e_J=[]
    %
    % Check pi_z_J: that all elements are positive, and that the rows sum to one
    if N_z>0
        if ~all(pi_z_J(:)>=0)
            warning('Some elements of pi_z_J, the exogenous markov transition probabilities matrix, are negative')
        end
        if ~all(all(squeeze(sum(pi_z_J,2))-1 <1e-13)) % sum of row must be within 1e-13 of 1
            warning('Some rows of pi_z_J, the exogenous markov transition probabilities matrix, do not sum to one')
        end
    end
    if isfield(vfoptions,'pi_e_J')
        if ~all(vfoptions.pi_e_J(:)>=0)
            warning('Some elements of pi_e_J, the exogenous i.i.d. probabilities matrix, are negative')
        end
        if ~all(squeeze(sum(vfoptions.pi_e_J,1))-1 <1e-13) % sum of row must be within 1e-13 of 1
            warning('Some columns of pi_e_J, the exogenous i.i.d. probabilities matrix, do not sum to one')
        end
    end
elseif vfoptions.alreadygridvals==1
    z_gridvals_J=z_grid;
    pi_z_J=pi_z;
end


%% Semi-exogenous shock gridvals and pi
if vfoptions.alreadygridvals_semiexo==0
    if prod(vfoptions.n_semiz)>0
        % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
        vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
        % output: vfoptions.semiz_gridvals_J, vfoptions.pi_semiz_J
        % size(semiz_gridvals_J)=[prod(n_z),length(n_z),N_j]
        % size(pi_semiz_J)=[prod(n_semiz),prod(n_semiz),prod(n_dsemiz),N_j]
        % If no semiz, then vfoptions just does not contain these field
    end
end

%% Implement new way of handling ReturnFn inputs
if isempty(ReturnFnParamNames)
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
end
% Basic setup: the first inputs of ReturnFn will be (d,aprime,a,z,..) and everything after this is a parameter, so we get the names of all these parameters.
% But this changes if you have e, semiz, or just multiple d, and if you use riskyasset, expasset, etc.
% So figure out which setup we have, and get the relevant ReturnFnParamNames


%% Implement new way of handling warm-glow of bequests (currently only used by Epstein-Zin preferences)
if isfield(vfoptions,'WarmGlowBequestsFn')
    l_a=length(n_a);
    temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
    vfoptions.WarmGlowBequestsFnParamsNames={temp{l_a+1:end}}; % the first inputs will always be aprime
end
% clear l_d l_a l_z l_e % These are all messed up so make sure they are not reused later


%% Print out vfoptions (if vfoptions.verbose=1)
if vfoptions.verbose>=1
    vfoptions
end

%% Split n_d/n_a for whichever non-standard endogenous state is in use (experience assets, riskyasset, residualasset)
% Does nothing when none of them is in use. The splits are unpacked from vfoptions at the point of use below.
vfoptions=SetupNonStandardEndoStates_FHorz(n_d,n_a,d_grid,a_grid,vfoptions);

%% Top-level dispatch: on the preferences
% Each exotic preference has a single dispatcher, which handles every asset type it supports and
% returns. Everything after this block is therefore the exoticpreferences='None' path.
% First a whitelist check on the string: an unrecognized name would otherwise fall through the
% whole dispatch and silently solve the standard (exponential discounting) problem.
if ~any(strcmp(vfoptions.exoticpreferences,{'None','QuasiHyperbolic','EpsteinZin','GulPesendorfer','AmbiguityAversion'}))
    error(['vfoptions.exoticpreferences=',vfoptions.exoticpreferences,' is not a recognized option (must be one of None, QuasiHyperbolic, EpsteinZin, GulPesendorfer, AmbiguityAversion; check spelling and capital letters)'])
end
if strcmp(vfoptions.exoticpreferences,'None')
    % Nothing to dispatch on: fall through to the asset-type blocks below.
elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    % All of quasi-hyperbolic now goes through this one dispatcher, including the experience-asset
    % variants: it reads beta0 once and then dispatches on the asset type.
    [V,Policy, Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The dispatcher always returns four outputs (Policyalt is [] under Sophisticated). Trimming to
    % three here is deliberate: under Sophisticated there is no Policyalt, so a caller asking for one
    % should get an error rather than a silent empty. Hence this is not folded into the dispatcher.
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        varargout={V, Policy,Valt,Policyalt}; % Vtilde, Policytilde, V, Policy (the last two are the exponential discounter)
    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        varargout={V, Policy,Valt}; % Vhat, Policyhat, Vunderbar (the last is the exponential discounter V from the Policyhat)
    end
    return
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
    % All of Epstein-Zin now goes through this one dispatcher, including riskyasset:
    % it dispatches on the asset type before setting up the Epstein-Zin constants.
    [V, Policy]=ValueFnIter_FHorz_EpsteinZin(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif strcmp(vfoptions.exoticpreferences,'GulPesendorfer')
    [V, Policy]=ValueFnIter_FHorz_GulPesendorfer(n_d,n_a,n_z,N_j,d_gridvals,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
elseif strcmp(vfoptions.exoticpreferences,'AmbiguityAversion')
    [V, Policy]=ValueFnIter_FHorz_AmbiguityAversion(n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions); % d_grid (not d_gridvals): the riskyasset branch splits it by refine_d
    varargout={V, Policy};
    return
end


%% Experience asset (exoticpreferences='None'; the exotic preferences handle their own asset types)
% experienceasset: aprime(d,a)
% experienceassetu: aprime(d,a,u)
% experienceassetz: aprime(d,a,z)
% experienceassete: aprime(d,a,e)
% experienceassetze: aprime(d,a,z,e)

if vfoptions.experienceasset>=1 || vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
    % The split itself is done in SetupNonStandardEndoStates_FHorz (called above); unpack it here.
    n_d1=vfoptions.n_d1;
    n_d2=vfoptions.n_d2;
    n_a1=vfoptions.n_a1;
    n_a2=vfoptions.n_a2;
    d1_grid=vfoptions.d1_grid;
    d2_grid=vfoptions.d2_grid;
    a1_grid=vfoptions.a1_grid;
    a2_grid=vfoptions.a2_grid;
    if prod(vfoptions.n_semiz)>0
        n_d3=vfoptions.n_d3;
        d3_grid=vfoptions.d3_grid;
    end

    % Now just send all this to the right value fn iteration command
    if vfoptions.experienceasset>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAssetSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAsset(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassetu>=1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAssetuSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAssetu(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassete>=1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAsseteSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAssete(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassetz>=1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAssetzSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAssetz(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassetze>=1
        if prod(vfoptions.n_semiz)>0
            [V,Policy]=ValueFnIter_FHorz_ExpAssetzeSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [V,Policy]=ValueFnIter_FHorz_ExpAssetze(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.experienceassetsemiz>=1
        % semiz is always present (it drives the experience asset); aprimeFn=aprimeFn(d2,a2,semiz)
        if prod(vfoptions.n_semiz)==0
            error('When using experienceassetsemiz you must set vfoptions.n_semiz (the semi-exogenous state drives the experience asset)')
        end
        [V,Policy]=ValueFnIter_FHorz_ExpAssetsemiz(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end

    varargout={V, Policy};
    return
end


%% Risky asset (exoticpreferences='None'; Epstein-Zin handles riskyasset in its own dispatcher)
if vfoptions.riskyasset==1
    % The split itself is done in SetupNonStandardEndoStates_FHorz (called above); unpack it here.
    n_a1=vfoptions.n_a1;
    n_a2=vfoptions.n_a2;
    a1_grid=vfoptions.a1_grid;
    a2_grid=vfoptions.a2_grid;

    % Now just send all this to the right value fn iteration command
    if prod(vfoptions.n_semiz)>0
        [V, Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo(n_d,n_a1,n_a2,vfoptions.n_semiz,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, vfoptions.semiz_gridvals_J,z_gridvals_J, vfoptions.u_grid, vfoptions.pi_semiz_J, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [V,Policy]=ValueFnIter_FHorz_RiskyAsset(n_d,n_a1,n_a2,n_z,vfoptions.n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.u_grid, pi_z_J, vfoptions.pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end

    varargout={V, Policy};
    return
end

%% Residual asset (exoticpreferences='None'; no exotic preference supports residualasset)
if vfoptions.residualasset==1
    % Note: this block returns before divideandconquer and gridinterplayer are ever consulted, so
    % they have to be caught here. Silently ignoring them (the previous behaviour) handed the user
    % a brute-force solve while they believed they had asked for DC or GI.
    if vfoptions.divideandconquer==1
        error('divide-and-conquer is not yet implemented for residual assets (only the base solver exists)')
    end
    if vfoptions.gridinterplayer==1
        error('grid interpolation layer is not yet implemented for residual assets (only the base solver exists)')
    end
    % The split itself is done in SetupNonStandardEndoStates_FHorz (called above); unpack it here.
    n_a1=vfoptions.n_a1;
    n_r=vfoptions.n_r;
    a1_grid=vfoptions.a1_grid;
    r_grid=vfoptions.r_grid;

    % Now just send all this to the right value fn iteration command
    [V,Policy]=ValueFnIter_FHorz_ResidAsset(n_d,n_a1,n_r,n_z, N_j, d_grid, a1_grid, r_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V, Policy};
    return
end

%% Deal with StateDependentVariables_z if need to do that.
if isfield(vfoptions,'StateDependentVariables_z')==1
    if vfoptions.verbose==1
        fprintf('StateDependentVariables_z option is being used \n')
    end

    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_FHorz_nod_SDVz_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % Policy without d
        PolicyKron=shiftdim(PolicyKron,-1);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_SDVz_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end

    if vfoptions.outputkron==1
        varargout={VKron, PolicyKron};
        return
    end

    if N_d==0
        n_daprime=n_a;
    else
        n_daprime=[n_d,n_a];
    end

    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    varargout={V, Policy};
    return
end

%% Deal with dynasty if need to do that.
if vfoptions.dynasty==1
    if vfoptions.verbose==1
        fprintf('dynasty option is being used \n')
    end
    if ~isfield(vfoptions,'tolerance')
        vfoptions.tolerance=10^(-9);
    end

    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_FHorz_Dynasty_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % Policy without d
        PolicyKron=shiftdim(PolicyKron,-1);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_Dynasty_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end

    if vfoptions.outputkron==1
        varargout={VKron, PolicyKron};
        return
    end

    if N_d==0
        n_daprime=n_a;
    else
        n_daprime=[n_d,n_a];
    end

    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    varargout={V, Policy};
    return
end



%% Semi-exogenous state
% The transition matrix of the exogenous shocks depends on the value of the 'last' decision variable(s).
if prod(vfoptions.n_semiz)>0
    if length(n_d)>vfoptions.l_dsemiz
        n_d1=n_d(1:end-vfoptions.l_dsemiz);
        d1_grid=d_grid(1:sum(n_d1));
    else
        n_d1=0; d1_grid=[];
    end
    n_d2=n_d(end-vfoptions.l_dsemiz+1:end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
    d2_grid=d_grid(sum(n_d1)+1:end);

    d1_gridvals=CreateGridvals(n_d1,d1_grid,1);
    d2_gridvals=CreateGridvals(n_d2,d2_grid,1);

    % Now that we have pi_semiz_J we are ready to compute the value function.
    % profile clear
    % profile on
    tic;
    [V_ref,Policy_ref]=ValueFnIter_FHorz_SemiExo(n_d1,n_d2,n_a,vfoptions.n_semiz,n_z,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    time_ref=toc;
    % profile off
    % profile viewer

    %% GPU Solve
    % profile clear
    % profile on
    tic;
    [V_new, Policy_new] = ValueFnIter_FHorz_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    time_new=toc;
    % profile off
    % profile viewer

    benchcheck();
    varargout={V_ref,Policy_ref};
    return
end

%% Just do the standard case
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    % Solve by doing Divide-and-Conquer, and then a grid interpolation layer
    tic;
    [V_ref,Policy_ref]=ValueFnIter_FHorz_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    time_ref=toc;
    tic;
    [V_new,Policy_new]=ValueFnIter_FHorz_DC_GI_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    time_new=toc;
    benchcheck();
    varargout={V_ref,Policy_ref};
    return
elseif vfoptions.divideandconquer==1
    % Solve using Divide-and-Conquer algorithm
    tic;
    [V_ref,Policy_ref]=ValueFnIter_FHorz_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    time_ref=toc;
    tic;
    [V_new,Policy_new]=ValueFnIter_FHorz_DC_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    time_new=toc;
    benchcheck();
    varargout={V_ref,Policy_ref};
    return
elseif vfoptions.gridinterplayer==1
    % Solve using grid interpolation layer
    tic;
    [V_ref,Policy_ref]=ValueFnIter_FHorz_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    time_ref=toc;
    tic;
    [V_new,Policy_new]=ValueFnIter_FHorz_GI_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    time_new=toc;
    benchcheck();
    varargout={V_ref,Policy_ref};
    return
end

% vfoptions.parallel==2 % GPU

% profile clear
% profile on
tic;
if N_e==0
    if N_z==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_z==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end
time_ref=toc;
% profile off
% profile viewer

%% GPU Solve
% profile clear
% profile on
tic;
[VKron_new, PolicyKron_new] = ValueFnIter_FHorz_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
time_new=toc;
% profile off
% profile viewer

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    varargout={VKron, PolicyKron};
    return
end

if N_d==0
    n_daprime=n_a;
else
    n_daprime=[n_d,n_a];
end

if N_e==0
    if N_z==0
        V_ref=reshape(VKron,[n_a,N_j]);
        Policy_ref=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_daprime,n_a,N_j,vfoptions);
    else
        V_ref=reshape(VKron,[n_a,n_z,N_j]);
        Policy_ref=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    end
else
    if N_z==0
        V_ref=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy_ref=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
    else
        V_ref=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy_ref=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    end
end

if N_e==0
    if N_z==0
        V_new=reshape(VKron_new,[n_a,N_j]);
        Policy_new=UnKronPolicyIndexes1_FHorz_noz(PolicyKron_new,n_daprime,n_a,N_j,vfoptions);
    else
        V_new=reshape(VKron_new,[n_a,n_z,N_j]);
        Policy_new=UnKronPolicyIndexes1_FHorz_z(PolicyKron_new,n_daprime,n_a,n_z,N_j,vfoptions);
    end
else
    if N_z==0
        V_new=reshape(VKron_new,[n_a,vfoptions.n_e,N_j]);
        Policy_new=UnKronPolicyIndexes1_FHorz_z(PolicyKron_new,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
    else
        V_new=reshape(VKron_new,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy_new=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron_new,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    end
end

benchcheck();

varargout={V_ref, Policy_ref};


    function benchcheck()

        % --- AUTOMATED POLICY MISMATCH MASKING ---
        % Identify all valid states where the agent can actually survive (V_ref > -Inf)
        valid_states = (V_ref > -Inf);

        % Check if policies match ONLY on valid, reachable states
        if any(Policy_new(:, valid_states) ~= Policy_ref(:, valid_states))
            sum(Policy_new(:) ~= Policy_ref(:))
            idx = find(Policy_new(:) ~= Policy_ref(:), 1); [r, c, p, q, t] = ind2sub(size(Policy_new), idx); disp(['Mismatch at [', num2str([r,c,p,q,t]), ']']);
            error('Policy_new ~= Policy_ref on reachable states');
        elseif any(Policy_new(:) ~= Policy_ref(:))
            disp('Policy check passed on all reachable states! (Ignored dead -Inf states)');
        end

        tol = 1e-3; % Could need to loosen as necessary

        % Mask out non-finite states in the reference solution
        valid_mask = isfinite(V_ref(:));

        % Option A: Mean absolute level of finite states (The Scale Benchmark)
        ref_scale = mean(abs(V_ref(valid_mask)));

        % Option B: Dynamic Range Benchmark (invariant to constant shifts in utility)
        % ref_scale = max(V_ref(valid_mask)) - min(V_ref(valid_mask));

        % Safe relative difference across all reachable states
        diff_vec = abs(V_new(valid_mask) - V_ref(valid_mask));
        rel_diff = max(diff_vec) / max(ref_scale, eps);

        if rel_diff > tol
            % Is this a floating-point rounding difference or a massive math failure?
            max_diff = max(abs(V_new(:) - V_ref(:)));
            disp(['Maximum absolute difference: ', num2str(max_diff)]);
            fprintf('Legacy NaNs: %d | Legacy -Infs: %d\n', sum(isnan(V_ref(:))), sum(V_ref(:) == -Inf));
            fprintf('Tensor NaNs: %d | Tensor -Infs: %d\n', sum(isnan(V_new(:))), sum(V_new(:) == -Inf));

            % Find the exact linear coordinate of the biggest difference
            [~, bad_idx] = max(abs(V_new(:) - V_ref(:)));

            % Dynamically resolve the N-dimensional coordinates using a cell array
            ndim_V = ndims(V_new);
            coords = cell(1, ndim_V);
            [coords{:}] = ind2sub(size(V_new), bad_idx);

            % Format the coordinates for display
            coord_str = sprintf('%d, ', cell2mat(coords));
            coord_str = coord_str(1:end-2); % Remove trailing comma and space

            fprintf('Worst mismatch at -> ND-Coords: [%s] (Linear Idx: %d)\n', coord_str, bad_idx);

            % Show the actual values and signed difference (New - Ref)
            val_ref = V_ref(bad_idx);
            val_new = V_new(bad_idx);
            signed_diff = val_new - val_ref;

            fprintf('Legacy V: %f\n', val_ref);
            fprintf('Tensor V: %f\n', val_new);
            fprintf('Signed Difference (Tensor - Legacy): %f\n', signed_diff);

            if signed_diff > 0
                warning('>>> TENSOR FOUND A HIGHER VALUE (Better optimization peak) <<<');
            else
                error('>>> LEGACY FOUND A HIGHER VALUE (Tensor missed the peak) <<<');
            end

            % error("V_new ~= V_ref");
        end

        xx=dbstack();
        filenames={xx.file};
        % The caller of this function is the test case we are measuring, or caller of caller if called via _Ptype
        fprintf('%s -- time reference: %.2f seconds; time difference: %.2f seconds; time ratio to ref: %.0f%%\n', ...
            filenames{4-isempty(strfind(filenames{3}, 'Horz_PType'))}, time_ref, time_new-time_ref, 100*time_new/time_ref);


    end


end
