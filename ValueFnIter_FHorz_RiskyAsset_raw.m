function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d1=max(1,prod(n_d1(n_d1>0)));
N_d2=max(1,prod(n_d2(n_d2>0)));
N_d3=max(1,prod(n_d3(n_d3>0)));
N_d=N_d1*N_d2*N_d3;
N_a1=max(1,prod(n_a1(n_a1>0)));
N_a2=max(1,prod(n_a2(n_a2>0)));
N_a  = N_a1 * N_a2;
N_z =max(1,prod(n_z(n_z>0)));
N_u =max(1,prod(n_u(n_u>0)));

% (For _e_raw only, also add):
% N_e=max(1, prod(n_e(n_e>0)));

% For ReturnFn (d1 and d3 only)
n_d13 = [n_d1(n_d1 > 0), n_d3(n_d3 > 0)];
N_d13 = N_d1 * N_d3;
d13_grid = [d1_grid; d3_grid];

% For aprimeFn (d2 and d3)
n_d23 = [n_d2(n_d2 > 0), n_d3(n_d3 > 0)];
N_d23 = N_d2 * N_d3;
d23_grid = [d2_grid; d3_grid];

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_j,'gpuArray'); % d1, d2, d3 and a1prime

%%
u_grid=gpuArray(u_grid);

n_d13a1=[n_d1, n_d3, n_a1];
grid_d13a1=[d1_grid; d3_grid; a1_grid];
d13a1_gridvals=CreateGridvals(n_d13a1(n_d13a1>0), grid_d13a1,1);

n_a12=[n_a1, n_a2];
grid_a12=[a1_grid; a2_grid];
a12_gridvals=CreateGridvals(n_a12(n_a12>0), grid_a12,1);

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

aind=gpuArray(0:1:N_a-1);
zind = shiftdim(gpuArray(0:1:N_z-1)', -2); % [1,1,N_z]

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], n_z, d13a1_gridvals, a12_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        dindex=rem(maxindex-1,N_d1*N_d3)+1;
        Policy(1,:,:,N_j)=shiftdim(rem(dindex-1,N_d1)+1,-1);
        Policy(2,:,:,N_j)=1; % is meaningless anyway
        Policy(3,:,:,N_j)=shiftdim(ceil(dindex/N_d1),-1);
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindex/(N_d1*N_d3)),-1);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], special_n_z, d13a1_gridvals, a12_gridvals, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            dindex=rem(maxindex-1,N_d1*N_d3)+1;
            Policy(1,:,z_c,N_j)=shiftdim(rem(dindex-1,N_d1)+1,-1);
            Policy(2,:,z_c,N_j)=1; % is meaningless anyway
            Policy(3,:,z_c,N_j)=shiftdim(ceil(dindex/N_d1),-1);
            Policy(4,:,z_c,N_j)=shiftdim(ceil(maxindex/(N_d1*N_d3)),-1);
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
    % aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], n_z, d13a1_gridvals, a12_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z)

        EV=V_Jplus1.*shiftdim(pi_z_J(:,:,N_j)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension

        % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
        % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
        skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1))); % Note, probably just do this off of a2prime values
        aprimeProbs=repmat(a2primeProbs,N_a1,N_z);  % [N_d*N_a1,N_u]
        aprimeProbs(skipinterp)=0;
        aprimeProbs=reshape(aprimeProbs,[N_d23*N_a1,N_u,N_z]);

        % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        EV1=EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)); % (d,u,z), the lower aprime
        EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)); % (d,u,z), the upper aprime

        % Apply the aprimeProbs
        EV1=reshape(EV1,[N_d23*N_a1,N_u,N_z]).*aprimeProbs; % probability of lower grid point
        EV2=reshape(EV2,[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbs); % probability of upper grid point
        EV1(isnan(EV1))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so the term contributes nothing
        EV2(isnan(EV2))=0;

        % Expectation over u (using pi_u), and then add the lower and upper
        EV=sum((EV1.*pi_u'),2)+sum((EV2.*pi_u'),2); % (d&a1prime,1,z), sum over u
        % EV is over (d&a1prime,1,z)

        % Time to refine
        % First: ReturnMatrix, we can refine out d1
        [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix,[N_d1,N_d3*N_a1,N_a,N_z]),[],1);
        % Second: EV, we can refine out d2
        [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_z]),[],1);
        % Now put together entireRHS, which just depends on d3
        entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(3,:,:,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy(1,:,:,N_j)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*zind),1);
        Policy(2,:,:,N_j)=shiftdim(d2index(maxindex+N_d3*zind),1);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], special_n_z, d13a1_gridvals, a12_gridvals, z_val, ReturnFnParamsVec);

            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=V_Jplus1.*pi_z_J(z_c,:,N_j);
            EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_z=sum(EV_z,2);

            % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
            % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbs to zero)
            skipinterp=logical(EV_z(aprimeIndex)==EV_z(aprimeplus1Index)); % Note, probably just do this off of a2prime values
            aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
            aprimeProbs(skipinterp)=0;

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1_z=aprimeProbs.*reshape(EV_z(aprimeIndex),[N_d23*N_a1,N_u]); % (d,u), the lower aprime
            EV2_z=(1-aprimeProbs).*reshape(EV_z(aprimeplus1Index),[N_d23*N_a1,N_u]); % (d,u), the upper aprime
            EV1_z(isnan(EV1_z))=0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so the term contributes nothing
            EV2_z(isnan(EV2_z))=0;
            % Already applied the probabilities from interpolating onto grid

            % Expectation over u (using pi_u), and then add the lower and upper
            EV_z=sum((EV1_z.*pi_u'),2)+sum((EV2_z.*pi_u'),2); % (d&a1prime,u), sum over u
            % EV_z is over (d&a1prime,1)

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_z,[N_d1,N_d3*N_a1,N_a]),[],1);
            % Second: EV, we can refine out d2
            [EV_onlyd3,d2index]=max(reshape(EV_z,[N_d2,N_d3*N_a1,1]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS_z=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(3,:,z_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
            Policy(4,:,z_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
            Policy(1,:,z_c,N_j)=shiftdim(d1index(maxindex+N_d3*N_a1*aind),1);
            Policy(2,:,z_c,N_j)=shiftdim(d2index(maxindex),1);
        end
    end
end



%% Iterate backwards through j
for jj=N_j:-1:1
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n', jj,N_j)
    end
    
    % 1. Prepare Parameters for period jj
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    is_terminal=(jj==N_j) && ~isfield(vfoptions, 'V_Jplus1');
    
    if ~is_terminal
        DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
        
        % Setup EV base
        if jj==N_j
            EV_base=reshape(vfoptions.V_Jplus1, [N_a,N_z]);
        else
            EV_base=V(:,:, jj+1);
        end
        
        % Create standard aprime indices (Strictly (d23*a1, u) - NO z expansion!)
        [a2primeIndex, a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec, 2);
        
        aprimeIndex=repelem((1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
        aprimeplus1Index=repelem((1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);
        
        % Base probability matrix (d23*a1, u)
        baseProbs=repmat(a2primeProbs,N_a1,1);
    end
    
    % 2. Setup Evaluation Blocks based on lowmemory
    if vfoptions.lowmemory==0
        z_iter=1;
        special_n_z=n_z;
    else
        z_iter=1:N_z;
        special_n_z=ones(1, length(n_z));
    end
    
    % 3. Core Evaluation Loop
    for z_c=z_iter
        if vfoptions.lowmemory==0
            z_val=z_gridvals_J(:,:, jj);
            z_idx=1:N_z;
            z_offset=zind; % For indexing policy correctly
        else
            z_val=z_gridvals_J(z_c,:, jj);
            z_idx=z_c;
            z_offset=0; % Block is a single slice
        end
        
        % A. Evaluate ReturnMatrix for this block
        ReturnMatrix_block=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d13, n_a1], [n_a1, n_a2], special_n_z, d13a1_gridvals, a12_gridvals, z_val, ReturnFnParamsVec);
        [ReturnMatrix_onlyd3, d1index]=max(reshape(ReturnMatrix_block, [N_d1,N_d3*N_a1,N_a, length(z_idx)]), [],1);
        
        if is_terminal
            % Terminal Period Logic (No Continuation Value)
            entireRHS=shiftdim(ReturnMatrix_onlyd3,1);
            [Vtemp, maxindex]=max(entireRHS, [],1);
            
            V(:, z_idx, jj)=shiftdim(Vtemp,1);
            Policy(3,:, z_idx, jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
            Policy(4,:, z_idx, jj)=shiftdim(ceil(maxindex/N_d3), -1);
            Policy(1,:, z_idx, jj)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*z_offset),1);
            Policy(2,:, z_idx, jj)=1; % d2 is meaningless without continuation
        else
            % B. Compute Continuation Value (EV)
            if vfoptions.lowmemory==0
                EV_z=EV_base .* shiftdim(pi_z_J(:,:, jj)', -1);
            else
                EV_z=EV_base .* pi_z_J(z_c,:, jj);
            end
            EV_z(isnan(EV_z))=0;
            EV_z=sum(EV_z, 2); 
            
            % Interpolation check mapped onto active slice
            skipinterp=logical(EV_z(aprimeIndex(:)+N_a*((1:length(z_idx))-1))==EV_z(aprimeplus1Index(:)+N_a*((1:length(z_idx))-1)));
            
            % Expand probabilities locally for this block ONLY
            blockProbs=repmat(baseProbs,1, length(z_idx));
            blockProbs(skipinterp)=0;
            blockProbs=reshape(blockProbs, [N_d23*N_a1,N_u, length(z_idx)]);
            
            % Extract EV bounds and apply U-shocks
            EV1=reshape(EV_z(aprimeIndex(:)+N_a*((1:length(z_idx))-1)), [N_d23*N_a1,N_u, length(z_idx)]) .* blockProbs;
            EV2=reshape(EV_z(aprimeplus1Index(:)+N_a*((1:length(z_idx))-1)), [N_d23*N_a1,N_u, length(z_idx)]) .* (1 - blockProbs);
            
            EV1(isnan(EV1))=0; EV2(isnan(EV2))=0;
            EV_block=sum((EV1 .* pi_u'), 2)+sum((EV2 .* pi_u'), 2);
            
            % Refine d2 out of Continuation Value
            [EV_onlyd3, d2index]=max(reshape(DiscountFactorParamsVec*EV_block, [N_d2,N_d3*N_a1,1, length(z_idx)]), [],1);
            
            % C. Maximize total RHS
            entireRHS=shiftdim(ReturnMatrix_onlyd3+EV_onlyd3,1);
            [Vtemp, maxindex]=max(entireRHS, [],1);
            
            V(:, z_idx, jj)=shiftdim(Vtemp,1);
            Policy(3,:, z_idx, jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
            Policy(4,:, z_idx, jj)=shiftdim(ceil(maxindex/N_d3), -1);
            Policy(1,:, z_idx, jj)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*z_offset),1);
            Policy(2,:, z_idx, jj)=shiftdim(d2index(maxindex+N_d3*z_offset),1);
        end
    end
end

%% Shrink-wrap Policy to remove inactive choice dimensions
% The raw evaluator statically allocates 4 rows: [d1, d2, d3, a1prime]
% We must dynamically strip the dummy rows before returning.
has_d1 = (sum(n_d1) > 0);
has_d2 = (sum(n_d2) > 0);
has_d3 = (sum(n_d3) > 0);
has_a1 = (sum(n_a1(1)) > 0); 

active_rows = [];
if has_d1, active_rows(end+1) = 1; end
if has_d2, active_rows(end+1) = 2; end
if has_d3, active_rows(end+1) = 3; end
if has_a1, active_rows(end+1) = 4; end

% Slice out only the active rows
slice_idx = repmat({':'}, 1, ndims(Policy));
slice_idx{1} = active_rows;
Policy = Policy(slice_idx{:});


end
