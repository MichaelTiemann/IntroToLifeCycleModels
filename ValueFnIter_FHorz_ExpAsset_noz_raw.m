function [V,Policy]=ValueFnIter_FHorz_ExpAsset_noz_raw(n_d1, n_d2,n_a1,n_a2,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1 = prod(n_d1);
N_d2 = prod(n_d2);
has_d1 = (N_d1 > 0); Nd1_eff = max(N_d1, 1);
has_d2 = (N_d2 > 0); Nd2_eff = max(N_d2, 1);

N_a1 = prod(n_a1);
has_a1 = (N_a1 > 0); Na1_eff = max(N_a1, 1);
N_a2 = prod(n_a2);
N_a = Na1_eff * N_a2;

if ~has_d1
    d_gridvals = d2_gridvals;
    n_d1 = 0; % ensures CreateReturnFnMatrix handles it correctly as having no d1
end

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
% n_a1prime=n_a1;
% a1prime_gridvals=a1_gridvals;
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % l_a2==1: a2primeIndex/a2primeProbs are [Nd2_eff,N_a2] (legacy lower-corner + prob)
    % l_a2>1 : a2primeIndex/a2primeProbs are [l_a2,Nd2_eff,N_a2] per-dim factored (lower index and prob within each a2 dim; NOT a Kron fold)

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:Na1_eff)',Nd2_eff,N_a2)+Na1_eff*repmat((a2primeIndex-1),Na1_eff,1); % [Nd2_eff*Na1_eff,N_a2]
        aprimeplus1Index=repelem((1:1:Na1_eff)',Nd2_eff,N_a2)+Na1_eff*repmat(a2primeIndex,Na1_eff,1); % [Nd2_eff*Na1_eff,N_a2]
        aprimeProbs=repmat(a2primeProbs,Na1_eff,1);  % [Nd2_eff*Na1_eff,N_a2]

        Vlower=reshape(EVpre(aprimeIndex(:)),[Nd2_eff*Na1_eff,N_a2]);
        Vupper=reshape(EVpre(aprimeplus1Index(:)),[Nd2_eff*Na1_eff,N_a2]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0; % effectively skips interpolation
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2)
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % a2primeIndex/a2primeProbs shape [l_a2=2, Nd2_eff, N_a2] per-dim. Nested 2-corner with skipinterp.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[Nd2_eff,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[Nd2_eff,N_a2]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:),[Nd2_eff,N_a2]),Na1_eff,1);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:),[Nd2_eff,N_a2]),Na1_eff,1);

        a1prime_offsets=repelem((1:1:Na1_eff)',Nd2_eff,N_a2);
        aprime_ll=a1prime_offsets+Na1_eff*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,Na1_eff,1);
        aprime_hl=a1prime_offsets+Na1_eff*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,Na1_eff,1);
        aprime_lh=a1prime_offsets+Na1_eff*repmat(loIdx_1+n_a2_1*loIdx_2-1,Na1_eff,1);
        aprime_hh=a1prime_offsets+Na1_eff*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,Na1_eff,1);
        V_ll=reshape(EVpre(aprime_ll(:)),[Nd2_eff*Na1_eff,N_a2]);
        V_hl=reshape(EVpre(aprime_hl(:)),[Nd2_eff*Na1_eff,N_a2]);
        V_lh=reshape(EVpre(aprime_lh(:)),[Nd2_eff*Na1_eff,N_a2]);
        V_hh=reshape(EVpre(aprime_hh(:)),[Nd2_eff*Na1_eff,N_a2]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    EV(isnan(EV))=0;

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,Nd1_eff,Na1_eff);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,N_j)=shiftdim(Vtemp,1);
    Policy(:,N_j)=shiftdim(maxindex,1);

end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % l_a2==1: [Nd2_eff,N_a2] legacy; l_a2>1: [l_a2,Nd2_eff,N_a2] per-dim factored (NOT a Kron fold)

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:Na1_eff)',Nd2_eff,N_a2)+Na1_eff*repmat((a2primeIndex-1),Na1_eff,1);
        aprimeplus1Index=repelem((1:1:Na1_eff)',Nd2_eff,N_a2)+Na1_eff*repmat(a2primeIndex,Na1_eff,1);
        aprimeProbs=repmat(a2primeProbs,Na1_eff,1);

        Vlower=reshape(V(aprimeIndex(:),jj+1),[Nd2_eff*Na1_eff,N_a2]);
        Vupper=reshape(V(aprimeplus1Index(:),jj+1),[Nd2_eff*Na1_eff,N_a2]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % a2primeIndex/a2primeProbs shape [l_a2=2, Nd2_eff, N_a2] per-dim. Nested 2-corner with skipinterp.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[Nd2_eff,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[Nd2_eff,N_a2]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:),[Nd2_eff,N_a2]),Na1_eff,1);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:),[Nd2_eff,N_a2]),Na1_eff,1);

        a1prime_offsets=repelem((1:1:Na1_eff)',Nd2_eff,N_a2);
        aprime_ll=a1prime_offsets+Na1_eff*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,Na1_eff,1);
        aprime_hl=a1prime_offsets+Na1_eff*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,Na1_eff,1);
        aprime_lh=a1prime_offsets+Na1_eff*repmat(loIdx_1+n_a2_1*loIdx_2-1,Na1_eff,1);
        aprime_hh=a1prime_offsets+Na1_eff*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,Na1_eff,1);
        V_ll=reshape(V(aprime_ll(:),jj+1),[Nd2_eff*Na1_eff,N_a2]);
        V_hl=reshape(V(aprime_hl(:),jj+1),[Nd2_eff*Na1_eff,N_a2]);
        V_lh=reshape(V(aprime_lh(:),jj+1),[Nd2_eff*Na1_eff,N_a2]);
        V_hh=reshape(V(aprime_hh(:),jj+1),[Nd2_eff*Na1_eff,N_a2]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    EV(isnan(EV))=0;

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,Nd1_eff,Na1_eff);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,jj)=shiftdim(Vtemp,1);
    Policy(:,jj)=shiftdim(maxindex,1);

end

%%
Policy=shiftdim(Policy,-1);



end
