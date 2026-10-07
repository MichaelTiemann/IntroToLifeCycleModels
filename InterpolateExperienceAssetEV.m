function EV = InterpolateExperienceAssetEV(EVpre, n_a2, N_d2, N_a1, N_a2, N_z, a2primeIndex, a2primeProbs)
% Interpolates the expected value function for experience assets based on 
% optimal continuous policy choices. Handles 1 or 2 experience dimensions.

z_offset = N_a1*N_a2 * shiftdim(0:N_z-1, -1);

if length(n_a2)==1
    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1);

    aprimeProbs=repmat(a2primeProbs,N_a1,1);

    Vlower=reshape(EVpre(aprimeIndex+z_offset),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index+z_offset),[N_d2*N_a1,N_a2,N_z]);

    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV(aprimeProbs==0)=Vupper(aprimeProbs==0);
    EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
else
    n_a2_1=n_a2(1);

    loIdx_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_z]);
    loIdx_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_z]);

    prob_1_exp=repmat(reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_z]),N_a1,1);
    prob_2_exp=repmat(reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_z]),N_a1,1);

    a1prime_offsets=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2);
    aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1);
    aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1);
    aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1);
    aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1);

    V_ll=reshape(EVpre(aprime_ll+z_offset),[N_d2*N_a1,N_a2,N_z]);
    V_hl=reshape(EVpre(aprime_hl+z_offset),[N_d2*N_a1,N_a2,N_z]);
    V_lh=reshape(EVpre(aprime_lh+z_offset),[N_d2*N_a1,N_a2,N_z]);
    V_hh=reshape(EVpre(aprime_hh+z_offset),[N_d2*N_a1,N_a2,N_z]);

    p1_loy=prob_1_exp;
    p1_loy(V_ll==V_hl)=0;
    c_ll=p1_loy.*V_ll;          c_ll(isnan(c_ll))=0;
    c_hl=(1-p1_loy).*V_hl;      c_hl(isnan(c_hl))=0;
    EV_loy=c_ll+c_hl;

    p1_hiy=prob_1_exp;
    p1_hiy(V_lh==V_hh)=0;
    c_lh=p1_hiy.*V_lh;          c_lh(isnan(c_lh))=0;
    c_hh=(1-p1_hiy).*V_hh;      c_hh(isnan(c_hh))=0;
    EV_hiy=c_lh+c_hh;

    p2=prob_2_exp;
    p2(EV_loy==EV_hiy)=0;
    c_loy=p2.*EV_loy;           c_loy(isnan(c_loy))=0;
    c_hiy=(1-p2).*EV_hiy;       c_hiy(isnan(c_hiy))=0;
    EV=c_loy+c_hiy;
end


end
