function F=LifeCycleModel21_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,wg1,wg2,wg3,beta,sj)
% Is LifeCycleModel8_ReturnFn, but modified to include medical expense
% shocks when retired.

% F=-Inf;
if agej<Jr % If working age
    c=w.*kappa_j.*z.*h+(1+r).*a-aprime;
else % Retirement
    c=pension+(1+r).*a-z-aprime+0.*h; % Subtract z here
end

% Avoid evaluating a fractional power at non-positive consumption, then
% restore the original infeasibility penalty on exactly those elements.
feasible=(c>0);
c_for_utility=c;
c_for_utility(!feasible)=1;
F=(c_for_utility.^(1-sigma))./(1-sigma) ...
    -psi.*(h.^(1+eta))./(1+eta); % The utility function
F(!feasible)=-Inf;

% add the warm glow to the return, but only near end of life
if agej>=Jr+10
    % Warm glow of bequests: bequest are a luxury good
    warmglow=wg1.*((1+aprime./wg2).^(1-wg3))./(1-wg3);
    % Modify for beta and sj (get the warm glow next period if die)
    warmglow=beta.*(1-sj).*warmglow;
    % add the warm glow to the return
    F=F+warmglow;
end

end
