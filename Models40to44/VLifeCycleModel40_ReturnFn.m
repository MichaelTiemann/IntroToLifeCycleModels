function F = VLifeCycleModel40_ReturnFn(aprime, hprime, a, h, z, w, r, p, sigma, theta, upsilon, gamma, phi, delta_o, agej, Jr, pension, kappa_j, wg1, wg2, wg3, beta, sj)

% 1. Income (working age vs retired - agej is a scalar so 'if' is safe here)
if agej < Jr
    income = w * kappa_j .* z;
else
    income = pension;
end

% 2. Housing transactions cost
tau_hhprime = phi .* h .* (hprime ~= h);

% 3. Resources available before purchasing new house and choosing aprime
resources = income + (1+r).*a + (1-delta_o).*h - tau_hhprime;

% ==========================================
% RENTER LOGIC (hprime == 0)
% ==========================================
cspend = resources - aprime;

if upsilon == 0 % Cobb-Douglas limit
    c_rent = theta .* cspend;
    d_rent = (1-theta) .* cspend ./ p;
else
    c_rent = cspend ./ (1 + (p^(upsilon/(upsilon-1))) * ((theta/(1-theta))^(1/(upsilon-1))));
    d_rent = (cspend - c_rent) ./ p;
end

% Shield against complex numbers from negative fractional powers
c_rent_safe = max(c_rent, 1e-10);
d_rent_safe = max(d_rent, 1e-10);

if upsilon == 0
    uinner_rent = (c_rent_safe.^theta) .* (d_rent_safe.^(1-theta));
else
    uinner_rent = (theta .* (c_rent_safe.^upsilon) + (1-theta) .* (d_rent_safe.^upsilon)).^(1/upsilon);
end

F_rent = (uinner_rent.^(1-sigma)) ./ (1-sigma);

% Because cspend is fully 6D, invalid_rent is safely 6D
invalid_rent = (cspend <= 0) | (c_rent <= 0) | (d_rent <= 0) | (aprime < 0);
F_rent(invalid_rent) = -Inf;

% ==========================================
% OWNER LOGIC (hprime > 0)
% ==========================================
c_own = resources - aprime - hprime;

c_own_safe = max(c_own, 1e-10);
hprime_safe = max(hprime, 1e-10);

if upsilon == 0
    uinner_own = (c_own_safe.^theta) .* (hprime_safe.^(1-theta));
else
    uinner_own = (theta .* (c_own_safe.^upsilon) + (1-theta) .* (hprime_safe.^upsilon)).^(1/upsilon);
end

F_own = (uinner_own.^(1-sigma)) ./ (1-sigma);

invalid_own = (c_own <= 0) | (aprime < -(1-gamma).*hprime);
F_own(invalid_own) = -Inf;

% ==========================================
% COMBINE RENTERS AND OWNERS
% ==========================================
F = F_own;

% Force the 2D renter mask to explicitly expand to the massive 6D size
is_renter_full = (hprime == 0) | false(size(F_own));
F(is_renter_full) = F_rent(is_renter_full);

% ==========================================
% WARM GLOW OF BEQUESTS
% ==========================================
if agej >= Jr + 10
    bequest = aprime + (1-delta_o).*hprime;

    % Shield fractional power from negative arguments
    beq_safe = max(bequest, -wg2 + 1e-10);

    warmglow = wg1 .* ((1 + beq_safe./wg2).^(1-wg3)) ./ (1-wg3);
    warmglow = beta * (1-sj) .* warmglow;

    % Mathematical masking!
    % If F is -Inf, adding a finite number leaves it -Inf.
    % We bypass the CUDA size crash entirely.
    F = F + warmglow .* (bequest > -wg2);
end


end