function F = VLifeCycleModel40_ReturnFn(aprime, hprime, a, h, z, w, r, p, sigma, theta, upsilon, gamma, phi, delta_o, agej, Jr, pension, kappa_j, wg1, wg2, wg3, beta, sj)

% 1. Income (working age vs retired) -> TINY ARRAY
if agej < Jr
    income = w * kappa_j .* z;
else
    income = pension;
end

% 2. Housing transactions cost -> TINY ARRAY
tau_hhprime = phi .* h .* (hprime ~= h);

% 3. Resources -> MEDIUM ARRAY (Only 4 active dims, no aprime yet)
resources = income + (1+r).*a + (1-delta_o).*h - tau_hhprime;

% ==========================================
% ALGEBRAIC PRE-COLLAPSING: RENTER SCALARS
% ==========================================
% c_rent and d_rent are strictly proportional to cspend.
% We collapse the entire CES utility into a single scalar multiplier!
if upsilon == 0
    K_rent = (theta^theta) * (((1-theta)/p)^(1-theta));
else
    K_c = 1 / (1 + (p^(upsilon/(upsilon-1))) * ((theta/(1-theta))^(1/(upsilon-1))));
    K_d = (1 - K_c) / p;
    K_rent = (theta * K_c^upsilon + (1-theta) * K_d^upsilon)^(1/upsilon);
end
K_F_rent = (K_rent^(1-sigma)) / (1-sigma);

% ==========================================
% OWNER LOGIC (Evaluated Globally for Speed)
% ==========================================
% This is the FIRST massive array allocation
c_own = resources - aprime - hprime;

hprime_safe = max(hprime, 1e-10); % TINY ARRAY

% Collapsed power operations to minimize VRAM bandwidth
if upsilon == 0
    h_pow = hprime_safe.^((1-theta)*(1-sigma)) ./ (1-sigma);
    F = max(c_own, 1e-10).^(theta*(1-sigma)) .* h_pow;
else
    h_pow = (1-theta) .* hprime_safe.^upsilon;
    uinner_pow = theta .* max(c_own, 1e-10).^upsilon + h_pow;
    F = uinner_pow.^((1-sigma)/upsilon) ./ (1-sigma);
end

% Owner Constraints (Math-masking bypasses logical find())
invalid_borrow = aprime < -(1-gamma).*hprime; % TINY ARRAY
F(c_own <= 0 | invalid_borrow) = -Inf; % SECOND massive allocation

% ==========================================
% RENTER LOGIC (Subscript Overwrite)
% ==========================================
hprime_1d = squeeze(hprime);
renter_idx = find(hprime_1d == 0);

if ~isempty(renter_idx)
    % Extract ONLY the renter slice from resources to save memory
    res_renter = resources(:, :, renter_idx, :, :, :);

    % Shield against differing aprime geometries (Branch 1 vs Slicer)
    if size(aprime, 3) > 1
        aprime_renter = aprime(:, :, renter_idx, :, :, :);
    else
        aprime_renter = aprime;
    end

    cspend = res_renter - aprime_renter; % Strictly collapsed array

    % The entire renter utility is now executed in ONE operation
    F_rent_slice = K_F_rent .* max(cspend, 1e-10).^(1-sigma);
    F_rent_slice(cspend <= 0 | aprime_renter < 0) = -Inf;

    % Direct subscript overwrite (zero logical masking overhead)
    F(:, :, renter_idx, :, :, :) = F_rent_slice;
end

% ==========================================
% WARM GLOW OF BEQUESTS
% ==========================================
if agej >= Jr + 10
    bequest = aprime + (1-delta_o).*hprime; % TINY ARRAY
    beq_safe = max(bequest, -wg2 + 1e-10);
    warmglow = wg1 .* ((1 + beq_safe./wg2).^(1-wg3)) ./ (1-wg3);
    warmglow = beta * (1-sj) .* warmglow;

    % Mathematical masking! If F is -Inf, adding a finite number leaves it -Inf.
    % This completely bypasses the CUDA 6D expansion crash.
    F = F + warmglow .* (bequest > -wg2);
end


end