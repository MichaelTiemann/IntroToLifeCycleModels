function F = LifeCycleModelV21_ReturnFn(h, aprime, a, z, w, sigma, psi, eta, agej, Jr, pension, r, kappa_j, wg1, wg2, wg3, beta, sj)

% 1. Determine consumption across broadcasted tensor dimensions
%    Dim 1: a, Dim 2: z, Dim 4: h, Dim 5: aprime
if agej < Jr
    c = w .* kappa_j .* z .* h + (1 + r) .* a - aprime;
else
    % In retirement: z acts as the out-of-pocket medical expense shock
    c = pension + (1 + r) .* a - z - aprime + 0 .* h;
end

% 2. Safe base evaluation for CRRA utility over consumption
c_safe = max(c, realmin('like', c));
F = (c_safe .^ (1 - sigma)) ./ (1 - sigma);

% 3. Labor disutility (active during working age)
if agej < Jr
    h_safe = max(h, realmin('like', h));
    F = F - psi .* (h_safe .^ (1 + eta)) ./ (1 + eta);
end

% 4. Add warm glow of bequests near end of life
if agej >= Jr + 10
    wg_base = max(1 + aprime ./ wg2, realmin('like', aprime));
    warmglow = wg1 .* (wg_base .^ (1 - wg3)) ./ (1 - wg3);
    warmglow = beta .* (1 - sj) .* warmglow;
    F = F + warmglow;
end

% 5. Mask invalid states (non-positive consumption or NaN/Inf)
invalid = ~isfinite(c) | (c <= 0);
F(invalid) = -Inf;

end