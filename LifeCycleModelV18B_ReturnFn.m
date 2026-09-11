function F = LifeCycleModelV18B_ReturnFn(aprime, a, z, w, sigma, agej, Jr, pension, r, kappa_j, wg1, wg2, wg3, beta, sj, meanearningsratio)

% 1. Determine consumption across broadcasted tensor dimensions
if agej < Jr
    c = meanearningsratio .* w .* kappa_j .* z + (1 + r) .* a - aprime;
else
    c = pension + (1 + r) .* a - aprime;
end

% 2. Safe base evaluation for standard CRRA utility
c_safe = max(c, realmin('like', c));
F = (c_safe .^ (1 - sigma)) ./ (1 - sigma);

% 3. Add warm glow directly into the return surface near end-of-life
if agej >= Jr + 10
    % Guard base (1 + aprime/wg2) against negative/complex evaluation
    wg_base = max(1 + aprime ./ wg2, realmin('like', aprime));
    warmglow = wg1 .* (wg_base .^ (1 - wg3)) ./ (1 - wg3);
    
    % Discount and weight by mortality probability
    warmglow = beta .* (1 - sj) .* warmglow;
    F = F + warmglow;
end

% 4. Mask non-positive consumption with -Inf
invalid = ~isfinite(c) | (c <= 0);
F(invalid) = -Inf;

end
