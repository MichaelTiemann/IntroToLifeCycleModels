function F = LifeCycleModelV27_ReturnFn(h1, h2, aprime, a, z1, z2, e1, e2, ...
    w, sigma, psi, eta, agej, Jr, pension, r, kappa_j_1, kappa_j_2, wg1, wg2, wg3, beta, sj)

% 1. Compute consumption tensor across broadcasted state and choice dimensions
%    Preserve Dim 4 (choices h1, h2) and Dim 2/3 (z, e) during retirement
if agej < Jr
    c = w .* kappa_j_1 .* z1 .* e1 .* h1 + ...
        w .* kappa_j_2 .* z2 .* e2 .* h2 + ...
        (1 + r) .* a - aprime;
else
    c = pension + (1 + r) .* a - aprime + 0 .* (h1 + h2 + z1 + z2 + e1 + e2);
end

% 2. CRRA utility over consumption (clamped for numeric stability before masking)
c_safe = max(c, realmin('like', c));
F = (c_safe .^ (1 - sigma)) ./ (1 - sigma);

% 3. Disutility of labor for each earner (active during working age)
if agej < Jr
    h1_safe = max(h1, realmin('like', h1));
    h2_safe = max(h2, realmin('like', h2));
    F = F - psi .* (h1_safe .^ (1 + eta)) ./ (1 + eta) ...
        - psi .* (h2_safe .^ (1 + eta)) ./ (1 + eta);
end

% 4. Warm glow of bequests near end of life
if agej >= Jr + 10
    wg_base = max(1 + aprime ./ wg2, realmin('like', aprime));
    warmglow = wg1 .* (wg_base .^ (1 - wg3)) ./ (1 - wg3);
    warmglow = beta .* (1 - sj) .* warmglow;
    F = F + warmglow;
end

% 5. Mask infeasible choices (non-positive consumption or NaN/Inf)
invalid = ~isfinite(c) | (c <= 0);
F(invalid) = -Inf;

end
