function F = LifeCycleModelV28_ReturnFn(h, f, aprime, a, n1, n2, z, ...
    w, sigma, psi, eta, agej, eta1, eta2, eta3, nbar, hbar, h_c, ...
    childcarec, Jr, pension, r, kappa_j, wg1, wg2, wg3, beta, sj)

% 1. Time and financial child costs
infanttime     = h_c .* n1;
childcarecosts = childcarec .* n1 .* (h > 0);

% 2. Consumption tensor across broadcasted dimensions
%    Preserve Dim 4 (choices h, f) and Dim 2 (semi-exogenous / exogenous states) during retirement
if agej < Jr
    c = w .* kappa_j .* z .* h + (1 + r) .* a - childcarecosts - aprime;
else
    c = pension + (1 + r) .* a - aprime + 0 .* (h + f + n1 + n2 + z);
end

leisure = hbar - h - infanttime;
consumption_equiv_units = 1 + 0.3 .* n1 + 0.5 .* n2;

% Age-specific utility value of children
utility_of_children = (eta1 .* exp(agej - eta3) ./ (1 + exp(agej - eta3))) .* ...
                      ((nbar + n1 + n2) .^ eta2);

% 3. CRRA Utility over equivalence-scaled consumption and labor disutility
c_scaled = c ./ consumption_equiv_units;
c_safe   = max(c_scaled, realmin('like', c));
F = (c_safe .^ (1 - sigma)) ./ (1 - sigma);

labor_used = max(hbar - leisure, realmin('like', leisure));
F = F - psi .* (labor_used .^ (1 + eta2)) ./ (1 + eta) + utility_of_children;

% 4. Warm glow of bequests near end of life
if agej >= Jr + 10
    wg_base  = max(1 + aprime ./ wg2, realmin('like', aprime));
    warmglow = wg1 .* (wg_base .^ (1 - wg3)) ./ (1 - wg3);
    warmglow = beta .* (1 - sj) .* warmglow;
    F = F + warmglow;
end

% 5. Mask infeasible choices (non-positive consumption, leisure constraint, or non-finite)
invalid = ~isfinite(c) | (c <= 0) | (leisure >= 1);
F(invalid) = -Inf;

end