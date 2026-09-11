function F = LifeCycleModelV15_ReturnFn(aprime, a, z, w, sigma, agej, Jr, pension, r, kappa_j, borrowingconstraint)

% 1. Determine consumption across all broadcasted tensor dimensions
if agej < Jr
    c = w .* kappa_j + (1 + r) .* a - aprime;
else
    c = pension + (1 + r) .* a - aprime;
end

% 2. Safe base evaluation for GPU execution (prevents 0^(negative) and complex numbers)
c_safe = max(c, realmin('like', c));
F = (c_safe .^ (1 - sigma)) ./ (1 - sigma);

% 3. Mask out invalid consumption and borrowing constraint violations
invalid = ~isfinite(c) | (c <= 0) | (aprime < borrowingconstraint);
F(invalid) = -Inf;

end