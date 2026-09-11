function F = LifeCycleModelV12_ReturnFn(h, aprime, a, z, w, sigma, psi, eta, agej, Jr, pension, r, kappa_j)

% 1. Determine consumption across all broadcasted tensor dimensions
if agej < Jr
    % Working age: labor income + financial wealth - asset choice
    c = w .* kappa_j .* z .* h + (1 + r) .* a - aprime;
else
    % Retirement: pension + financial wealth - asset choice
    c = pension + (1 + r) .* a - aprime;
end

% 2. Initialize utility tensor matching the broadcasted shape
F = -Inf(size(c), 'like', c);

% 3. Evaluate utility on strictly positive consumption
valid = (c > 0);

if agej < Jr
    % Utility from consumption and disutility from hours
    F(valid) = (c(valid) .^ (1 - sigma)) ./ (1 - sigma) - ...
        psi .* (h(valid) .^ (1 + eta)) ./ (1 + eta);
else
    % In retirement, h is typically 0 or inactive; disutility term drops
    F(valid) = (c(valid) .^ (1 - sigma)) ./ (1 - sigma);
end

end