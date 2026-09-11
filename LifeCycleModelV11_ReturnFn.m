function F = LifeCycleModelV11_ReturnFn(h, aprime, a, z, e, w, sigma, psi, eta, agej, Jr, pension, r, kappa_j, wg1, wg2, wg3, beta, sj)

% Evaluate consumption across broadcast dimensions
if agej < Jr
    c = w .* kappa_j .* z .* e .* h + (1 + r) .* a - aprime;
else
    c = pension + (1 + r) .* a - aprime + 0.* (h + z + e);
end

% Base return: -Inf where budget constraint is violated (c <= 0)
F = -Inf(size(c), 'like', c);

valid_c = (c > 0);
if any(valid_c(:))
    % Labor disutility term
    disutil_h = psi .* (h.^(1 + eta)) ./ (1 + eta);
    
    if sigma == 1
        F(valid_c) = log(c(valid_c)) - disutil_h(valid_c);
    else
        F(valid_c) = (c(valid_c).^(1 - sigma)) ./ (1 - sigma) - disutil_h(valid_c);
    end
end

% Warm glow bequest near end of life
if agej >= Jr + 10
    warmglow = wg1 .* ((1 + aprime ./ wg2).^(1 - wg3)) ./ (1 - wg3);
    warmglow = (beta * (1 - sj)) .* warmglow;
    F = F + warmglow;
end

end