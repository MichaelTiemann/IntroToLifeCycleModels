function F = LifeCycleModelV8_ReturnFn(h, aprime, a, z, w, sigma, psi, eta, agej, Jr, pension, r, kappa_j, wg1, wg2, wg3, beta, sj)

% In this model, z is the fourth input: (h, aprime, a, z, ...)
% z represents the employment/unemployment shock [1; 0].

F = -Inf(size(h), 'like', h);

if agej < Jr 
    % Vectorized budget constraint for working age: wage * productivity * shock * hours + assets - next_period_assets
    c = w .* kappa_j .* z .* h + (1 + r) .* a - aprime; 
else 
    % Retirement budget constraint (unaffected by z)
    c = pension + (1 + r) .* a - aprime;
end

% Logical mask for valid consumption
valid_c = c > 0;

% Safe c to prevent complex numbers
c_safe = c;
c_safe(~valid_c) = 1;

% Element-wise utility calculation
utility = (c_safe.^(1 - sigma)) ./ (1 - sigma) - psi .* (h.^(1 + eta)) ./ (1 + eta);

% Map valid utilities
F(valid_c) = utility(valid_c);

% Add warm glow of bequests near end of life
if agej >= Jr + 10
    warmglow = wg1 .* ((1 + aprime ./ wg2).^(1 - wg3)) ./ (1 - wg3);
    warmglow = beta .* (1 - sj) .* warmglow;
    F = F + warmglow;
end

end
