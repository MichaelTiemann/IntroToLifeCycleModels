function F = LifeCycleModelV5_ReturnFn(h, aprime, a, w, sigma, psi, eta, agej, Jr, pension, r, kappa_j)

% agej and Jr are time-period scalars passed from the backward induction loop.
% kappa_j is also passed as a scalar for the specific age j.
if agej < Jr 
    % Vectorized budget constraint for working age with age-dependent productivity
    c = w .* kappa_j .* h + (1 + r) .* a - aprime; 
else 
    % Vectorized budget constraint for retirement
    c = pension + (1 + r) .* a - aprime;
end

% Logical mask for valid consumption
valid_c = c > 0;

% Safe c to prevent complex numbers
c_safe = c;
c_safe(~valid_c) = 1;

% Element-wise operators (.^, .*) across the entire multi-million point grid
utility = (c_safe.^(1 - sigma)) ./ (1 - sigma) - psi .* (h.^(1 + eta)) ./ (1 + eta);

% Preallocate F and map valid utilities
F = -Inf(size(h), 'like', h);
F(valid_c) = utility(valid_c);

end
