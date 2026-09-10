function F = LifeCycleModelV7_ReturnFn(h, aprime, a, w, sigma, psi, eta, agej, Jr, pension, r, kappa_j, wg1, wg2, wg3, beta, sj)

% agej and Jr are time-period scalars passed from the backward induction loop.
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

% Add the warm glow to the return, but only near end of life
if agej >= Jr + 10
    % Warm glow of bequests: bequest are a luxury good (vectorized over aprime)
    warmglow = wg1 .* ((1 + aprime ./ wg2).^(1 - wg3)) ./ (1 - wg3);

    % Modify for beta and sj (get the warm glow next period if die)
    % beta and sj arrive as scalars for the current age j
    warmglow = beta .* (1 - sj) .* warmglow;

    % Add the warm glow to the return
    % Note: In MATLAB, -Inf + finite_number = -Inf, so invalid choices remain invalid.
    F = F + warmglow;
end


end