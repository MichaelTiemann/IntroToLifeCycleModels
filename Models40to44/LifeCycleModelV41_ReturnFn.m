function F = LifeCycleModelV41_ReturnFn(p, aprime, a, h, z, w, sigma, psi, y_m, childcarecosts, agej, Jr, pension, r, wg1, wg2, wg3, beta, sj)

% 1. Create a dummy tensor to force perfect implicit expansion across all states
dummy = aprime .* a .* h .* z .* 0;

% 2. Initialize F to the exact size of the fully expanded space
F = -inf(size(dummy), 'like', a);

% 3. Calculate consumption, padded with the dummy so it spans all dimensions (even in retirement)
if agej < Jr % Working age
    c = w .* h .* z .* p - childcarecosts .* p + y_m + (1 + r) .* a - aprime + dummy;
else % Retirement
    c = pension + (1 + r) .* a - aprime + dummy;
end

% 4. Create a logical mask for valid consumption
valid_c = c > 0;

% 5. Calculate utility only where consumption is strictly positive
if any(valid_c, 'all')
    % p is a scalar from the Map-Reduce loop. We must expand it before logical indexing!
    p_expanded = p + dummy; 
    F(valid_c) = (c(valid_c).^(1 - sigma)) ./ (1 - sigma) - psi .* p_expanded(valid_c);
end

% 6. Add warm glow of bequests (only near end of life)
if agej >= Jr + 10
    warmglow = wg1 .* ((1 + aprime ./ wg2).^(1 - wg3)) ./ (1 - wg3);
    
    % Pad warmglow with the dummy so its dimensions match valid_c exactly
    warmglow = beta .* (1 - sj) .* warmglow + dummy;
    
    % Add warm glow only to valid states
    F(valid_c) = F(valid_c) + warmglow(valid_c);
end


end
