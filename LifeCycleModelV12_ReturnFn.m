function F = LifeCycleModelV12_ReturnFn(h, aprime, a, z, w, sigma, psi, eta, agej, Jr, pension, r, kappa_j)

if agej < Jr
    c = w .* kappa_j .* z .* h + (1 + r) .* a - aprime;
    
    % 1. Create strictly safe bases for the entire tensor
    c_safe = max(c, realmin('like', c));
    h_safe = max(h, 0); 
    
    % 2. Evaluate utility everywhere (no size changes, no syncs)
    u_c = (c_safe .^ (1 - sigma)) ./ (1 - sigma);
    u_h = psi .* (h_safe .^ (1 + eta)) ./ (1 + eta);
    F = u_c - u_h;
    
    % 3. Mask out invalid states purely via logical assignment
    invalid = ~isfinite(c) | (c <= 0) | ~isfinite(h) | (h < 0);
    F(invalid) = -Inf;
    
else
    c = pension + (1 + r) .* a - aprime;
    
    c_safe = max(c, realmin('like', c));
    F = (c_safe .^ (1 - sigma)) ./ (1 - sigma);
    
    invalid = ~isfinite(c) | (c <= 0);
    F(invalid) = -Inf;
end

end