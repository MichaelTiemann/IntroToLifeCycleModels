function aprime = LifeCycleModelV34_aprimeFn(riskyshare, savings, u, r)

% Note: riskyshare is d2, savings is d3, u is the shock

% Create logical masks for branching
pos_mask = (savings > 0);
neg_mask = (savings <= 0);

% Vectorized evaluation
% Positive savings get split between safe and risky returns. 
% Negative savings (borrowing) strictly get the safe rate.
aprime = pos_mask .* ((1 + r) * (1 - riskyshare) .* savings + (1 + r + u) .* riskyshare .* savings) + ...
    neg_mask .* ((1 + r) .* savings);


end