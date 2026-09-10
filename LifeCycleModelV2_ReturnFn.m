function F = LifeCycleModelV2_ReturnFn(h, aprime, a, w, sigma, psi, eta, agej, Jr, pension)

% agej and Jr are time-period scalars passed from the backward induction loop.
% We use standard if/else branching because this is evaluated once per age j.
if agej < Jr 
    c = w .* h; 
else 
    % Expand scalar pension to match the size of the vectorized h grid
    c = pension + zeros(size(h), 'like', h);
end

valid_c = c > 0;

c_safe = c;
c_safe(~valid_c) = 1;

% Element-wise operators (.^, .*) across the entire h grid
utility = (c_safe.^(1 - sigma)) ./ (1 - sigma) - psi .* (h.^(1 + eta)) ./ (1 + eta);

% Preallocate F and map valid utilities
F = -Inf(size(h), 'like', h);
F(valid_c) = utility(valid_c);


end
