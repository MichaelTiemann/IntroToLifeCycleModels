function F = LifeCycleModelV34_ReturnFn(h, savings, a, z, w, sigma, agej, Jr, pension, kappa_j, eta, psi)

% h is d1, savings is d3

% 1. Calculate Consumption
if agej < Jr % Working age
    c = w .* kappa_j .* z .* h + a - savings;
else % Retirement
    % h is still evaluated, but yields no wage. 
    % The maximizer will naturally push h=0 to avoid disutility.
    c = pension + a - savings; 
end

% 2. NaN Shield for Consumption (c > 0)
valid_c = (c > 0);
c_safe = c;
c_safe(~valid_c) = 1; % Prevent complex numbers or NaNs

% 3. NaN Shield for Labor (0 <= h < 1)
% Prevents (1-h) from becoming <= 0 and blowing up the exponent
valid_h = (h >= 0) & (h < 1);
h_safe = h;
h_safe(~valid_h) = 0; % Prevent NaNs

% 4. Evaluate Utility
U_c = (c_safe.^(1 - sigma)) ./ (1 - sigma);
U_h = psi .* ((1 - h_safe).^(1 - eta)) ./ (1 - eta);

F = U_c + U_h;

% 5. Apply infinite penalty where constraints are violated
F(~valid_c | ~valid_h) = -Inf;


end