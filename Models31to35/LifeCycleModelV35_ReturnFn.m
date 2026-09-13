function F = LifeCycleModelV35_ReturnFn(savings, hprime, h, a, z, w, sigma, agej, Jr, pension, kappa_j, sigma_h, f_htc, minhouse, rentprice, f_coll, houseservices)

% 1. Housing Transaction Costs (htc)
% Boolean mask: 1 if moving/buying/selling, 0 if staying put
moved = (hprime ~= h);
htc = f_htc .* (h + hprime) .* moved;

% 2. Housing Services (s) and Rental Costs
% Boolean mask: 1 if renting, 0 if owning
is_renter = (h == 0);

% Allocate housing services: base services for owners, fallback minhouse for renters
s = (houseservices .* h .* ~is_renter) + (0.5 .* houseservices .* minhouse .* is_renter);
rentalcosts = rentprice .* is_renter;

% 3. Consumption (c)
if agej < Jr % Working age (Scalar evaluation, safe for 'if')
    c = w .* kappa_j .* z + a - savings + (h - hprime) - htc - rentalcosts;
else % Retirement
    c = pension + a - savings + (h - hprime) - htc - rentalcosts; 
end

% 4. NaN Shield for Consumption
valid_c = (c > 0);
c_safe = c;
c_safe(~valid_c) = 1; % Prevent complex numbers from negative fractional exponents

% 5. Utility Evaluation
% Cobb-Douglas aggregate of consumption and housing, nested inside CRRA
F = (((c_safe.^(1 - sigma_h)) .* (s.^sigma_h)).^(1 - sigma)) ./ (1 - sigma);

% 6. Apply Constraints (Set invalid states to -Inf)
% A. Negative Consumption
F(~valid_c) = -Inf;

% B. Collateral constraint on borrowing
collateral_violation = (savings < -f_coll .* hprime);
F(collateral_violation) = -Inf;

% C. Ban pensioners from negative assets
if agej >= Jr
    pensioner_borrowing = (savings < 0);
    F(pensioner_borrowing) = -Inf;
end
end