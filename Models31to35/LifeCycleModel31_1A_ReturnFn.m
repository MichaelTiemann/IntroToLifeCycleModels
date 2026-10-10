function F = LifeCycleModel31_1A_ReturnFn(d_m, aprime, a, m, z, w, r, sigma, agej, Jr, pension, kappa_j)

% Inputs are strictly SCALAR for arrayfun compatibility.
% aprime is the DC+GI continuous choice variable (Safe Savings).
% d_m is the discrete choice variable (Absolute Risky Investment).
% a is the current safe savings.
% m is the realized risky asset value today (after u-shock).

% 1. Calculate Consumption
if agej < Jr % Working age
    cons = w * kappa_j * z + (1 + r) * a + m - aprime - d_m;
else % Retirement
    cons = pension + (1 + r) * a + m - aprime - d_m;
end

% 2. Evaluate Utility
if cons > 0
    F = (cons^(1 - sigma) - 1) / (1 - sigma);
else
    F = -Inf;
end


end
