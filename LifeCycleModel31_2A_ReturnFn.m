function F = LifeCycleModel31_2A_ReturnFn(d_m, aprime_safe, aprime_future, a_safe, a_future, m, z, w, r_safe, r_future, sigma, agej, Jr, pension, kappa_j)

% 3D Asset Return Function: Safe, Future, and Risky
% aprime_safe   : DC+GI continuous choice (Safe Savings)
% aprime_future : Discrete folded choice (Future Fund)
% d_m           : Discrete choice (Absolute Risky Investment)

% 1. Calculate the available cash from the Future Fund
% If they withdraw from the future fund before retirement, hit them with a 10% penalty on the withdrawn amount.
future_cash = (1 + r_future) * a_future;
penalty = 0;
if agej < Jr
    withdrawal = max(0, future_cash - aprime_future);
    penalty = 0.10 * withdrawal;
end

% 2. Calculate Consumption
if agej < Jr
    % Working age
    cons = w * kappa_j * z + (1 + r_safe) * a_safe + future_cash + m - aprime_safe - aprime_future - d_m - penalty;
else
    % Retirement
    cons = pension + (1 + r_safe) * a_safe + future_cash + m - aprime_safe - aprime_future - d_m;
end

% 3. Evaluate Utility
if cons > 0
    F = (cons^(1 - sigma) - 1) / (1 - sigma);
else
    F = -Inf;
end


end
