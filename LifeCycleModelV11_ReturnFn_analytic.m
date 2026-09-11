function [F, h_opt] = LifeCycleModelV11_ReturnFn_analytic(aprime, a, z, e, w, sigma, psi, eta, agej, Jr, pension, r, kappa_j, wg1, wg2, wg3, beta, sj, h_bounds)
% Analytically solves optimal labor h* given candidate (a, a', z, e)
% Eliminates the n_d discrete choice grid entirely.

if nargin < 19 || isempty(h_bounds)
    h_min = 0;
    h_max = 1;
else
    h_min = h_bounds(1);
    h_max = h_bounds(2);
end

x = (1 + r) .* a - aprime;

if agej < Jr
    w_eff = w .* kappa_j .* z .* e;

    % Analytical FOC for sigma = 1, eta = 1:
    % h^2 + (x/w_eff)*h - 1/psi = 0
    discriminant = x.^2 + (4 .* (w_eff.^2)) ./ psi;
    h_star = (-x + sqrt(discriminant)) ./ (2 .* w_eff);

    % Feasibility clamping
    h_opt = min(h_max, max(h_min, h_star));
    c = w_eff .* h_opt + x;
    disutil_h = psi .* (h_opt.^(1 + eta)) ./ (1 + eta);
else
    h_opt = zeros(size(x), 'like', a);
    c = pension + x;
    disutil_h = 0;
end

% Smooth element-wise utility evaluation
pos_c = max(c, realmin('like', c));

if sigma == 1
    u = log(pos_c);
else
    u = (pos_c.^(1 - sigma)) ./ (1 - sigma);
end

F = u - disutil_h;
F(c <= 0) = -Inf;

% Warm glow bequest near end of life
if agej >= Jr + 10
    warmglow = wg1 .* ((1 + aprime ./ wg2).^(1 - wg3)) ./ (1 - wg3);
    warmglow = (beta * (1 - sj)) .* warmglow;
    F = F + warmglow;
end

end
