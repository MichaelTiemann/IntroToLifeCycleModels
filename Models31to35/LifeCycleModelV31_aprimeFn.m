function aprime = LifeCycleModelV31_aprimeFn(riskyshare, savings, u, r)

% Natively vectorizes across [N_d2, 1, 1], [1, N_d3, 1], and [1, 1, N_u] tensors
% using MATLAB implicit expansion.
aprime = (1 + r) .* (1 - riskyshare) .* savings + (1 + r + u) .* riskyshare .* savings;


end