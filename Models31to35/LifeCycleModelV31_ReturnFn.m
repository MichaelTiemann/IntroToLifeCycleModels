function F = LifeCycleModelV31_ReturnFn(savings, a, z, w, sigma, agej, Jr, pension, kappa_j)

% Natively vectorizes across [N_choice, N_block, N_z] tensors

if agej < Jr % If working age
    c = w .* kappa_j .* z + a - savings;
else % Retirement
    c = pension + a - savings;
end

% Preallocate the Return block with strict -Inf
F = -Inf(size(c), 'like', c);

% Find all valid (positive) consumption nodes
valid = (c > 0);

% Only calculate utility on the valid nodes to prevent complex numbers or NaNs
F(valid) = (c(valid).^(1 - sigma)) ./ (1 - sigma);


end