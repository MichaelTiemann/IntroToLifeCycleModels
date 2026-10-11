function Fmatrix = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, n_bothz, n_e, d_gridvals, a1prime_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec, step)

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

% Robust dimension lengths
l_d = length(n_d);      if prod(n_d) == 0; l_d = 0; end
l_a1 = length(n_a1);    if prod(n_a1) == 0; l_a1 = 0; end
l_a2 = length(n_a2);    if prod(n_a2) == 0; l_a2 = 0; end
l_bz = length(n_bothz); if prod(n_bothz) == 0; l_bz = 0; end
l_e = length(n_e);      if prod(n_e) == 0; l_e = 0; end

% Shift dimensions for implicit expansion
d_vals = cell(1, l_d);
if step == 1
    for i = 1:l_d; d_vals{i} = d_gridvals(:, i); end
else
    % Step 2: Dynamically replicate d to match the flattened a1prime array length
    if l_d > 0
        maxgap_plus_1 = size(a1prime_gridvals, 1) / size(d_gridvals, 1);
        for i = 1:l_d; d_vals{i} = repmat(d_gridvals(:, i), maxgap_plus_1, 1); end
    end
end

a1prime_vals = cell(1, l_a1);
if step == 1
    for i = 1:l_a1; a1prime_vals{i} = shiftdim(a1prime_gridvals(:, i), -1); end
else
    for i = 1:l_a1; a1prime_vals{i} = a1prime_gridvals; end
end

a1_vals = cell(1, l_a1); for i = 1:l_a1; a1_vals{i} = shiftdim(a1_gridvals(:, i), -2); end
a2_vals = cell(1, l_a2); for i = 1:l_a2; a2_vals{i} = shiftdim(a2_gridvals(:, i), -3); end
bz_vals = cell(1, l_bz); for i = 1:l_bz; bz_vals{i} = shiftdim(bothz_gridvals(:, i), -4); end
e_vals = cell(1, l_e); for i = 1:l_e; e_vals{i} = shiftdim(e_gridvals(:, i), -5); end

GridParamsCell = [d_vals, a1prime_vals, a1_vals, a2_vals, bz_vals, e_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});


end
