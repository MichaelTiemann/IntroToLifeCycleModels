function Fmatrix = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, n_bothz, n_e, d_gridvals, a1prime_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec, step)
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

% Extract raw sizes, but explicitly zero them out if the config declares them inactive
l_d = size(d_gridvals, 2);      if prod(n_d) == 0; l_d = 0; end
l_a1 = size(a1_gridvals, 2);    if prod(n_a1) == 0; l_a1 = 0; end
l_a2 = size(a2_gridvals, 2);    if prod(n_a2) == 0; l_a2 = 0; end
l_bz = size(bothz_gridvals, 2); if prod(n_bothz) == 0; l_bz = 0; end
l_e = size(e_gridvals, 2);      if prod(n_e) == 0; l_e = 0; end

d_vals = cell(1, l_d);
for i = 1:l_d; d_vals{i} = d_gridvals(:, i); end % Dim 1

a1prime_vals = cell(1, l_a1);
if step == 1
    for i = 1:l_a1; a1prime_vals{i} = shiftdim(a1prime_gridvals(:, i), -1); end % Dim 2
else
    for i = 1:l_a1; a1prime_vals{i} = a1prime_gridvals; end % N-D array exactly as passed
end

a1_vals = cell(1, l_a1); for i = 1:l_a1; a1_vals{i} = shiftdim(a1_gridvals(:, i), -2); end % Dim 3
a2_vals = cell(1, l_a2); for i = 1:l_a2; a2_vals{i} = shiftdim(a2_gridvals(:, i), -3); end % Dim 4
bz_vals = cell(1, l_bz); for i = 1:l_bz; bz_vals{i} = shiftdim(bothz_gridvals(:, i), -4); end % Dim 5
e_vals = cell(1, l_e); for i = 1:l_e; e_vals{i} = shiftdim(e_gridvals(:, i), -5); end % Dim 6

GridParamsCell = [d_vals, a1prime_vals, a1_vals, a2_vals, bz_vals, e_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});


end
