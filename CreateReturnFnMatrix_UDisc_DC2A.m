function Fmatrix = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, n_bothz, n_e, d_gridvals, a1prime_gridvals, a2prime_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec, step)
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

% Robust dimension lengths
l_d = length(n_d);      if prod(n_d) == 0; l_d = 0; end
l_a1 = length(a1_gridvals);
l_a2 = length(a2_gridvals);
l_bz = length(n_bothz); if prod(n_bothz) == 0; l_bz = 0; end
l_e = length(n_e);      if prod(n_e) == 0; l_e = 0; end

% Shift dimensions for implicit expansion
d_vals = cell(1, l_d);
for i = 1:l_d; d_vals{i} = d_gridvals(:, i); end % Dim 1

a1prime_vals = cell(1, 1);
if step == 1
    a1prime_vals{1} = shiftdim(a1prime_gridvals, -1); % Dim 2
else
    a1prime_vals{1} = a1prime_gridvals; % Passed exactly as shaped (N-D array)
end

a2prime_vals = {shiftdim(a2prime_gridvals, -2)}; % Dim 3
a1_vals = {shiftdim(a1_gridvals, -3)};           % Dim 4
a2_vals = {shiftdim(a2_gridvals, -4)};           % Dim 5

bz_vals = cell(1, l_bz); for i = 1:l_bz; bz_vals{i} = shiftdim(bothz_gridvals(:, i), -5); end % Dim 6
e_vals = cell(1, l_e); for i = 1:l_e; e_vals{i} = shiftdim(e_gridvals(:, i), -6); end         % Dim 7

GridParamsCell = [d_vals, a1prime_vals, a2prime_vals, a1_vals, a2_vals, bz_vals, e_vals];

% Evaluate function using arrayfun (No trailing reshapes required!)
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});


end
