function Fmatrix = CreateReturnFnMatrix_UDisc_DC1(ReturnFn, n_d, n_a, n_bothz, n_e, d_gridvals, a1p_grid, a1_grid, bothz_grid, e_grid, ReturnFnParamsVec, step)

% Pack parameters into cells so arrayfun treats them as scalars
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

% Conditionally append grid states ONLY if the model actually defines them.
GridParamsCell = {};

if sum(n_d) > 0
    for i = 1:size(d_gridvals, 2); GridParamsCell{end+1} = d_gridvals(:, i); end % Dim 1
end

if step == 1
    GridParamsCell{end+1} = shiftdim(a1p_grid, -1); % Dim 2
else
    GridParamsCell{end+1} = a1p_grid; % Step 2 N-D tensor
end

GridParamsCell{end+1} = shiftdim(a1_grid, -2); % Dim 3

if sum(n_bothz) > 0
    for i = 1:size(bothz_grid, 2); GridParamsCell{end+1} = shiftdim(bothz_grid(:, i), -3); end % Dim 4
end

if sum(n_e) > 0
    for i = 1:size(e_grid, 2); GridParamsCell{end+1} = shiftdim(e_grid(:, i), -4); end % Dim 5
end

Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});


end
