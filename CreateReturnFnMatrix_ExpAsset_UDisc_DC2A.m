function Fmatrix = CreateReturnFnMatrix_ExpAsset_UDisc_DC2A(ReturnFn, n_d, n_a1_c, n_a2_c, n_exp, n_bothz, n_e, d_gridvals, a1p_grid, a2p_grid, a1_grid, a2_grid, exp_grid, bothz_grid, e_grid, ReturnFnParamsVec, step)

% Pack parameters into cells so arrayfun treats them as scalars
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

% Conditionally append grid states ONLY if the model actually defines them.
GridParamsCell = {};

if sum(n_d) > 0
    for i = 1:size(d_gridvals, 2); GridParamsCell{end+1} = d_gridvals(:, i); end % Dim 1
end

if sum(n_a1_c) > 0
    if step == 1
        for i = 1:size(a1p_grid, 2); GridParamsCell{end+1} = shiftdim(a1p_grid(:, i), -1); end % Dim 2
    else
        GridParamsCell{end+1} = a1p_grid; % Step 2 tensor
    end
end

if sum(n_a2_c) > 0
    for i = 1:size(a2p_grid, 2); GridParamsCell{end+1} = shiftdim(a2p_grid(:, i), -2); end % Dim 3
end

if sum(n_a1_c) > 0
    for i = 1:size(a1_grid, 2); GridParamsCell{end+1} = shiftdim(a1_grid(:, i), -3); end % Dim 4
end

if sum(n_a2_c) > 0
    for i = 1:size(a2_grid, 2); GridParamsCell{end+1} = shiftdim(a2_grid(:, i), -4); end % Dim 5
end

if sum(n_exp) > 0
    for i = 1:size(exp_grid, 2); GridParamsCell{end+1} = shiftdim(exp_grid(:, i), -5); end % Dim 6
end

if sum(n_bothz) > 0
    if ~isempty(bothz_grid); bothz_grid = bothz_grid(:, n_bothz > 0); end
    for i = 1:size(bothz_grid, 2); GridParamsCell{end+1} = shiftdim(bothz_grid(:, i), -6); end % Dim 7
end

if sum(n_e) > 0
    for i = 1:size(e_grid, 2); GridParamsCell{end+1} = shiftdim(e_grid(:, i), -7); end % Dim 8
end

Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});


end
