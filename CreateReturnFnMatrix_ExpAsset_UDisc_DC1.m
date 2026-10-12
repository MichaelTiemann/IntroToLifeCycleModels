function Fmatrix = CreateReturnFnMatrix_ExpAsset_UDisc_DC1(ReturnFn, n_d, n_a1, n_a2, n_bothz, n_e, d_gridvals, a1prime_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec, step)

% Pack parameters into cells so arrayfun treats them as scalars
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

% Conditionally append grid states ONLY if the model actually defines them.
% This cleanly filters out dummy zero-matrices so arrayfun signature matches perfectly.

GridParamsCell = {};

if sum(n_d) > 0
    for i = 1:size(d_gridvals, 2); GridParamsCell{end+1} = d_gridvals(:, i); end % Dim 1
end

if sum(n_a1) > 0
    if step == 1
        for i = 1:size(a1prime_gridvals, 2)
            GridParamsCell{end+1} = shiftdim(a1prime_gridvals(:, i), -1); % Dim 2
        end
    else
        % Step 2 passes the pre-shaped N-D tensor.
        GridParamsCell{end+1} = a1prime_gridvals;
    end
    for i = 1:size(a1_gridvals, 2); GridParamsCell{end+1} = shiftdim(a1_gridvals(:, i), -2); end % Dim 3
end

if sum(n_a2) > 0
    for i = 1:size(a2_gridvals, 2); GridParamsCell{end+1} = shiftdim(a2_gridvals(:, i), -3); end % Dim 4
end

if sum(n_bothz) > 0
    for i = 1:size(bothz_gridvals, 2); GridParamsCell{end+1} = shiftdim(bothz_gridvals(:, i), -4); end % Dim 5
end

if sum(n_e) > 0
    for i = 1:size(e_gridvals, 2); GridParamsCell{end+1} = shiftdim(e_gridvals(:, i), -5); end % Dim 6
end

Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});


end
