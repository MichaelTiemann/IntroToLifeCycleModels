function Fmatrix = CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, aprime_grid, a_grid, ReturnFnParamsVec, Level)
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';
N_d = prod(n_d);
N_a = length(a_grid); % Because l_a=1

l_d = length(n_d);
if N_d == 0
    l_d = 0;
end

if l_d > 4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)');
end

if Level == 1 || Level == 4 || Level == 5
    N_aprime = size(aprime_grid, 1);
    aprime_grid = shiftdim(aprime_grid, -1);
elseif Level == 2 || Level == 3 || Level == 6
    N_aprime = size(aprime_grid, 2);
end

% Build dynamic parameters
d_vals = cell(1, l_d);
for i = 1:l_d
    d_vals{i} = d_gridvals(:, i);
end

GridParamsCell = [d_vals, {aprime_grid}, {shiftdim(a_grid, -2)}];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

% Reshape dynamically based on Level and presence of d
if l_d == 0
    Fmatrix = reshape(Fmatrix, [N_aprime, N_a]);
else
    if Level == 2 || Level == 5
        Fmatrix = reshape(Fmatrix, [N_d * N_aprime, N_a]);
    else
        Fmatrix = reshape(Fmatrix, [N_d, N_aprime, N_a]);
    end
end


end