function Fmatrix = CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, aprime_grid, a_grid, z_gridvals, ReturnFnParamsVec, Level)
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';
N_d = prod(n_d);
N_a = length(a_grid);
N_z = prod(n_z);

l_d = length(n_d);
if N_d == 0; l_d = 0; end
l_z = length(n_z);

if l_d > 4 || l_z > 4
    error('Using GPU for the return fn does not allow for more than four of d or z variables');
end

if Level == 1 || Level == 4 || Level == 5
    N_aprime = size(aprime_grid, 1);
    aprime_grid = shiftdim(aprime_grid, -1);
elseif Level == 2 || Level == 3 || Level == 6
    N_aprime = size(aprime_grid, 2);
end

% Build dynamic parameters
d_vals = cell(1, l_d);
for i = 1:l_d; d_vals{i} = d_gridvals(:, i); end

z_vals = cell(1, l_z);
for i = 1:l_z; z_vals{i} = shiftdim(z_gridvals(:, i), -3); end

GridParamsCell = [d_vals, {aprime_grid}, {shiftdim(a_grid, -2)}, z_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

if l_d == 0
    Fmatrix = reshape(Fmatrix, [N_aprime, N_a, N_z]);
else
    if Level == 2 || Level == 5
        Fmatrix = reshape(Fmatrix, [N_d * N_aprime, N_a, N_z]);
    else
        Fmatrix = reshape(Fmatrix, [N_d, N_aprime, N_a, N_z]);
    end
end


end