function Fmatrix = CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, aprime_grid, a_grid, z_gridvals, ReturnFnParamsVec, Level)
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';
N_d = prod(n_d);
N_a = length(a_grid);
N_z = prod(n_z);

l_d = length(n_d); if N_d == 0; l_d = 0; end
l_z = length(n_z); if N_z == 0; l_z = 0; end

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
if l_d == 0
    d_vals = {};
    Nd_eff = 1;
else
    d_vals = cell(1, l_d);
    for i = 1:l_d; d_vals{i} = d_gridvals(:, i); end
    Nd_eff = N_d;
end

if l_z == 0
    z_vals = {};
    Nz_eff = 1;
else
    z_vals = cell(1, l_z);
    for i = 1:l_z; z_vals{i} = shiftdim(z_gridvals(:, i), -3); end
    Nz_eff = N_z;
end

GridParamsCell = [d_vals, {aprime_grid}, {shiftdim(a_grid, -2)}, z_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

if Level == 2 || Level == 5
    Fmatrix = reshape(Fmatrix, [Nd_eff * N_aprime, N_a, Nz_eff]);
else
    Fmatrix = reshape(Fmatrix, [Nd_eff, N_aprime, N_a, Nz_eff]);
end


end