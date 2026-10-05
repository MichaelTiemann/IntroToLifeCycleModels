function Fmatrix = CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec, Level)

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

N_d = prod(n_d);
N_a = length(a_grid);
N_z = prod(n_z);
N_e = prod(n_e);

l_d = length(n_d); if N_d == 0; l_d = 0; end
l_z = length(n_z); if N_z == 0; l_z = 0; end
l_e = length(n_e);

if l_d > 4 || l_z > 4 || l_e > 4
    error('Using GPU for the return fn does not allow for more than four of d, z, or e variables');
end

if Level == 1 || Level == 4 || Level == 5 || Level == 3
    N_aprime = size(aprime_grid, 1);
    if Level ~= 3
        aprime_grid = shiftdim(aprime_grid, -1);
    end
elseif Level == 2 || Level == 6
    N_aprime = size(aprime_grid, 2);
end

if l_d == 0
    d_vals = {};
else
    d_vals = cell(1, l_d); for i = 1:l_d; d_vals{i} = d_gridvals(:, i); end
end

if l_z == 0
    z_vals = {};
else
    z_vals = cell(1, l_z); for i = 1:l_z; z_vals{i} = shiftdim(z_gridvals(:, i), -3); end
end

e_vals = cell(1, l_e); for i = 1:l_e; e_vals{i} = shiftdim(e_gridvals(:, i), -4); end

GridParamsCell = [d_vals, {aprime_grid}, {shiftdim(a_grid, -2)}, z_vals, e_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

Nd_eff = max(N_d, 1);
Nz_eff = max(N_z, 1);

if Level == 2 || Level == 5
    Fmatrix = reshape(Fmatrix, [Nd_eff * N_aprime, N_a, Nz_eff, N_e]);
else
    Fmatrix = reshape(Fmatrix, [Nd_eff, N_aprime, N_a, Nz_eff, N_e]);
end


end