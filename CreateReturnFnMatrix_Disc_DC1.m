function Fmatrix = CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d, n_z, d_gridvals, aprime_grid, a_grid, z_gridvals, ReturnFnParamsVec, Level)

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';
N_d = max(1, prod(n_d));
N_a = length(a_grid);
N_z = max(1, prod(n_z));

if Level == 1 || Level == 4 || Level == 5
    N_aprime = size(aprime_grid, 1);
    aprime_grid = shiftdim(aprime_grid, -1);
elseif Level == 2 || Level == 3 || Level == 6
    N_aprime = size(aprime_grid, 2);
end

% Build dynamic parameters
if isempty(d_gridvals); d_args = {}; else; d_args = num2cell(d_gridvals, 1); end
if isempty(z_gridvals); z_args = {}; else; z_args = cellfun(@(x) shiftdim(x, -3), num2cell(z_gridvals, 1), 'UniformOutput', false); end

Fmatrix = arrayfun(ReturnFn, d_args{:}, aprime_grid, shiftdim(a_grid, -2), z_args{:}, ReturnFnParamsCell{:});

if Level == 2 || Level == 5
    Fmatrix = reshape(Fmatrix, [N_d * N_aprime, N_a, N_z]);
else
    Fmatrix = reshape(Fmatrix, [N_d, N_aprime, N_a, N_z]);
end


end