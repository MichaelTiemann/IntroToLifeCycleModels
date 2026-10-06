function Fmatrix = CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d, n_z, n_e, d_gridvals, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec, Level)

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

N_d = max(1, prod(n_d));
N_a = length(a_grid);
N_z = max(1, prod(n_z));
N_e = prod(n_e);

if Level == 1
    N_aprime = length(aprime_grid);
    aprime_grid = shiftdim(aprime_grid, -1);
elseif Level == 2 || Level == 3
    N_aprime = size(aprime_grid, 2); 
end

% --- DYNAMIC ARRAYFUN EXPANSION ---
if isempty(d_gridvals) || prod(n_d) == 0; d_args = {}; else; d_args = num2cell(d_gridvals, 1); end
if isempty(z_gridvals) || prod(n_z) == 0; z_args = {}; else; z_args = cellfun(@(x) shiftdim(x, -3), num2cell(z_gridvals, 1), 'UniformOutput', false); end
e_args = cellfun(@(x) shiftdim(x, -4), num2cell(e_gridvals, 1), 'UniformOutput', false);

Fmatrix = arrayfun(ReturnFn, d_args{:}, aprime_grid, shiftdim(a_grid, -2), z_args{:}, e_args{:}, ReturnFnParamsCell{:});
% ----------------------------------

if Level == 2 || Level == 5
    Fmatrix = reshape(Fmatrix, [N_d * N_aprime, N_a, N_z, N_e]);
else
    Fmatrix = reshape(Fmatrix, [N_d, N_aprime, N_a, N_z, N_e]);
end


end