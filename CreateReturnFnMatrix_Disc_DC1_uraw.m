function Fmatrix = CreateReturnFnMatrix_Disc_DC1_uraw(ReturnFn, n_d, n_z, n_e, d_gridvals, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec, Level)

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

N_d_raw = prod(n_d);
N_d = max(N_d_raw, 1);
N_a = length(a_grid);
N_z_raw = prod(n_z);
N_z = max(N_z_raw, 1);
N_e_raw = prod(n_e);
N_e = max(N_e_raw, 1);

if Level == 1 || Level == 4 || Level == 5
    N_aprime = size(aprime_grid, 1);
    aprime_arg = {shiftdim(aprime_grid, -1)};
elseif Level == 2 || Level == 3 || Level == 6
    N_aprime = size(aprime_grid, 2);
    aprime_arg = {aprime_grid};
end

% Build dynamic parameters
if N_d_raw == 0 || isempty(d_gridvals)
    d_args = {};
else
    d_args = num2cell(d_gridvals, 1);
end

a_arg = {shiftdim(a_grid, -2)};

if N_z_raw == 0 || isempty(z_gridvals)
    z_args = {};
else
    z_args = cellfun(@(x) shiftdim(x, -3), num2cell(z_gridvals, 1), 'UniformOutput', false);
end

if N_e_raw == 0 || isempty(e_gridvals)
    e_args = {};
else
    e_args = cellfun(@(x) shiftdim(x, -4), num2cell(e_gridvals, 1), 'UniformOutput', false);
end

% Combine all inputs
all_inputs = [d_args, aprime_arg, a_arg, z_args, e_args, ReturnFnParamsCell'];

% Evaluate function using arrayfun
Fmatrix = arrayfun(ReturnFn, all_inputs{:});

% Reshape output matrix
if Level == 2 || Level == 5
    Fmatrix = reshape(Fmatrix, [N_d * N_aprime, N_a, N_z, N_e]);
else
    Fmatrix = reshape(Fmatrix, [N_d, N_aprime, N_a, N_z, N_e]);
end


end
