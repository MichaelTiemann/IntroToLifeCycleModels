function Fmatrix = CreateReturnFnMatrix_UDisc_DC2A(ReturnFn, n_d, n_z, n_e, d_gridvals, a1prime_grid, a2prime_grid, a1_grid, a2_grid, z_gridvals, e_gridvals, ReturnFnParamsVec, Level, ~)

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

N_d_raw = prod(n_d);
N_d = max(N_d_raw, 1);
N_a1 = length(a1_grid);
N_a2 = length(a2_grid);
N_z_raw = prod(n_z);
N_z = max(N_z_raw, 1);
N_e_raw = prod(n_e);
N_e = max(N_e_raw, 1);

if Level == 1 || Level == 4 || Level == 5
    N_a1prime = size(a1prime_grid, 1);
    a1prime_arg = {shiftdim(a1prime_grid, -1)};
elseif Level == 2 || Level == 3 || Level == 6
    N_a1prime = size(a1prime_grid, 2);
    a1prime_arg = {a1prime_grid};
end

a2prime_arg = {shiftdim(a2prime_grid, -2)};
a1_arg = {shiftdim(a1_grid, -3)};
a2_arg = {shiftdim(a2_grid, -4)};

% Build dynamic parameters
if N_d_raw == 0 || isempty(d_gridvals)
    d_args = {};
else
    d_args = num2cell(d_gridvals, 1);
end

if N_z_raw == 0 || isempty(z_gridvals)
    z_args = {};
else
    z_args = cellfun(@(x) shiftdim(x, -5), num2cell(z_gridvals, 1), 'UniformOutput', false);
end

if N_e_raw == 0 || isempty(e_gridvals)
    e_args = {};
else
    e_args = cellfun(@(x) shiftdim(x, -6), num2cell(e_gridvals, 1), 'UniformOutput', false);
end

% Combine all inputs
all_inputs = [d_args, a1prime_arg, a2prime_arg, a1_arg, a2_arg, z_args, e_args, ReturnFnParamsCell'];

% Evaluate function using arrayfun
Fmatrix = arrayfun(ReturnFn, all_inputs{:});

% Reshape output matrix
if Level == 2 || Level == 5
    Fmatrix = reshape(Fmatrix, [N_d * N_a1prime * N_a2, N_a1, N_a2, N_z, N_e]);
else
    Fmatrix = reshape(Fmatrix, [N_d, N_a1prime, N_a2, N_a1, N_a2, N_z, N_e]);
end


end
