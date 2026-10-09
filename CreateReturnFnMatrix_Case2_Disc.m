function Fmatrix = CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_gridvals, z_gridvals, ReturnFnParamsVec)

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

args = {};

% 1. d variables (dim 1)
col_idx = 1;
for i = 1:length(n_d)
    if n_d(i) > 0
        args{end+1} = d_gridvals(:, col_idx);
        col_idx = col_idx + 1;
    end
end

% 2. a variables (dim 2)
col_idx = 1;
for i = 1:length(n_a)
    if n_a(i) > 0
        args{end+1} = shiftdim(a_gridvals(:, col_idx), -1);
        col_idx = col_idx + 1;
    end
end

% 3. z variables (dim 3)
col_idx = 1;
for i = 1:length(n_z)
    if n_z(i) > 0
        args{end+1} = shiftdim(z_gridvals(:, col_idx), -2);
        col_idx = col_idx + 1;
    end
end

% Execute arrayfun dynamically
Fmatrix = arrayfun(ReturnFn, args{:}, ReturnFnParamsCell{:});

% Reshape (Filter out 0s so empty grids become size 1 for tensor dimensions)
N_d = max(1, prod(n_d(n_d > 0)));
N_a = max(1, prod(n_a(n_a > 0)));
N_z = max(1, prod(n_z(n_z > 0)));

Fmatrix = reshape(Fmatrix, [N_d, N_a, N_z]);


end
