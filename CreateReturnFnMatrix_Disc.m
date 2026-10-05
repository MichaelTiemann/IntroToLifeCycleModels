function Fmatrix = CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec, Refine)
% If no d variable, just input n_d=0 and d_gridvals=[]

if nargin < 9
    Refine = 0;
end

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);

l_d = length(n_d); if N_d == 0; l_d = 0; end
l_a = length(n_a);
l_z = length(n_z); if N_z == 0; l_z = 0; end

% Validation checks
if l_d > 4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)');
end
if l_a > 4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)');
end
if l_z > 5
    error('Using GPU for the return fn does not allow for more than five of z variable (you have length(n_z)>5)');
end

% Collect 'a' and 'a_prime' values dynamically
a_prime_vals = cell(1, l_a);
a_vals = cell(1, l_a);
a_cum = 0;

for i = 1:l_a
    n_ai = n_a(i);
    idx_range = (a_cum + 1):(a_cum + n_ai);
    a_prime_vals{i} = shiftdim(a_grid(idx_range), -i);
    a_vals{i} = shiftdim(a_grid(idx_range), -(l_a + i));
    a_cum = a_cum + n_ai;
end

% Collect 'd' values dynamically
if l_d == 0
    d_vals = {};
else
    d_vals = cell(1, l_d);
    for i = 1:l_d
        d_vals{i} = d_gridvals(:, i);
    end
end

% Collect 'z' values dynamically
if l_z == 0
    z_vals = {};
    Nz_eff = 1;
else
    z_vals = cell(1, l_z);
    z_shift_base = 2 * l_a;
    for i = 1:l_z
        z_vals{i} = shiftdim(z_gridvals(:, i), -(1 + z_shift_base));
    end
    Nz_eff = N_z;
end

% Combine all inputs into a single argument cell array for arrayfun
all_inputs = [d_vals, a_prime_vals, a_vals, z_vals, ReturnFnParamsCell'];

% Evaluate function using arrayfun
Fmatrix = arrayfun(ReturnFn, all_inputs{:});

% Reshape output matrix
if l_d == 0
    Fmatrix = reshape(Fmatrix, [N_a, N_a, Nz_eff]);
else
    if Refine == 1
        Fmatrix = reshape(Fmatrix, [N_d, N_a, N_a, Nz_eff]);
    else
        Fmatrix = reshape(Fmatrix, [N_d * N_a, N_a, Nz_eff]);
    end
end
end