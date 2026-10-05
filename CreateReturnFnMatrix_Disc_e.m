function Fmatrix = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, ReturnFnParamsVec, Refine)
if nargin < 11
    Refine = 0;
end

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);
N_e = prod(n_e);

l_d = length(n_d); if N_d == 0; l_d = 0; end
l_a = length(n_a);
l_z = length(n_z); if N_z == 0; l_z = 0; end
l_e = length(n_e);

if l_d > 4 || l_a > 4 || l_z > 5 || l_e > 5
    error('Variable dimension limits exceeded for GPU return fn evaluation.');
end

% 1. Dynamically build a_prime and a
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

% 2. Dynamically build d
if l_d == 0
    d_vals = {};
else
    d_vals = cell(1, l_d); for i = 1:l_d; d_vals{i} = d_gridvals(:, i); end
end

% 3. Dynamically build z and e with their proper shift bases
if l_z == 0
    z_vals = {};
else
    z_vals = cell(1, l_z);
    z_shift_base = 2 * l_a;
    for i = 1:l_z
        z_vals{i} = shiftdim(z_gridvals(:, i), -(1 + z_shift_base));
    end
end

e_vals = cell(1, l_e);
e_shift_base = 2 * l_a + 1;
for i = 1:l_e
    e_vals{i} = shiftdim(e_gridvals(:, i), -(1 + e_shift_base));
end

% 4. Assemble GridParamsCell and execute
GridParamsCell = [d_vals, a_prime_vals, a_vals, z_vals, e_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

Nd_eff = max(N_d, 1);
Nz_eff = max(N_z, 1);

% 5. Reshape
if l_d == 0
    Fmatrix = reshape(Fmatrix, [N_a, N_a, Nz_eff, N_e]);
else
    if Refine == 1
        Fmatrix = reshape(Fmatrix, [Nd_eff, N_a, N_a, Nz_eff, N_e]);
    else
        Fmatrix = reshape(Fmatrix, [Nd_eff * N_a, N_a, Nz_eff, N_e]);
    end
end


end