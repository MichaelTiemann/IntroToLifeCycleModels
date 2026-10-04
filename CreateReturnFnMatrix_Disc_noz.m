function Fmatrix = CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec, Refine)
if nargin < 7
    Refine = 0;
end

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

N_d = prod(n_d);
N_a = prod(n_a);

l_d = length(n_d);
if N_d == 0
    l_d = 0;
end
l_a = length(n_a);

if l_d > 4 || l_a > 4
    error('Using GPU for the return fn does not allow for more than four of d or a variables');
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
d_vals = cell(1, l_d);
for i = 1:l_d
    d_vals{i} = d_gridvals(:, i);
end

% 3. Assemble GridParamsCell and execute
GridParamsCell = [d_vals, a_prime_vals, a_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

% 4. Reshape
if l_d == 0
    Fmatrix = reshape(Fmatrix, [N_a, N_a]);
else
    if Refine == 1
        Fmatrix = reshape(Fmatrix, [N_d, N_a, N_a]);
    else
        Fmatrix = reshape(Fmatrix, [N_d * N_a, N_a]);
    end
end
end