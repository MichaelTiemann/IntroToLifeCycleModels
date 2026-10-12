function Fmatrix = CreateReturnFnMatrix_UDisc(ReturnFn, n_d, n_a, n_bothz, n_e, d_gridvals, a_grid, bothz_grid, e_grid, ReturnFnParamsVec, Refine)
if nargin < 11; Refine = 0; end
ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';
GridParamsCell = {};

l_a = length(n_a);
N_a = max(prod(n_a), 1);
N_d = size(d_gridvals, 1);
N_bothz = size(bothz_grid, 1); if N_bothz == 0; N_bothz = 1; end
N_e = size(e_grid, 1); if N_e == 0; N_e = 1; end

% Dim 1: Decisions
if sum(n_d) > 0
    for i = 1:size(d_gridvals, 2); GridParamsCell{end+1} = d_gridvals(:, i); end
end

% Dim 2 to (1+l_a): Future Endogenous States (a')
a_cum = 0;
for i = 1:l_a
    n_ai = n_a(i);
    idx_range = (a_cum + 1):(a_cum + n_ai);
    GridParamsCell{end+1} = shiftdim(a_grid(idx_range), -i);
    a_cum = a_cum + n_ai;
end

% Dim (2+l_a) to (1+2*l_a): Current Endogenous States (a)
a_cum = 0;
for i = 1:l_a
    n_ai = n_a(i);
    idx_range = (a_cum + 1):(a_cum + n_ai);
    GridParamsCell{end+1} = shiftdim(a_grid(idx_range), -(l_a + i));
    a_cum = a_cum + n_ai;
end

% Dim (2+2*l_a): Markov Shocks (bothz)
if sum(n_bothz) > 0
    for i = 1:size(bothz_grid, 2); GridParamsCell{end+1} = shiftdim(bothz_grid(:, i), -(2*l_a + 1)); end
end

% Dim (3+2*l_a): IID Shocks (e)
if sum(n_e) > 0
    for i = 1:size(e_grid, 2); GridParamsCell{end+1} = shiftdim(e_grid(:, i), -(2*l_a + 2)); end
end

Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

% Reshape output array
if sum(n_d) == 0
    Fmatrix = reshape(Fmatrix, [N_a, N_a, N_bothz, N_e]);
else
    if Refine == 1
        Fmatrix = reshape(Fmatrix, [N_d, N_a, N_a, N_bothz, N_e]);
    else
        Fmatrix = reshape(Fmatrix, [N_d * N_a, N_a, N_bothz, N_e]);
    end
end


end
