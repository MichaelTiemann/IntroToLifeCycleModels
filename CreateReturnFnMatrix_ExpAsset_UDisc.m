function Fmatrix = CreateReturnFnMatrix_ExpAsset_UDisc(ReturnFn, n_d, n_a1, n_a2, n_bothz, n_e, d_gridvals, a1prime_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals, e_gridvals, ReturnFnParamsVec, Level)

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';
GridParamsCell = {};

N_d     = max(size(d_gridvals, 1), 1);
N_a1p   = max(size(a1prime_gridvals, 1), 1);
N_a1    = max(size(a1_gridvals, 1), 1);
N_a2    = max(size(a2_gridvals, 1), 1);
N_bothz = max(size(bothz_gridvals, 1), 1);
N_e     = max(size(e_gridvals, 1), 1);

% Dim 1: Decisions (d)
if sum(n_d) > 0
    for i = 1:size(d_gridvals, 2)
        GridParamsCell{end+1} = d_gridvals(:, i);
    end
end

% Dim 2 & Dim 3: Next-period and current-period standard endogenous states (a1', a1)
if sum(n_a1) > 0
    for i = 1:size(a1prime_gridvals, 2)
        GridParamsCell{end+1} = shiftdim(a1prime_gridvals(:, i), -1);
    end
    for i = 1:size(a1_gridvals, 2)
        GridParamsCell{end+1} = shiftdim(a1_gridvals(:, i), -2);
    end
end

% Dim 4: Experience / Risky asset state (a2)
if sum(n_a2) > 0
    for i = 1:size(a2_gridvals, 2)
        GridParamsCell{end+1} = shiftdim(a2_gridvals(:, i), -3);
    end
end

% Dim 5: Combined Markov & Semi-Exogenous shocks (bothz)
if sum(n_bothz) > 0
    for i = 1:size(bothz_gridvals, 2)
        GridParamsCell{end+1} = shiftdim(bothz_gridvals(:, i), -4);
    end
end

% Dim 6: i.i.d shocks (e)
if sum(n_e) > 0
    for i = 1:size(e_gridvals, 2)
        GridParamsCell{end+1} = shiftdim(e_gridvals(:, i), -5);
    end
end

Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

if Level == 1
    Fmatrix = reshape(Fmatrix, [N_d, N_a1p, N_a1, N_a2, N_bothz, N_e]);
else
    Fmatrix = reshape(Fmatrix, [N_d * N_a1p, N_a1 * N_a2, N_bothz, N_e]);
end

end
