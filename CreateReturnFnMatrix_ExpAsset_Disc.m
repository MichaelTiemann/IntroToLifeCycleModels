function Fmatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1, n_d2, n_a1prime, n_a1,n_a2, n_z, d_gridvals, a1prime_gridvals, a1_gridvals, a2_gridvals, z_gridvals, ReturnFnParamsVec,Level,Refine)
% Note: d_gridvals is both d1 and d2 (unless n_d1=1 so there is no d1, in which case is just d2)
% a1: standard endogenous state
% a2: experienceasset
% Level and Refine are about different shapes of inputs/output
% Set Level=0, unless using Divide-and-Conquer
% Refine=1 splits d1 out as the leading dimension (useful for when EV doesn't depend on d1).
% Refine=0 keeps d stacked as before.

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

if n_d1(1) == 0
    n_d = n_d2;
else
    n_d = [n_d1, n_d2];
end

N_d = max(prod(n_d), 1);
N_d1 = max(prod(n_d1), 1);
N_d2 = max(prod(n_d2), 1);
N_a1prime = max(prod(n_a1prime), 1);
N_a1 = max(prod(n_a1), 1);
N_a2 = max(prod(n_a2), 1);
N_z = max(prod(n_z), 1);

l_d = length(n_d); if prod(n_d)==0; l_d=0; end
l_a1 = length(n_a1); if prod(n_a1)==0; l_a1=0; end
l_a2 = length(n_a2);
l_z = length(n_z); if prod(n_z)==0; l_z=0; end

if l_d > 4 || l_a1 > 4 || l_z > 8
    error('Using GPU for the return fn does not allow for more than 4 d, 4 a, or 8 z variables');
end
if l_a2 > 2
    error('experienceasset currently supports length(n_a2) in {1,2}');
end

% Build dynamic parameters
d_vals = cell(1, l_d);
for i = 1:l_d; d_vals{i} = d_gridvals(:, i); end

a1prime_vals = cell(1, l_a1);
for i = 1:l_a1
    if Level == 0 || Level == 1
        a1prime_vals{i} = shiftdim(a1prime_gridvals(:, i), -1);
    else % Level 2 or 3
        a1prime_vals{i} = a1prime_gridvals(:, i);
    end
end

a1_vals = cell(1, l_a1);
for i = 1:l_a1; a1_vals{i} = shiftdim(a1_gridvals(:, i), -2); end

a2_vals = cell(1, l_a2);
for i = 1:l_a2; a2_vals{i} = shiftdim(a2_gridvals(:, i), -3); end

z_vals = cell(1, l_z);
for i = 1:l_z; z_vals{i} = shiftdim(z_gridvals(:, i), -4); end

GridParamsCell = [d_vals, a1prime_vals, a1_vals, a2_vals, z_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

% Reshape
if Level == 0 || Level == 2
    if Refine == 0 || prod(n_d1) == 0
        Fmatrix = reshape(Fmatrix, [N_d * N_a1prime, N_a1 * N_a2, N_z]);
    elseif Refine == 1
        Fmatrix = reshape(Fmatrix, [N_d1, N_d2 * N_a1prime, N_a1 * N_a2, N_z]);
    end
elseif Level == 1 || Level == 3
    if Refine == 0 || prod(n_d1) == 0
        Fmatrix = reshape(Fmatrix, [N_d, N_a1prime, N_a1, N_a2, N_z]);
    elseif Refine == 1
        Fmatrix = reshape(Fmatrix, [N_d1, N_d2 * N_a1prime, N_a1, N_a2, N_z]);
    end
end


end
