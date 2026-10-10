function Fmatrix = CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d2, n_a2, n_a3, n_z, d_gridvals, a1prime_grid, a2prime_gridvals, a1_grid, a2_gridvals, a3_gridvals, z_gridvals, ReturnFnParamsVec, Level)
% Divide-and-conquer in the first standard endogenous state (a1, single dim).
% Folded standard middle endogenous states (a2, may be multi-dim, l_a2 up to 3).
% Last endogenous state is an experience asset (a3, may be multi-dim, l_a3 in {1,2}).
%
% Note: d_gridvals is both d1 and d2 (unless n_d1=0 so there is no d1, in which case is just d2)
% a1: standard endogenous state which will have DC applied
% a2: standard endogenous state
% a3: experienceasset
%
% Level==1: n-monotonicity sweep (a1prime full, a1 at level1 subset).
%   Output shape: [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_z].
% Level==2: narrow-band sweep, direct max (j=N_j no-V_Jplus1 case).
%   Output shape: [N_d*N_a1prime*N_a2prime, N_a1*N_a2*N_a3, N_z].
% Level==3: narrow-band sweep, multi-D output.
%   Output shape: [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_z].

ReturnFnParamsCell = num2cell(ReturnFnParamsVec)';

% Safely calculate dimensions for n_d combination
N_d1_raw = prod(n_d1);
N_d2_raw = prod(n_d2);

if N_d1_raw == 0 && N_d2_raw == 0
    n_d = 0;
elseif N_d1_raw == 0
    n_d = n_d2;
elseif N_d2_raw == 0
    n_d = n_d1;
else
    n_d = [n_d1, n_d2];
end

% Safely floor N_ parameters to 1 for final reshaping
N_d  = max(prod(n_d), 1);
N_a2 = max(prod(n_a2), 1);
N_a3 = max(prod(n_a3), 1);
N_z  = max(prod(n_z), 1);

l_d  = length(n_d);
l_a2 = length(n_a2);
l_a3 = length(n_a3);
l_z  = length(n_z);

% Pre-process a1prime based on Level
if Level == 1
    N_a1prime = size(a1prime_grid, 1);
    a1prime_grid = shiftdim(a1prime_grid, -1);
elseif Level == 2 || Level == 3
    N_a1prime = size(a1prime_grid, 2);
end

N_a2prime = N_a2;
N_a1 = size(a1_grid, 1);

%% Build dynamic parameters using cell arrays and shiftdim for exact GPU broadcasting

% 1. d_vals (Dim 1: no shift)
d_vals = cell(1, l_d);
for i = 1:l_d
    if l_d == 1; v = d_gridvals; else; v = d_gridvals(:, i); end
    d_vals{i} = v; 
end

% 2. a1prime (Dim 2: shifted by -1 handled in Level block)
a1prime_vals = {a1prime_grid}; 

% 3. a2prime (Dim 3: shift -2)
a2prime_vals = cell(1, l_a2);
for i = 1:l_a2
    if l_a2 == 1; v = a2prime_gridvals; else; v = a2prime_gridvals(:, i); end
    a2prime_vals{i} = shiftdim(v, -2);
end

% 4. a1 (Dim 4: shift -3)
a1_vals = {shiftdim(a1_grid, -3)};

% 5. a2 (Dim 5: shift -4)
a2_vals = cell(1, l_a2);
for i = 1:l_a2
    if l_a2 == 1; v = a2_gridvals; else; v = a2_gridvals(:, i); end
    a2_vals{i} = shiftdim(v, -4);
end

% 6. a3 (Dim 6: shift -5)
a3_vals = cell(1, l_a3);
for i = 1:l_a3
    if l_a3 == 1; v = a3_gridvals; else; v = a3_gridvals(:, i); end
    a3_vals{i} = shiftdim(v, -5);
end

% 7. z (Dim 7: shift -6)
z_vals = cell(1, l_z);
for i = 1:l_z
    if l_z == 1; v = z_gridvals; else; v = z_gridvals(:, i); end
    z_vals{i} = shiftdim(v, -6);
end

% Concatenate all parameter cells into a single list
GridParamsCell = [d_vals, a1prime_vals, a2prime_vals, a1_vals, a2_vals, a3_vals, z_vals];

%% Evaluate
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

%% Reshape Output
if Level == 1 || Level == 3
    Fmatrix = reshape(Fmatrix, [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_z]);
elseif Level == 2
    Fmatrix = reshape(Fmatrix, [N_d * N_a1prime * N_a2prime, N_a1 * N_a2 * N_a3, N_z]);
end


end
