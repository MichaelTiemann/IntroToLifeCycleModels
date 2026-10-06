function Fmatrix=CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d, n_z, n_e, d_gridvals, a1prime_grid, a2prime_grid, a1_grid, a2_grid, z_gridvals, e_gridvals, ReturnFnParamsVec, Level, Refine)
% Refine=1 at Level=1 collapses N_d into the N_a1prime row (useful when d is singular, e.g. inside a d2_c loop with special_n_d2=ones, so downstream can treat output like the _nod variant).
% Refine=1 at Level=2 keeps N_a1 and N_a2 as separate dims (useful for broadcasting an EV that has the level1iidiff axis as singleton).
% Refine=0 keeps the default shapes.

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d = max(1, prod(n_d));
N_a1 = length(a1_grid);
N_a2 = length(a2_grid);
N_z = max(1, prod(n_z));
N_e = max(1, prod(n_e));

if Level==1
    N_a1prime=length(a1prime_grid); % Because l_a=1
    a1prime_grid=shiftdim(a1prime_grid,-1);
elseif Level==2 || Level==3
    N_a1prime=size(a1prime_grid,2); % Because l_a=1
    % Level 3 has level 2 inputs but level 1 outputs, used for GI
% elseif Level==4
elseif Level==5 % Level 2 inputs, but for doing semiz without d1, so d2 is singular inside the loop over d2
    N_a1prime=size(a1prime_grid,1);
    a1prime_grid=shiftdim(a1prime_grid,-1); % extra -1 for the singular d2
end
N_a2prime=N_a2;

if isempty(d_gridvals); d_args = {}; else; d_args = num2cell(d_gridvals, 1); end
if isempty(z_gridvals); z_args = {}; else; z_args = cellfun(@(x) shiftdim(x, -5), num2cell(z_gridvals, 1), 'UniformOutput', false); end
if isempty(e_gridvals); e_args = {}; else; e_args = cellfun(@(x) shiftdim(x, -6), num2cell(e_gridvals, 1), 'UniformOutput', false); end

Fmatrix = arrayfun(ReturnFn, d_args{:}, a1prime_grid, shiftdim(a2prime_grid,-2), ...
    shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_args{:}, e_args{:}, ReturnFnParamsCell{:});


end
