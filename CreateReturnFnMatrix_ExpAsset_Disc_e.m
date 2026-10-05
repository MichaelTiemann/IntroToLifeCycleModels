function Fmatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1prime, n_a1,n_a2, n_z,n_e, d_gridvals, a1prime_gridvals, a1_gridvals, a2_gridvals, z_gridvals, e_gridvals, ReturnFnParamsVec,Level,Refine)
% Note: d_gridvals is both d1 and d2 (unless n_d1=1 so there is no d1, in which case is just d2)
% a1: standard endogenous state
% a2: experienceasset
% Level and Refine are about different shapes of inputs/output
% Set Level=0, unless using Divide-and-Conquer
% Refine=1 splits d1 out as the leading dimension (useful for when EV doesn't depend on d1).
% Refine=0 keeps d stacked as before.

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d1_raw = prod(n_d1);
N_d2_raw = prod(n_d2);

if N_d1_raw == 0 && N_d2_raw == 0
    n_d = 0;
elseif N_d1_raw == 0
    n_d = n_d2;
elseif N_d2_raw == 0
    n_d = n_d1;
else
    n_d = [n_d1, n_d2]; % Almost everything is done without distinguishing d1 and d2, just for some reshapes at the end
end

Nd_eff = max(prod(n_d), 1);
Nd1_eff = max(N_d1_raw, 1);
Nd2_eff = max(N_d2_raw, 1);
N_a1prime = max(prod(n_a1prime), 1);
N_a1 = max(prod(n_a1), 1);
N_a2 = max(prod(n_a2), 1);
Nz_eff = max(prod(n_z), 1);
N_e = prod(n_e);

l_d = length(n_d); if prod(n_d)==0; l_d=0; end
l_a1 = length(n_a1); if prod(n_a1)==0; l_a1=0; end
l_a2 = length(n_a2);
l_z = length(n_z); if prod(n_z)==0; l_z=0; end
l_e = length(n_e); if prod(n_e)==0; l_e=0; end

if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a1>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_a2>2
    error('experienceasset currently supports length(n_a2) in {1,2}')
end
if l_z>8
    error('Using GPU for the return fn does not allow for more than eight of semiz and z variables')
end
if l_e>5
    error('Using GPU for the return fn does not allow for more than five of e variable (you have length(n_e)>5)')
end

% Build dynamic parameters (preserve N-dimensional arrays natively when l_x==1)
if l_d == 0
    d_vals = {};
else
    d_vals = cell(1, l_d);
    for i = 1:l_d
        if l_d == 1; v = d_gridvals; else; v = d_gridvals(:, i); end
        d_vals{i} = v;
    end
end

if l_a1 == 0
    a1prime_vals = {};
    a1_vals= {};
else
    a1prime_vals = cell(1, l_a1);
    for i = 1:l_a1
        if l_a1 == 1; v = a1prime_gridvals; else; v = a1prime_gridvals(:, i); end
        if Level == 0 || Level == 1
            a1prime_vals{i} = shiftdim(v, -1);
        else % Level 2 or 3
            a1prime_vals{i} = v;
        end
    end
    
    a1_vals = cell(1, l_a1);
    for i = 1:l_a1
        if l_a1 == 1; v = a1_gridvals; else; v = a1_gridvals(:, i); end
        a1_vals{i} = shiftdim(v, -2);
    end
end

a2_vals = cell(1, l_a2);
for i = 1:l_a2
    if l_a2 == 1; v = a2_gridvals; else; v = a2_gridvals(:, i); end
    a2_vals{i} = shiftdim(v, -3);
end

if l_z == 0
    z_vals = {};
else
    z_vals = cell(1, l_z);
    for i = 1:l_z
        if l_z == 1; v = z_gridvals; else; v = z_gridvals(:, i); end
        z_vals{i} = shiftdim(v, -4);
    end
end

e_vals = cell(1, l_e);
for i = 1:l_e
    if l_e == 1; v = e_gridvals; else; v = e_gridvals(:, i); end
    e_vals{i} = shiftdim(v, -5);
end

GridParamsCell = [d_vals, a1prime_vals, a1_vals, a2_vals, z_vals, e_vals];
Fmatrix = arrayfun(ReturnFn, GridParamsCell{:}, ReturnFnParamsCell{:});

% Reshape
if Level == 0 || Level == 2
    if Refine == 0 || prod(n_d1) == 0
        Fmatrix = reshape(Fmatrix, [Nd_eff * N_a1prime, N_a1 * N_a2, Nz_eff, N_e]);
    elseif Refine == 1
        Fmatrix = reshape(Fmatrix, [Nd1_eff, Nd2_eff * N_a1prime, N_a1 * N_a2, Nz_eff, N_e]);
    end
elseif Level == 1 || Level == 3
    if Refine == 0 || prod(n_d1) == 0
        Fmatrix = reshape(Fmatrix, [Nd_eff, N_a1prime, N_a1, N_a2, Nz_eff, N_e]);
    elseif Refine == 1
        Fmatrix = reshape(Fmatrix, [Nd1_eff, Nd2_eff * N_a1prime, N_a1, N_a2, Nz_eff, N_e]);
    end
end


end
