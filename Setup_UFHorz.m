function [Dims, Grids] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, vfoptions)

% --- 1. Endogenous States (a) & Decisions (d) ---
Dims.N_a = max(prod(n_a), 1);
Dims.N_d = max(prod(n_d), 1);
Dims.n_d = n_d;

% --- 2. Exogenous Shocks (z, semiz, e) ---
Dims.n_z     = n_z;
Dims.n_semiz = vfoptions.n_semiz;
Dims.n_e     = vfoptions.n_e;

Dims.N_z     = max(prod(n_z), 1);
Dims.N_semiz = max(prod(vfoptions.n_semiz), 1);
Dims.N_e     = max(prod(vfoptions.n_e), 1);

Dims.has_z     = prod(n_z) > 0;
Dims.has_semiz = prod(Dims.n_semiz) > 0;
Dims.has_e     = prod(Dims.n_e) > 0;
Dims.has_bothz = Dims.has_semiz || Dims.has_z;

if Dims.has_semiz && Dims.has_z
    Dims.n_bothz = [Dims.n_semiz, Dims.n_z];
elseif Dims.has_semiz
    Dims.n_bothz = Dims.n_semiz;
else
    Dims.n_bothz = Dims.n_z;
end
Dims.N_bothz = Dims.N_semiz * Dims.N_z;

% --- 3. Experience Asset Flags ---
Dims.exp_flag = 0;
if vfoptions.experienceasset >= 1;      Dims.exp_flag = 1; end
if vfoptions.experienceassetz >= 1;     Dims.exp_flag = 2; end
if vfoptions.experienceassete >= 1;     Dims.exp_flag = 3; end
if vfoptions.experienceassetze >= 1;    Dims.exp_flag = 4; end
if vfoptions.experienceassetsemiz >= 1; Dims.exp_flag = 5; end
if vfoptions.experienceassetu >= 1;     Dims.exp_flag = 6; end

% --- 4. Unpack Splits for Endogenous States and Decisions ---
if Dims.exp_flag > 0
    Dims.n_a1 = vfoptions.n_a1;
    Dims.n_a2 = vfoptions.n_a2;
    Dims.N_a1 = max(prod(Dims.n_a1), 1);
    Dims.N_a2 = max(prod(Dims.n_a2), 1);

    Dims.n_d1 = vfoptions.n_d1;
    Dims.n_d2 = vfoptions.n_d2;
    Dims.N_d1 = max(prod(Dims.n_d1), 1);
    Dims.N_d2 = max(prod(Dims.n_d2), 1);
    if isfield(vfoptions, 'n_d3') && prod(vfoptions.n_d3) > 0
        Dims.n_d3 = vfoptions.n_d3;
        Dims.N_d3 = max(prod(Dims.n_d3), 1);
    else
        Dims.n_d3 = 0;
        Dims.N_d3 = 1;
    end
    Dims.N_d_inner = Dims.N_d1 * Dims.N_d2;
    Dims.N_d_outer = Dims.N_d3;
else
    Dims.n_a1 = n_a;
    Dims.n_a2 = 0;
    Dims.N_a1 = Dims.N_a;
    Dims.N_a2 = 1;

    if isfield(vfoptions, 'n_d1') && isfield(vfoptions, 'n_d2')
        Dims.n_d1 = vfoptions.n_d1;
        Dims.n_d2 = vfoptions.n_d2;
    elseif Dims.has_semiz
        l_dsemiz = 1;
        if isfield(vfoptions, 'l_dsemiz'); l_dsemiz = vfoptions.l_dsemiz; end
        if length(n_d) > l_dsemiz
            Dims.n_d1 = n_d(1:end-l_dsemiz);
        else
            Dims.n_d1 = 0;
        end
        Dims.n_d2 = n_d(end-l_dsemiz+1:end);
    else
        Dims.n_d1 = n_d;
        Dims.n_d2 = 0;
    end
    Dims.N_d1 = max(prod(Dims.n_d1), 1);
    Dims.N_d2 = max(prod(Dims.n_d2), 1);
    Dims.n_d3 = 0;
    Dims.N_d3 = 1;
    Dims.N_d_inner = Dims.N_d1;
    Dims.N_d_outer = Dims.N_d2;
end

Dims.has_d1 = prod(Dims.n_d1) > 0;
Dims.has_d2 = prod(Dims.n_d2) > 0;

% --- 5. Special Dimension Vectors for CreateReturnFnMatrix Kernels ---
if isempty(n_d) || n_d(1) == 0
    Dims.special_n_d = 0;
elseif Dims.N_d_outer > 1
    if Dims.exp_flag > 0
        l_d3 = length(Dims.n_d3);
        if Dims.has_d1
            Dims.special_n_d = [Dims.n_d1, Dims.n_d2, ones(1, l_d3)];
        else
            Dims.special_n_d = [Dims.n_d2, ones(1, l_d3)];
        end
    else
        l_d2 = length(Dims.n_d2);
        if Dims.has_d1
            Dims.special_n_d = [Dims.n_d1, ones(1, l_d2)];
        else
            Dims.special_n_d = ones(1, l_d2);
        end
    end
else
    Dims.special_n_d = n_d;
end

Dims.special_n_z     = n_z;          if ~Dims.has_z;     Dims.special_n_z = 0;     end
Dims.special_n_semiz = Dims.n_semiz; if ~Dims.has_semiz; Dims.special_n_semiz = 0; end
Dims.special_n_e     = Dims.n_e;     if ~Dims.has_e;     Dims.special_n_e = 0;     end
Dims.special_n_bothz = Dims.n_bothz; if ~Dims.has_bothz; Dims.special_n_bothz = 0; end

% --- 6. Unified Iteration Strategy & Slice Dimensions ---
if vfoptions.lowmemory == 0
    IterStrategy = 'Full';
elseif vfoptions.lowmemory == 1
    if Dims.has_e; IterStrategy = 'Loop_e';
    elseif Dims.has_semiz && Dims.has_z; IterStrategy = 'Loop_z_parallel_semiz';
    else; IterStrategy = 'Loop_bothz'; end
elseif vfoptions.lowmemory == 2
    if Dims.has_e && Dims.has_semiz && Dims.has_z; IterStrategy = 'Loop_z_parallel_semiz_e';
    elseif Dims.has_e; IterStrategy = 'Loop_bothz_e';
    elseif Dims.has_semiz && Dims.has_z; IterStrategy = 'Loop_bothz';
    else; error('Invalid lowmemory=2 for this shock combination.'); end
elseif vfoptions.lowmemory == 3
    if Dims.has_e && Dims.has_semiz && Dims.has_z; IterStrategy = 'Loop_bothz_e';
    else; error('Invalid lowmemory=3 for this shock combination.'); end
end

ones_z  = ones(1, length(n_z));
ones_bz = ones(1, length(Dims.n_bothz)); if ~Dims.has_bothz; ones_bz = 0; end

switch IterStrategy
    case 'Full'
        Dims.N_z_iter = 1;           Dims.N_z_slice = Dims.N_bothz; Dims.z_mode = 0; Dims.spec_n_bz = Dims.special_n_bothz;
        Dims.N_e_iter = 1;           Dims.N_e_slice = Dims.N_e;     Dims.e_mode = 0; Dims.spec_n_e  = Dims.special_n_e;
    case 'Loop_e'
        Dims.N_z_iter = 1;           Dims.N_z_slice = Dims.N_bothz; Dims.z_mode = 0; Dims.spec_n_bz = Dims.special_n_bothz;
        Dims.N_e_iter = Dims.N_e;    Dims.N_e_slice = 1;            Dims.e_mode = 1; Dims.spec_n_e  = ones(1, length(Dims.n_e));
    case 'Loop_z_parallel_semiz'
        Dims.N_z_iter = Dims.N_z;    Dims.N_z_slice = Dims.N_semiz; Dims.z_mode = 1; Dims.spec_n_bz = [Dims.n_semiz, ones_z];
        Dims.N_e_iter = 1;           Dims.N_e_slice = Dims.N_e;     Dims.e_mode = 0; Dims.spec_n_e  = Dims.special_n_e;
    case 'Loop_z_parallel_semiz_e'
        Dims.N_z_iter = Dims.N_z;    Dims.N_z_slice = Dims.N_semiz; Dims.z_mode = 1; Dims.spec_n_bz = [Dims.n_semiz, ones_z];
        Dims.N_e_iter = Dims.N_e;    Dims.N_e_slice = 1;            Dims.e_mode = 1; Dims.spec_n_e  = ones(1, length(Dims.n_e));
    case 'Loop_bothz'
        Dims.N_z_iter = Dims.N_bothz; Dims.N_z_slice = 1;           Dims.z_mode = 2; Dims.spec_n_bz = ones_bz;
        Dims.N_e_iter = 1;            Dims.N_e_slice = Dims.N_e;    Dims.e_mode = 0; Dims.spec_n_e  = Dims.special_n_e;
    case 'Loop_bothz_e'
        Dims.N_z_iter = Dims.N_bothz; Dims.N_z_slice = 1;           Dims.z_mode = 2; Dims.spec_n_bz = ones_bz;
        Dims.N_e_iter = Dims.N_e;     Dims.N_e_slice = 1;           Dims.e_mode = 1; Dims.spec_n_e  = ones(1, length(Dims.n_e));
end

% --- 7. Synthesized Grids (d12_gridvals, bothz_gridvals_J, ExpAsset grids) ---
if Dims.N_d_outer > 1 && ~isempty(d_gridvals)
    Grids.d12_gridvals = reshape(d_gridvals, [Dims.N_d_inner, Dims.N_d_outer, size(d_gridvals, 2)]);
else
    Grids.d12_gridvals = d_gridvals;
end

if Dims.has_semiz && Dims.has_z
    rep_semiz = repmat(vfoptions.semiz_gridvals_J, [Dims.N_z, 1, 1]);
    rep_z     = repelem(z_gridvals_J, Dims.N_semiz, 1);
    Grids.bothz_gridvals_J = [rep_semiz, rep_z];
elseif Dims.has_semiz
    Grids.bothz_gridvals_J = vfoptions.semiz_gridvals_J;
else
    Grids.bothz_gridvals_J = z_gridvals_J;
end

if Dims.exp_flag > 0
    Grids.a1_gridvals = gpuArray(vfoptions.a1_grid(1:Dims.N_a1));
    Grids.a2_gridvals = CreateGridvals(Dims.n_a2, gpuArray(vfoptions.a2_grid), 1);
    Grids.d2_gridvals = CreateGridvals(Dims.n_d2, gpuArray(vfoptions.d2_grid), 1);
    if Dims.exp_flag == 6
        Grids.u_gridvals = CreateGridvals(vfoptions.n_u, gpuArray(vfoptions.u_grid), 1);
    end

    l_d2_len = length(Dims.n_d2); if sum(Dims.n_d2) == 0; l_d2_len = 0; end
    l_a2_len = length(Dims.n_a2); if sum(Dims.n_a2) == 0; l_a2_len = 0; end
    l_z_len  = length(n_z);       if sum(n_z) == 0;       l_z_len = 0;  end
    l_e_len  = length(Dims.n_e);  if sum(Dims.n_e) == 0;  l_e_len = 0;  end
    l_sz_len = length(Dims.n_semiz); if sum(Dims.n_semiz) == 0; l_sz_len = 0; end
    if isfield(vfoptions, 'n_u') && sum(vfoptions.n_u) > 0; l_u_len = length(vfoptions.n_u); else; l_u_len = 0; end

    if Dims.exp_flag == 3 || Dims.exp_flag == 4; num_state_args = l_d2_len + l_a2_len + l_z_len + l_e_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 2; num_state_args = l_d2_len + l_a2_len + l_z_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 5; num_state_args = l_d2_len + l_a2_len + l_sz_len + (l_a2_len >= 2);
    elseif Dims.exp_flag == 6; num_state_args = l_d2_len + l_a2_len + l_u_len + (l_a2_len >= 2);
    else; num_state_args = l_d2_len + l_a2_len + (l_a2_len >= 2); end

    temp_args = getAnonymousFnInputNames(vfoptions.aprimeFn);
    if length(temp_args) > num_state_args
        Dims.aprimeFnParamNames = temp_args(num_state_args + 1 : end);
    else
        Dims.aprimeFnParamNames = {};
    end

    if Dims.exp_flag == 3 || Dims.exp_flag == 4
        Dims.N_e_interp = Dims.N_e;
    else
        Dims.N_e_interp = 1;
    end
end

% --- 8. Divide and Conquer ---
if isfield(vfoptions, 'level1n') && ~isempty(vfoptions.level1n)
    Dims.level1n = vfoptions.level1n(1);
else
    Dims.level1n = min(floor(sqrt(Dims.n_a1(1))), Dims.n_a1(1));
end

end
