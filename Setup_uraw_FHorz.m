function [Dims, Grids, Probs] = Setup_uraw_FHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions)

% Helper to robustly expand stationary grids/probs across time
    function out = expand_time(in, target_dim, N)
        if size(in, target_dim) == 1 && N > 1
            rep_dims = ones(1, target_dim);
            rep_dims(target_dim) = N;
            out = repmat(in, rep_dims);
        else
            out = in;
        end
    end

% --- 1. Unpack Exotic Dimensions from vfoptions ---
if prod(vfoptions.n_e) > 0
    n_e = vfoptions.n_e;
    e_gridvals_J = expand_time(vfoptions.e_gridvals_J, 3, N_j);
    pi_e_J = expand_time(shiftdim(vfoptions.pi_e_J, -2), 4, N_j+1);
else
    n_e = 0; e_gridvals_J = []; pi_e_J = [];
end
Dims.n_e=n_e;

if prod(vfoptions.n_semiz) > 0
    n_semiz = vfoptions.n_semiz;
    semiz_gridvals_J = expand_time(vfoptions.semiz_gridvals_J, 3, N_j);
    pi_semiz_J = expand_time(vfoptions.pi_semiz_J, 4, N_j);

    % Split n_d into n_d1 (standard) and n_d2 (drives semiz transition)
    if length(n_d) > vfoptions.l_dsemiz
        n_d1 = n_d(1:end-vfoptions.l_dsemiz);
        n_d2 = n_d(end-vfoptions.l_dsemiz+1:end);
    else
        n_d1 = 0; n_d2 = n_d;
    end
else
    n_semiz = 0; semiz_gridvals_J = []; pi_semiz_J = [];
    n_d1 = n_d; n_d2 = 0;
end

% --- 2. Build Dimensions (Dims) ---
if n_d1==0; Dims.n_d_comb = n_d2; elseif n_d2==0; Dims.n_d_comb = n_d1; else; Dims.n_d_comb = [n_d1, n_d2]; end
if n_semiz==0; Dims.n_bothz = n_z; elseif n_z==0; Dims.n_bothz = n_semiz; else; Dims.n_bothz = [n_semiz, n_z]; end

N_d1_raw = prod(n_d1); N_d2_raw = prod(n_d2);
Dims.N_a = prod(n_a);
N_z_raw = prod(n_z); N_semiz_raw = prod(n_semiz); N_e_raw = prod(n_e);

Dims.has_d1 = (N_d1_raw > 0); Dims.l_d1 = Dims.has_d1 * length(n_d1); Dims.N_d1 = max(N_d1_raw, 1);
Dims.has_d2 = (N_d2_raw > 0); Dims.l_d2 = Dims.has_d2 * length(n_d2); Dims.N_d2 = max(N_d2_raw, 1);
Dims.N_d = Dims.N_d1 * Dims.N_d2;

Dims.N_z = max(N_z_raw, 1);
Dims.N_semiz_raw = N_semiz_raw; % Needed to trigger Policy array sizing
Dims.N_semiz = max(N_semiz_raw, 1);
Dims.N_bothz = Dims.N_semiz * Dims.N_z;
Dims.N_e_raw = N_e_raw;
Dims.N_e = max(N_e_raw, 1);

% Memory dimensions
if vfoptions.lowmemory > 0
    if N_z_raw == 0; Dims.special_n_z = 0; else; Dims.special_n_z = ones(1, length(n_z)); end
    if N_e_raw == 0; Dims.special_n_e = 0; else; Dims.special_n_e = ones(1, length(n_e)); end
end
if vfoptions.lowmemory == 3
    Dims.special_n_bothz = ones(1, length(n_semiz) + length(n_z));
end

% --- 3. Build Grids & Probs ---
if N_z_raw == 0
    Probs.pi_z_J = ones(1, 1, N_j); Grids.z_gridvals_J = zeros(1, 1, N_j);
else
    Probs.pi_z_J = expand_time(pi_z_J, 3, N_j);
    Grids.z_gridvals_J = expand_time(z_gridvals_J, 3, N_j);
end

if N_semiz_raw == 0
    Probs.pi_semiz_J = ones(1, 1, Dims.N_d2, N_j); Grids.semiz_gridvals_J = zeros(1, 1, N_j);
else
    Probs.pi_semiz_J = pi_semiz_J; Grids.semiz_gridvals_J = semiz_gridvals_J;
end

if N_e_raw == 0
    Probs.pi_e_J = ones(1, 1, 1, N_j+1); Grids.e_gridvals_J = zeros(1, 1, N_j);
else
    Probs.pi_e_J = pi_e_J; Grids.e_gridvals_J = e_gridvals_J;
end

% Combined Grids
if N_semiz_raw == 0 && N_z_raw == 0
    Grids.bothz_gridvals_J = zeros(1, 1, N_j);
elseif N_semiz_raw == 0
    Grids.bothz_gridvals_J = Grids.z_gridvals_J;
elseif N_z_raw == 0
    Grids.bothz_gridvals_J = Grids.semiz_gridvals_J;
else
    Grids.bothz_gridvals_J = [repmat(Grids.semiz_gridvals_J, Dims.N_z, 1, 1), repelem(Grids.z_gridvals_J, Dims.N_semiz, 1, 1)];
end

if Dims.has_d2 && Dims.has_d1
    Grids.d12_gridvals = permute(reshape(d_gridvals, [Dims.N_d1, Dims.N_d2, Dims.l_d1 + Dims.l_d2]), [1, 3, 2]);
    Dims.special_n_d = [n_d1, ones(1, Dims.l_d2)];
else
    Grids.d12_gridvals = d_gridvals;
    Dims.special_n_d = Dims.n_d_comb;
end


end