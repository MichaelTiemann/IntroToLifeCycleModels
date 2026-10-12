function [Dims, Grids, Probs] = Setup_UFHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions)

% --- 1. Endogenous States (a) ---
Dims.N_a = max(prod(n_a), 1);

% --- 2. Decisions (d) ---
Dims.N_d = max(prod(n_d), 1);

% --- 3. Exogenous Shocks (z, semiz, e) ---
Dims.n_z = n_z;
Dims.n_semiz = vfoptions.n_semiz;
Dims.n_e = vfoptions.n_e;

Dims.N_z = max(prod(n_z), 1);
Dims.N_semiz = max(prod(vfoptions.n_semiz), 1);
Dims.N_e = max(prod(vfoptions.n_e), 1);

Dims.N_semiz_raw = prod(Dims.n_semiz);
Dims.N_e_raw     = prod(Dims.n_e);

% CRITICAL: The zeros are structurally load-bearing. We never drop them.
Dims.n_bothz = [Dims.n_semiz, Dims.n_z];
Dims.N_bothz = Dims.N_semiz * Dims.N_z;

% --- 4. Experience Asset Flags ---
Dims.exp_flag = 0;
if vfoptions.experienceasset>=1; Dims.exp_flag=1; end
if vfoptions.experienceassetz>=1; Dims.exp_flag=2; end
if vfoptions.experienceassete>=1; Dims.exp_flag=3; end
if vfoptions.experienceassetze>=1; Dims.exp_flag=4; end
if vfoptions.experienceassetsemiz>=1; Dims.exp_flag=5; end
if vfoptions.experienceassetu>=1; Dims.exp_flag=6; end

% --- 5. Unpack Splits for Endogenous States and Decisions ---
if Dims.exp_flag > 0
    % n_a1 and n_a2 are guaranteed to be populated by SetupNonStandardEndoStates
    Dims.n_a1 = vfoptions.n_a1;
    Dims.n_a2 = vfoptions.n_a2;
    Dims.N_a1 = max(prod(Dims.n_a1), 1);
    Dims.N_a2 = max(prod(Dims.n_a2), 1);
else
    Dims.n_a1 = n_a; Dims.n_a2 = 1;
    Dims.N_a1 = max(prod(n_a), 1); Dims.N_a2 = 1;
end

if isfield(vfoptions, 'n_d1') && isfield(vfoptions, 'n_d2')
    Dims.n_d1 = vfoptions.n_d1;
    Dims.n_d2 = vfoptions.n_d2;
else
    Dims.n_d1 = n_d;
    Dims.n_d2 = 0;
end
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
Dims.has_d1 = prod(Dims.n_d1) > 0;
Dims.has_d2 = prod(Dims.n_d2) > 0;

% Fallbacks for CreateReturnFnMatrix inputs
Dims.special_n_d = n_d;
if isempty(Dims.special_n_d); Dims.special_n_d = 0; end

Dims.special_n_z = n_z;
if isempty(Dims.special_n_z); Dims.special_n_z = 0; end

Dims.special_n_semiz = Dims.n_semiz;
if isempty(Dims.special_n_semiz); Dims.special_n_semiz = 0; end

Dims.special_n_e = Dims.n_e;
if isempty(Dims.special_n_e); Dims.special_n_e = 0; end

Dims.special_n_bothz = Dims.n_bothz;
if isempty(Dims.special_n_bothz); Dims.special_n_bothz = 0; end

% --- 6. Grid and Prob Assignments ---
Grids.d12_gridvals = d_gridvals;

if Dims.N_semiz_raw > 0
    rep_semiz = repmat(vfoptions.semiz_gridvals_J, [Dims.N_z, 1, 1]);
    rep_z = repelem(z_gridvals_J, Dims.N_semiz, 1);
    Grids.bothz_gridvals_J = [rep_semiz, rep_z];
else
    Grids.bothz_gridvals_J = z_gridvals_J;
end

if Dims.N_e_raw > 0
    Grids.e_gridvals_J = vfoptions.e_gridvals_J;
    Probs.pi_e_J = vfoptions.pi_e_J;
else
    Grids.e_gridvals_J = [];
    Probs.pi_e_J = [];
end

Probs.pi_z_J = pi_z_J;
if Dims.N_semiz_raw > 0
    Probs.pi_semiz_J = vfoptions.pi_semiz_J;
else
    Probs.pi_semiz_J = [];
end


end
