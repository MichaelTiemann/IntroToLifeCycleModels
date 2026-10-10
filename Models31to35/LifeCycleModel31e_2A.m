%% Life-Cycle Model 31_2A: Portfolio-Choice with 3 Assets
% Safe Asset (r_safe)
% Future Fund (r_future, but penalized if withdrawn before retirement)
% Risky Asset (stochastic return u)

%% Setup
Params.agejshifter = 19;
Params.J = 100 - Params.agejshifter;

% Grid sizes to use (Keep grids slightly smaller due to 3D state space)
n_d = [21];               % Risky investment choice
n_a = [31, 21, 21];       % [Safe Asset, Future Fund, Risky Asset]
n_z = 11;
n_e = 3;
n_u = 5;
N_j = Params.J;

vfoptions.riskyasset = 1;
simoptions.riskyasset = 1;
vfoptions.refine_d = [0, 0, 1];
vfoptions.lowmemory=2;
simoptions.refine_d = vfoptions.refine_d;

%% Parameters
Params.beta = 0.9;
Params.sigma = 10;
Params.w = 1;

% Asset returns
Params.r_safe = 0.02;     % Low return, fully liquid
Params.r_future = 0.05;   % Higher return, penalized if drawn early
Params.rp = 0.04;         % Risky premium
Params.sigma_u = 0.025;
Params.rho_u = 0;

[u_grid, pi_u] = discretizeAR1_FarmerToda(Params.rp, Params.rho_u, Params.sigma_u, n_u);
pi_u = pi_u(1,:)';

Params.agej = 1:1:Params.J;
Params.Jr = 46;
Params.pension = 0.3;
Params.kappa_j = [linspace(0.5, 2, Params.Jr-15), linspace(2, 1, 14), zeros(1, Params.J-Params.Jr+1)];
Params.rho_z = 0.9;
Params.sigma_epsilon_z = 0.03;
Params.sigma_epsilon_e = 0.2;

Params.dj = [0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000];
Params.sj = 1 - Params.dj(21:101);
Params.sj(end) = 0;

%% Grids
a_safe_grid   = 40 * (linspace(0, 1, n_a(1)).^3)';
a_future_grid = 40 * (linspace(0, 1, n_a(2)).^3)';
m_grid        = 40 * (linspace(0, 1, n_a(3)).^3)';

% Pack the asset grids in exact order: [safe, future, risky]
a_grid = [a_safe_grid; a_future_grid; m_grid];

% Decision grid: absolute amount to invest in risky asset
d_m_grid = 40 * (linspace(0, 1, n_d(1)).^3)';
d_grid = d_m_grid;

[z_grid, pi_z] = discretizeAR1_FarmerToda(0, Params.rho_z, Params.sigma_epsilon_z, n_z);
z_grid = exp(z_grid);
[mean_z, ~, ~, ~] = MarkovChainMoments(z_grid, pi_z);
z_grid = z_grid ./ mean_z;

% Now the iid normal process e
[e_grid,pi_e] = discretizeAR1_FarmerToda(0,0,Params.sigma_epsilon_e,n_e);
e_grid = exp(e_grid);
pi_e = pi_e(1,:)'; 
mean_e = pi_e'*e_grid;
e_grid = e_grid./mean_e; 

vfoptions.n_e = n_e;
vfoptions.e_grid = e_grid;
vfoptions.pi_e = pi_e;
simoptions.n_e = n_e;
simoptions.e_grid = e_grid;
simoptions.pi_e = pi_e;

%% Risky Asset Transition
aprimeFn = @(d_m, u) d_m .* u;

vfoptions.aprimeFn = aprimeFn;
vfoptions.n_u = n_u;
vfoptions.u_grid = u_grid;
vfoptions.pi_u = pi_u;
simoptions.aprimeFn = aprimeFn;
simoptions.n_u = n_u;
simoptions.u_grid = u_grid;
simoptions.pi_u = pi_u;
simoptions.a_grid = a_grid;
simoptions.d_grid = d_grid;

%% Return Function setup
DiscountFactorParamNames = {'beta', 'sj'};
ReturnFn = @(d_m, aprime_safe, aprime_future, a_safe, a_future, m, z, e, w, r_safe, r_future, sigma, agej, Jr, pension, kappa_j) ...
    LifeCycleModel31e_2A_ReturnFn(d_m, aprime_safe, aprime_future, a_safe, a_future, m, z, e, w, r_safe, r_future, sigma, agej, Jr, pension, kappa_j);

vfoptions.divideandconquer = 0;
vfoptions.gridinterplayer = 1;
vfoptions.ngridinterp = 10;
simoptions.gridinterplayer = vfoptions.gridinterplayer;
simoptions.ngridinterp = vfoptions.ngridinterp;

%% Solve
disp('Solve for Value fn and Policy fn using ValueFnIter command')
tic;
[V, Policy] = ValueFnIter_Case1_FHorz(n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Reshape and Plot V (Holding Future and Risky constant at 0)
zind = floor((n_z+1)/2);
eind = floor((n_e+1)/2); % Median e shock
future_ind = 1; % a_future = 0
m_ind = 1;      % m = 0

figure(1)
surf(1:Params.J, a_safe_grid, squeeze(V(:, future_ind, m_ind, zind, eind, :)))
title('Value function: Safe Wealth (median z, a_{future}=0, m=0)')
xlabel('Age j')
ylabel('Safe Assets (a_{safe})')
zlabel('Value')

%% Simulation
jequaloneDist = zeros([n_a, n_z, n_e], 'gpuArray');
jequaloneDist(1, 1, 1, floor((n_z+1)/2), floor((n_e+1)/2)) = 1; % Start with 0 in all assets

Params.mewj = ones(1, Params.J);
for jj=2:length(Params.mewj)
    Params.mewj(jj) = Params.sj(jj-1) * Params.mewj(jj-1);
end
Params.mewj = Params.mewj ./ sum(Params.mewj);
AgeWeightsParamNames = {'mewj'};

StationaryDist = StationaryDist_FHorz_Case1(jequaloneDist, AgeWeightsParamNames, Policy, n_d, n_a, n_z, N_j, pi_z, Params, simoptions);

% Grid Checks across all 3 dimensions
fprintf('Safe grid top mass: %1.2e\n', gather(sum(StationaryDist(end, :, :, :, :), 'all')));
fprintf('Future grid top mass: %1.2e\n', gather(sum(StationaryDist(:, end, :, :, :), 'all')));
fprintf('Risky grid top mass: %1.2e\n', gather(sum(StationaryDist(:, :, end, :, :), 'all')));

%% Life-Cycle Profiles
FnsToEvaluate.safe_share = @(d_m, ap1, ap2, a1, a2, m, z, e) ap1 ./ max(1e-10, d_m + ap1 + ap2);
FnsToEvaluate.future_share = @(d_m, ap1, ap2, a1, a2, m, z, e) ap2 ./ max(1e-10, d_m + ap1 + ap2);
FnsToEvaluate.risky_share = @(d_m, ap1, ap2, a1, a2, m, z, e) d_m ./ max(1e-10, d_m + ap1 + ap2);
FnsToEvaluate.total_wealth = @(d_m, ap1, ap2, a1, a2, m, z, e) a1 + a2 + m;

AgeConditionalStats = LifeCycleProfiles_FHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Params, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions);

figure(2)
plot(1:Params.J, AgeConditionalStats.safe_share.Mean, ...
    1:Params.J, AgeConditionalStats.future_share.Mean, ...
    1:Params.J, AgeConditionalStats.risky_share.Mean, 'LineWidth', 2)
title('Life Cycle Portfolio Allocation')
legend('Safe Share', 'Future Fund Share', 'Risky Share')
xlabel('Age')

%% Extract Polices for Scatters
% PolicyVals outputs [d_m, aprime_safe, aprime_future]
PolicyVals = PolicyInd2Val_FHorz(Policy, n_d, n_a, n_z, N_j, d_grid, a_grid, vfoptions);
Pol_reshaped = reshape(PolicyVals, [3, n_a(1), n_a(2), n_a(3), n_z, n_e, Params.J]);

target_age = 45;
d_m_vals    = reshape(Pol_reshaped(1, :, :, :, zind, eind, target_age), [n_a(1), n_a(2), n_a(3)]);
ap_safe     = reshape(Pol_reshaped(2, :, :, :, zind, eind, target_age), [n_a(1), n_a(2), n_a(3)]);
ap_future   = reshape(Pol_reshaped(3, :, :, :, zind, eind, target_age), [n_a(1), n_a(2), n_a(3)]);

total_savings = max(1e-10, d_m_vals + ap_safe + ap_future);
risky_ratio = d_m_vals ./ total_savings;

[A_safe_mesh, A_future_mesh, M_mesh] = ndgrid(a_safe_grid, a_future_grid, m_grid);
total_wealth_grid = A_safe_mesh + A_future_mesh + M_mesh;

figure(3)
scatter(total_wealth_grid(:), risky_ratio(:), 10, 'filled')
title(sprintf('Riskyshare vs Total Wealth at Age %i (Median z)', target_age + Params.agejshifter))
xlabel('Total Current Wealth (a_{safe} + a_{future} + m)')
ylabel('Riskyshare Ratio')
grid on;