function [z_grid, pi_z] = LifeCycleModel21_ExogShockFn(agej, Jr)

% 1. Boolean cohort masks along Dimension 3: size [1, 1, N_j]
is_work   = shiftdim(agej < Jr,  -2);
is_retire = shiftdim(agej >= Jr, -2);
is_trans  = shiftdim(agej == Jr, -2);
is_post   = shiftdim(agej > Jr,  -2);

% 2. Vectorized z_grid across all ages: size [2, 1, N_j]
% Working age: [1; 0], Retirement: [0.3; 0]
z_grid = [1; 0] .* is_work + [0.3; 0] .* is_retire;

% 3. Vectorized transition tensor across all ages: size [2, 2, N_j]
pi_work  = [0.7, 0.3; 0.5, 0.5];
pi_trans = [0.0, 1.0; 0.0, 1.0];
pi_post  = [0.2, 0.8; 0.3, 0.7];

pi_z = pi_work .* is_work + pi_trans .* is_trans + pi_post .* is_post;

end