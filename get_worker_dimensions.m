function [n_a2_worker, n_z_worker] = get_worker_dimensions(n_a2, n_z, n_proc)

% Mirror the exact slicing heuristic used in pararrayfun_vfi
len_a2 = prod(n_a2);
len_z  = prod(n_z);

if n_proc > len_z && len_a2 > len_z
    % Slicing along a2
    workers = min(n_proc, len_a2);
    % Ceiling division gives peak chunk size across workers
    n_a2_worker = ceil(len_a2 / workers);
    n_z_worker  = len_z;
else
    % Slicing along z (or sequential n_proc = 1)
    workers = min(max(1, n_proc), len_z);
    n_a2_worker = len_a2;
    n_z_worker  = ceil(len_z / workers);
end


end
