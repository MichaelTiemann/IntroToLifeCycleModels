function [V_new, Policy_new] = ValueFnIter_Case1_FHorz(varargin)
persistent original_func

if isempty(original_func)
    % Get the name of this file (e.g., 'my_shadowing_function.m')
    this_file = [mfilename '.m'];

    % Find all instances on the MATLAB path
    all_paths = which(this_file, '-all');

    % The shadowing function is always first (all_paths{1}),
    % so the shadowed function is the second item (all_paths{2})
    shadowed_file_path = all_paths{2};

    % Extract the directory containing the shadowed function
    shadowed_dir = fileparts(shadowed_file_path);

    % Temporarily change directories to get a clean handle to it
    current_dir = cd(shadowed_dir);
    original_func = str2func(mfilename);
    cd(current_dir); % Go back safely
end

% Your custom wrapper code goes here...
fprintf("reference ValueFnIter_Case1_FHorz\n");
tic;
% Call the shadowed function using the saved handle
[V_ref, Policy_ref] = original_func(varargin{:});
time_ref=toc;

% profile clear
% profile on

fprintf("new ValueFnIter_Case1_VFHorz\n");
tic;
% Call the shadowed function using the saved handle
[V_new, Policy_new] = ValueFnIter_Case1_VFHorz(varargin{:});
time_new=toc;

% profile off
% profile viewer

% --- AUTOMATED POLICY MISMATCH MASKING ---
% Identify all valid states where the agent can actually survive (V_ref > -Inf)
valid_states = (V_ref > -Inf);

% Check if policies match ONLY on valid, reachable states
if any(Policy_new(:, valid_states) ~= Policy_ref(:, valid_states))
    sum(Policy_new(:) ~= Policy_ref(:))
    idx = find(Policy_new(:) ~= Policy_ref(:), 1); [r, c, p, q, t] = ind2sub(size(Policy_new), idx); disp(['Mismatch at [', num2str([r,c,p,q,t]), ']']);
    error('Policy_new ~= Policy_ref on reachable states');
elseif any(Policy_new(:) ~= Policy_ref(:))
    disp('Policy check passed on all reachable states! (Ignored dead -Inf states)');
end

vfoptions=varargin{end};
tol = 1e-3; % Could need to loosen as necessary

% Mask out non-finite states in the reference solution
valid_mask = isfinite(V_ref(:));

% Option A: Mean absolute level of finite states (The Scale Benchmark)
ref_scale = mean(abs(V_ref(valid_mask)));

% Option B: Dynamic Range Benchmark (invariant to constant shifts in utility)
% ref_scale = max(V_ref(valid_mask)) - min(V_ref(valid_mask));

% Safe relative difference across all reachable states
diff_vec = abs(V_new(valid_mask) - V_ref(valid_mask));
rel_diff = max(diff_vec) / max(ref_scale, eps);

if rel_diff > tol
    % Is this a floating-point rounding difference or a massive math failure?
    max_diff = max(abs(V_new(:) - V_ref(:)));
    disp(['Maximum absolute difference: ', num2str(max_diff)]);
    fprintf('Legacy NaNs: %d | Legacy -Infs: %d\n', sum(isnan(V_ref(:))), sum(V_ref(:) == -Inf));
    fprintf('Tensor NaNs: %d | Tensor -Infs: %d\n', sum(isnan(V_new(:))), sum(V_new(:) == -Inf));

    % Find the exact linear coordinate of the biggest difference
    [~, bad_idx] = max(abs(V_new(:) - V_ref(:)));

    % Dynamically resolve the N-dimensional coordinates using a cell array
    ndim_V = ndims(V_new);
    coords = cell(1, ndim_V);
    [coords{:}] = ind2sub(size(V_new), bad_idx);

    % Format the coordinates for display
    coord_str = sprintf('%d, ', cell2mat(coords));
    coord_str = coord_str(1:end-2); % Remove trailing comma and space

    fprintf('Worst mismatch at -> ND-Coords: [%s] (Linear Idx: %d)\n', coord_str, bad_idx);

    % Show the actual values and signed difference (New - Ref)
    val_ref = V_ref(bad_idx);
    val_new = V_new(bad_idx);
    signed_diff = val_new - val_ref;

    fprintf('Legacy V: %f\n', val_ref);
    fprintf('Tensor V: %f\n', val_new);
    fprintf('Signed Difference (Tensor - Legacy): %f\n', signed_diff);

    if signed_diff > 0
        disp('>>> TENSOR FOUND A HIGHER VALUE (Better optimization peak) <<<');
    else
        disp('>>> LEGACY FOUND A HIGHER VALUE (Tensor missed the peak) <<<');
    end

    error("V_new ~= V_ref");
end

fprintf('time reference: %.2f seconds; time difference: %.2f seconds; time ratio to ref: %.0f%%\n', time_ref, time_new-time_ref, 100*time_new/time_ref);

end
