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

profile clear
profile on

fprintf("new ValueFnIter_Case1_VFHorz\n");
tic;
% Call the shadowed function using the saved handle
[V_new, Policy_new] = ValueFnIter_Case1_VFHorz(varargin{:});
time_new=toc;

profile off
profile viewer

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
tol = 1e-10;
if isfield(vfoptions, 'precision') && strcmp(vfoptions.precision, 'single')
    tol = 1e-5;
end

rel_diff = max(abs(V_new(:) - V_ref(:)) ./ (abs(V_ref(:)) + 1));
if rel_diff > tol
    % Is this a floating-point rounding difference or a massive math failure?
    max_diff = max(abs(V_new(:) - V_ref(:)));
    disp(['Maximum absolute difference: ', num2str(max_diff)]);
    fprintf('Legacy NaNs: %d | Legacy -Infs: %d\n', sum(isnan(V_ref(:))), sum(V_ref(:) == -Inf));
    fprintf('Tensor NaNs: %d | Tensor -Infs: %d\n', sum(isnan(V_new(:))), sum(V_new(:) == -Inf));
    % Find the exact coordinate of the biggest difference
    [~, bad_idx] = max(abs(V_new(:) - V_ref(:)));
    [bad_a, bad_z, bad_j] = ind2sub(size(V_new), bad_idx);
    fprintf('Worst mismatch at -> Asset idx: %d, Z idx: %d, Age j: %d\n', bad_a, bad_z, bad_j);

    % Show the actual values side-by-side
    fprintf('Legacy V: %f\n', V_ref(bad_a, bad_z, bad_j));
    fprintf('Tensor V: %f\n', V_new(bad_a, bad_z, bad_j));
    error("V_new ~= V_ref");
end

fprintf('time reference: %.2f seconds; time difference: %.2f seconds; time ratio to ref: %.0f%%\n', time_ref, time_new-time_ref, 100*time_new/time_ref);

end
