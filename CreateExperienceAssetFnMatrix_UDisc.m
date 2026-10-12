function [a2primeIndexes, a2primeProbs] = CreateExperienceAssetFnMatrix_UDisc(aprimeFn, n_d, n_a2, d_gridvals, a2_grid, aprimeFnParams, aprimeIndexAsColumn, varargin)
% Universal Experience Asset Matrix Builder for the UDisc (Unified Discrete) Architecture.
% Handles basic ExpAsset, plus any combination of exogenous states (z, e, semiz, u) dynamically.

% 1. Parameter Setup
ParamCell = num2cell(aprimeFnParams)';

N_d = max(prod(n_d), 1);
N_a2 = max(prod(n_a2), 1);
l_d = size(d_gridvals, 2);

GridParamsCell = {};

% Dim 1: Unified Decision Grid (d)
if N_d > 1 || l_d > 0
    for i = 1:l_d
        GridParamsCell{end+1} = d_gridvals(:, i);
    end
end

% Dim 2: Current Experience Asset (a2)
GridParamsCell{end+1} = shiftdim(a2_grid(:), -1);

% Dims 3+: Exogenous States (z, e, semiz, u) parsed from varargin
dim_offset = 2;
state_dims = [];
for v = 1:2:length(varargin)
    n_x = varargin{v};
    x_gridvals = varargin{v+1};
    N_x = max(prod(n_x), 1);
    state_dims = [state_dims, N_x];

    l_x = size(x_gridvals, 2);
    for i = 1:l_x
        GridParamsCell{end+1} = shiftdim(x_gridvals(:, i), -dim_offset);
    end
    dim_offset = dim_offset + 1;
end

if isempty(state_dims)
    state_dims = 1;
end

% 2. Evaluate the Function
% arrayfun natively broadcasts across the dynamically generated N-D cell array
a2primeVals = arrayfun(aprimeFn, GridParamsCell{:}, ParamCell{:});
a2primeVals = a2primeVals(:); % Flatten to column vector for vectorized interpolation

% 3. Interpolation & Bounding
total_elements = numel(a2primeVals);
a2_griddiff = a2_grid(2:end) - a2_grid(1:end-1);
a2_grid_col = a2_grid(:)'; % Row vector for fast broadcasting

if total_elements * length(a2_grid) < 10000000
    [~, a2primeIndexes] = max((a2_grid_col > a2primeVals), [], 2);
    a2primeIndexes = a2primeIndexes - 1;
    a2primeIndexes(a2primeIndexes == 0) = 1;
else
    a2primeIndexes = discretize(a2primeVals, a2_grid);
    a2primeIndexes(isnan(a2primeIndexes)) = 1; % Failsafe
end

% Handle Out-of-Bounds
offTopOfGrid = (a2primeVals >= a2_grid(end));
a2primeIndexes(offTopOfGrid) = length(a2_grid) - 1;

offBottomOfGrid = (a2primeVals <= a2_grid(1));
a2primeIndexes(offBottomOfGrid) = 1;

% Calculate Probabilities
aprime_residual = a2primeVals - a2_grid(a2primeIndexes);
a2primeProbs = 1 - aprime_residual ./ a2_griddiff(a2primeIndexes);

% Clean up boundary probabilities
a2primeProbs(offBottomOfGrid) = 1;
a2primeProbs(offTopOfGrid) = 0;

% 4. Reshape Output Based on Caller Request
if aprimeIndexAsColumn == 1
    % Returned flat as [N_d * N_a2 * states, 1]
elseif aprimeIndexAsColumn == 2
    out_sz = [N_d, N_a2, state_dims];
    if length(out_sz) == 2; out_sz = [out_sz, 1]; end
    a2primeIndexes = reshape(a2primeIndexes, out_sz);
    a2primeProbs = reshape(a2primeProbs, out_sz);
elseif aprimeIndexAsColumn == 3
    out_sz = [N_d * N_a2, state_dims];
    if length(out_sz) == 1; out_sz = [out_sz, 1]; end
    a2primeIndexes = reshape(a2primeIndexes, out_sz);
    a2primeProbs = reshape(a2primeProbs, out_sz);
end


end
