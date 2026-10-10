function [a2primeIndexes, a2primeProbs] = CreateRiskyAssetFnMatrix(aprimeFn, n_d, n_a2, n_u, d_grid, a2_grid, u_gridvals, aprimeFnParams, aprimeIndexAsColumn)
% Note: a2primeIndex is [N_a2*N_u,1], whereas a2primeProbs is [N_a2,N_u]
%
% Creates the grid points and their 'interpolation' probabilities
% Note: a2primeIndexes is always the 'lower' point (the upper points are
% just a2primeIndexes+1, so no need to waste memory storing them), and the
% a2primeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).

ParamCell = cell(length(aprimeFnParams), 1);
for ii = 1:length(aprimeFnParams)
    if ~isscalar(aprimeFnParams(ii))
        error('Using riskyasset does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell{ii, 1} = aprimeFnParams(ii);
end

% Safe Dimension Sizes (Handling empty grids gracefully)
N_d = max(1, prod(n_d(n_d > 0)));
N_a2 = max(1, prod(n_a2(n_a2 > 0)));
N_u = max(1, prod(n_u(n_u > 0)));
l_a2 = length(n_a2);

% Dynamically build the arguments list for arrayfun
args = {};

% A. Active 'd' variables (shift sequentially)
l_d_active = 0;
grid_idx = 1;
for i = 1:length(n_d)
    if n_d(i) > 0
        d_vals = d_grid(grid_idx : grid_idx + n_d(i) - 1);
        args{end+1} = shiftdim(d_vals, -l_d_active);
        l_d_active = l_d_active + 1;
        grid_idx = grid_idx + n_d(i);
    end
end

% B. Active 'u' variables (shift to dimensions following 'd')
l_u_active = 0;
for i = 1:length(n_u)
    if n_u(i) > 0
        args{end+1} = shiftdim(u_gridvals(:, i), -(l_d_active + l_u_active));
        l_u_active = l_u_active + 1;
    end
end

% Verify correct number of inputs to the anonymous function
if nargin(aprimeFn) ~= l_d_active + l_u_active + length(aprimeFnParams)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

% Evaluate across the entire state space dynamically
a2primeVals = arrayfun(aprimeFn, args{:}, ParamCell{:});


%% Calcuate grid indexes and probs from the values
if l_a2 == 1
    a2primeVals = reshape(a2primeVals, [1, N_d * N_u]);
    a2_griddiff = a2_grid(2:end) - a2_grid(1:end-1); % Distance between point and the next point

    % For small aprimeVals and a_grid, max() is faster than discretize()
    % http://discourse.vfitoolkit.com/t/example-attanasio-low-sanchez-marcos-2008/257/25
    if N_d * N_u * N_a2 < 1000000
        [~, a2primeIndexes] = max((a2_grid > a2primeVals), [], 1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
        % Note, this is going to find the 'first' grid point which is bigger than aprimeVals
        % This is the 'upper' grid point
        % Have to have special treatment for trying to leave the ends of the grid (I fix these below)

        % Switch to lower grid point index
        a2primeIndexes = a2primeIndexes - 1;
        a2primeIndexes(a2primeIndexes == 0) = 1;

        % Now, find the probabilities
        aprime_residual = a2primeVals' - a2_grid(a2primeIndexes);
        % Probability of the 'lower' points
        a2primeProbs = 1 - aprime_residual ./ a2_griddiff(a2primeIndexes);

        % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
        offTopOfGrid = (a2primeVals >= a2_grid(end));
        a2primeIndexes(offTopOfGrid) = N_a2 - 1; % lower grid point is the one before the end point
        a2primeProbs(offTopOfGrid) = 0;

        % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
        offBottomOfGrid = (a2primeVals <= a2_grid(1));
        % aprimeIndexes(offBottomOfGrid)=1; % Has already been handled
        a2primeProbs(offBottomOfGrid) = 1;

    else
        a2primeIndexes = discretize(a2primeVals, a2_grid); % Finds the lower grid point

        % Have to have special treatment for trying to leave the ends of the grid
        % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
        offBottomOfGrid = (a2primeVals <= a2_grid(1));
        a2primeIndexes(offBottomOfGrid) = 1; % Has already been handled

        % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
        offTopOfGrid = (a2primeVals >= a2_grid(end));
        a2primeIndexes(offTopOfGrid) = N_a2 - 1; % lower grid point is the one before the end point

        % Now, find the probabilities
        aprime_residual = a2primeVals' - a2_grid(a2primeIndexes);
        % Probability of the 'lower' points
        a2primeProbs = 1 - aprime_residual ./ a2_griddiff(a2primeIndexes);

        % And clean up the ends of the grid
        a2primeProbs(offBottomOfGrid) = 1;
        a2primeProbs(offTopOfGrid) = 0;
    end

    if aprimeIndexAsColumn == 1
        % value fn codes want column, simulation codes want matrix
        %     aprimeIndexes=reshape(aprimeIndexes,[N_d*N_u,1]);
        a2primeIndexes = a2primeIndexes'; % This is just doing the commented out reshape above
    else
        a2primeIndexes = reshape(a2primeIndexes, [N_d, N_u]);
    end
    a2primeProbs = reshape(a2primeProbs, [N_d, N_u]);
end


end
