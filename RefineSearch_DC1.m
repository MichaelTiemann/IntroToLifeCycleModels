function [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, loweredge, maxgap, N_d, reshape_size, EV_RHS_slice)

% 1. Evaluate the narrow choice band
% Explicitly use transpose (') to create a column vector for implicit expansion
a1primeindexes = loweredge + (0:1:maxgap)';

% Expand the grid to evaluate all decision combinations
if N_d > 1
    a1primeindexes = kron(a1primeindexes, ones(N_d, 1));
end

ReturnMatrix = ReturnFnHandle(a1primeindexes);

% 2. Add expected value (if not in a terminal period)
if nargin > 5 && ~isempty(EV_RHS_slice)
    % The Expected Value must also be expanded to match the decision space
    if N_d > 1
        EV_RHS_slice = kron(EV_RHS_slice, ones(N_d, 1));
    end
    entireRHS = ReturnMatrix + EV_RHS_slice;
else
    entireRHS = ReturnMatrix;
end

% 3. Maximize
[Vtempii, maxindex] = max(entireRHS, [], 1);

% 4. Extract dind for the allind offset calculation
if N_d > 1
    dind = rem(maxindex - 1, N_d) + 1;
    maxindex = ceil(maxindex / N_d);
else
    dind = 1;
end


end
