function [Vtempii, maxindex, dind] = RefineSearch_DC1(ReturnFnHandle, loweredge, maxgap, N_d, reshape_size, EV_RHS_slice)

% 1. Evaluate the narrow choice band
a1primeindexes = loweredge + (0:1:maxgap);
ReturnMatrix = ReturnFnHandle(a1primeindexes);

% 2. Apply reshape if provided by caller
if nargin > 4 && ~isempty(reshape_size)
    ReturnMatrix = reshape(ReturnMatrix, reshape_size);
end

% 3. Add expected value (if not in a terminal period)
if nargin > 5 && ~isempty(EV_RHS_slice)
    entireRHS = ReturnMatrix + EV_RHS_slice;
else
    entireRHS = ReturnMatrix;
end

% 4. Maximize
[Vtempii, maxindex] = max(entireRHS, [], 1);

% 5. Extract dind for the allind offset calculation
dind = rem(maxindex-1, N_d) + 1;


end