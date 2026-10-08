function [Vtempii, maxindexfix, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap, N_d, N_a1, reshape_size, EV_RHS_slice)

% 1. Evaluate the narrow choice band
a1primeindexes = loweredge + (0:1:maxgap);
ReturnMatrix = ReturnFnHandle(a1primeindexes);

% 2. Apply the dynamic reshape shape passed by the caller
ReturnMatrix = reshape(ReturnMatrix, reshape_size);

% 3. Add expected value (if not in a terminal period)
if nargin > 6 && ~isempty(EV_RHS_slice)
    entireRHS = ReturnMatrix + EV_RHS_slice;
else
    entireRHS = ReturnMatrix;
end

% 4. Maximize
[Vtempii, maxindex] = max(entireRHS, [], 1);

% 5. Reconstruct the global explicit indices (natively handles maxgap==0)
dind = rem(maxindex-1, N_d) + 1;
a1primeind = rem(ceil(maxindex/N_d)-1, maxgap+1);
a2primeind = ceil(maxindex/(N_d*(maxgap+1))) - 1;

maxindexfix = dind + N_d*a1primeind + N_d*N_a1*a2primeind;


end