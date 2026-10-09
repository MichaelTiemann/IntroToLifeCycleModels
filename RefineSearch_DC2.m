function [Vtempii, maxindexfix, dind] = RefineSearch_DC2(ReturnFnHandle, loweredge1, loweredge2, maxgap1, maxgap2, N_d, N_a1, EV_RHS_slice)

% 1. Create the local grid offsets
a1_offsets = 0:1:maxgap1;
a2_offsets = 0:1:maxgap2;

% 2. Broadcast and expand (Works safely even if maxgaps are 0)
a1primeindexes = loweredge1 + repmat(a1_offsets, 1, maxgap2 + 1);
a2primeindexes = loweredge2 + repelem(a2_offsets, 1, maxgap1 + 1);

% 3. Evaluate the return function
ReturnMatrix = ReturnFnHandle(a1primeindexes, a2primeindexes);

% 4. Add expected value (if provided by caller)
if nargin > 7 && ~isempty(EV_RHS_slice)
    entireRHS = ReturnMatrix + EV_RHS_slice;
else
    entireRHS = ReturnMatrix;
end

% 5. Maximize (Combines d, a1prime, and a2prime)
[Vtempii, maxindex] = max(entireRHS, [], 1);

% 6. Extract relative indices
dind = rem(maxindex-1, N_d) + 1;
a1primeind = rem(ceil(maxindex/N_d)-1, maxgap1 + 1);
a2primeind = ceil(maxindex/(N_d*(maxgap1 + 1))) - 1;

% 7. Compute the fixed maxindex using the global N_a1 stride
maxindexfix = dind + N_d * a1primeind + N_d * N_a1 * a2primeind;


end
