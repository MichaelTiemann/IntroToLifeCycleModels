function F = LifeCycleModel1_VReturnFn(aprime, a, w, r, sigma, beta)

% 1. Calculate Consumption using element-wise operations (.*, .+, ./)
% Both 'a' and 'aprime' will be large 1D column vectors.
c = w + (1 + r) * a - aprime;

% 2. Prepare logical masking for valid consumption
valid_c = c > 0;

% Safe c to prevent complex numbers when evaluating c^(1-sigma) for negative c
c_safe = c;
c_safe(~valid_c) = 1; 

% 3. Vectorized Utility Calculation
utility = (c_safe.^(1 - sigma)) ./ (1 - sigma);

% 4. Initialize F and assign valid utilities
% Use 'like' to maintain gpuArray typing if the inputs are on the GPU
F = -Inf(size(c), 'like', c);
F(valid_c) = utility(valid_c);


end
