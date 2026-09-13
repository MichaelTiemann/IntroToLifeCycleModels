function W = LifeCycleModelV33_WarmGlowBequestsFn(aprime, wg, sigma, agej, J)

% Natively vectorizes across any aprime tensor size using logical masking

if agej == J
    % Preallocate the entire tensor with strict -Inf
    W = -Inf(size(aprime), 'like', aprime);
    
    % Locate all nodes with strictly positive bequests
    valid = (aprime > 0);
    
    % Calculate warm glow utility only on valid nodes to prevent NaNs/complex numbers
    W(valid) = wg .* (aprime(valid).^(1 - sigma)) ./ (1 - sigma);
else
    % For all periods before J, bequest utility is strictly zero
    W = zeros(size(aprime), 'like', aprime);
end


end