function [V, Policy] = ValueFnIter_UFHorz_ExpAsset_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
    
    % --- 1. Level1n Default Expansion ---
    if ~isfield(vfoptions,'level1n')
        if length(n_a) == 2
            vfoptions.level1n = floor(sqrt(n_a(1)));
            if n_a(1) < 5; error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable'); end
        elseif length(n_a) >= 3
            vfoptions.level1n = [floor(sqrt(n_a(1))), n_a(2)]; % default DC2A
            if n_a(1) < 5; error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable'); end
        end
        if vfoptions.verbose == 1
            fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n \n')
        end
    else
        if ~isscalar(n_a) && isscalar(vfoptions.level1n)
            % Safely expand level1n for standard endo states, explicitly dropping the final exp asset
            vfoptions.level1n = [vfoptions.level1n, n_a(2:end-1)]; 
        end
    end

    % --- 2. Raw Engine Dispatch ---
    if length(n_a) == 2
        [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_DC1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    elseif length(n_a) >= 3
        if length(vfoptions.level1n) > 1
            if vfoptions.level1n(2) >= n_a(2)
                vfoptions.level1n = vfoptions.level1n(1); % Only first one is relevant for DC2A
            else
                error('With ExpAsset DC2A, can only do divide-and-conquer on the first standard endogenous state')
            end
        end
        [VKron, PolicyKron] = ValueFnIter_UFHorz_ExpAsset_DC2A_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        error('Experience Asset requires at least 2 endogenous states.');
    end
    
    if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
        V = VKron;
        Policy = PolicyKron;
        return;
    end
    
    % --- 3. Native UnKron ---
    N_d = max(prod(n_d), 1);
    l_d = length(n_d);
    if N_d == 1 && (isempty(n_d) || n_d(1) == 0); l_d = 0; end
    
    % CRITICAL FIX: Experience Assets only permit choosing a1prime!
    if l_d == 0
        n_actions = n_a(1);
    else
        n_actions = [n_d, n_a(1)];
    end
    l_actions = length(n_actions);
    
    sz_Kron = size(PolicyKron);
    if length(sz_Kron) < 2; sz_Kron = [sz_Kron, 1]; end
    sz_Pol = sz_Kron;
    sz_Pol(1) = l_actions;
    
    Policy = zeros(sz_Pol, 'gpuArray');
    
    cum_dims = 1;
    for i = 1:l_actions
        slice = rem(ceil(PolicyKron / cum_dims) - 1, n_actions(i)) + 1;
        
        if length(sz_Pol) == 6
            Policy(i, :, :, :, :, :) = slice;
        elseif length(sz_Pol) == 5
            Policy(i, :, :, :, :) = slice;
        elseif length(sz_Pol) == 7
            Policy(i, :, :, :, :, :, :) = slice;
        elseif length(sz_Pol) == 4
            Policy(i, :, :, :) = slice;
        else
            error('Unexpected dimensions in PolicyKron');
        end
        
        cum_dims = cum_dims * n_actions(i);
    end
    
    % --- 4. Final Legacy API Shaping ---
    sz_V = n_a;
    if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0; sz_V = [sz_V, vfoptions.n_semiz]; end
    if prod(n_z) > 0; sz_V = [sz_V, n_z]; end
    if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0; sz_V = [sz_V, vfoptions.n_e]; end
    sz_V = [sz_V, N_j];
    
    sz_P = [l_actions, sz_V];
    
    V = reshape(VKron, sz_V);
    Policy = reshape(Policy, sz_P);


end
