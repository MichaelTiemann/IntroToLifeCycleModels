function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% With z, no e.

N_d1 = max(1, prod(n_d1(n_d1 > 0)));
N_d2 = max(1, prod(n_d2(n_d2 > 0)));
N_d3 = max(1, prod(n_d3(n_d3 > 0)));
N_a1 = max(1, prod(n_a1(n_a1 > 0)));
N_a2 = max(1, prod(n_a2(n_a2 > 0)));
N_a  = N_a1 * N_a2;
N_z  = max(1, prod(n_z(n_z > 0)));
N_u  = max(1, prod(n_u(n_u > 0)));

% For ReturnFn (d1 and d3 only)
n_d13 = [n_d1, n_d3];
N_d13 = N_d1 * N_d3;
d13_grid = [d1_grid; d3_grid];

% For aprimeFn (d2 and d3)
n_d23 = [n_d2, n_d3];
N_d23 = N_d2 * N_d3;
d23_grid = [d2_grid; d3_grid];

V = zeros(N_a, N_z, N_j, 'gpuArray');
Policy = zeros(6, N_a, N_z, N_j, 'gpuArray');
% (1)=d1, (2)=d2, (3)=d3, (4)=a1prime midpoint, (5)=a1primeL2ind
Policy(6, :, :, :) = 2; % d2 stored directly into Policy(2,...); no separate d2Policy slab

%%
u_grid = gpuArray(u_grid);

a2_gridvals = CreateGridvals(n_a2(n_a2>0), a2_grid, 1);
a1_gridvals = a1_grid;
d13_gridvals = CreateGridvals(n_d13(n_d13>0), d13_grid, 1);

% Setup for DC
level1ii = round(linspace(1, n_a1, vfoptions.level1n));
level1iidiff = level1ii(2:end) - level1ii(1:end-1) - 1;

% Setup for GI
n2short = vfoptions.ngridinterp;
n2long = vfoptions.ngridinterp * 2 + 3;
a1prime_grid = interp1(1:1:n_a1(1), a1_gridvals, linspace(1, n_a1(1), n_a1(1) + (n_a1(1) - 1) * n2short));
N_a1prime = length(a1prime_grid);

% Precompute Array Indices
aind = gpuArray(0:1:N_a-1);
zBind = shiftdim(gpuArray(0:1:N_z-1)', -2); % [1, 1, N_z]
d3ind = repelem(gpuArray(1:1:N_d3)', N_d1, 1); % [N_d13,1]; maps full d13-index to d3-component

%% Iterate backwards through j
for jj = N_j:-1:1
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', jj, N_j)
    end

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    is_terminal = (jj == N_j) && ~isfield(vfoptions, 'V_Jplus1');

    if ~is_terminal
        DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));

        % Build a2primeIndex and a2primeProbs for RiskyAsset
        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
        [a2primeIndex, a2primeProbs] = CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec, 2);
        aprimeIndex = repelem((1:N_a1)', N_d23, N_u) + N_a1 * repmat(a2primeIndex-1, N_a1, 1);
        aprimeplus1Index = repelem((1:N_a1)', N_d23, N_u) + N_a1 * repmat(a2primeIndex, N_a1, 1);

        % Get EV in terms of next period endogenous states
        if jj == N_j
            EV = reshape(vfoptions.V_Jplus1, [N_a, N_z]);
            pi_z_step = pi_z_J(:, :, N_j);
        else
            EV = V(:, :, jj+1);
            pi_z_step = pi_z_J(:, :, jj);
        end

        EV = EV .* shiftdim(pi_z_step', -1);
        EV(isnan(EV)) = 0;
        EV = reshape(sum(EV, 2), [N_a, N_z]);

        % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
        skipinterp = logical(EV(aprimeIndex(:) + N_a*((1:N_z)-1)) == EV(aprimeplus1Index(:) + N_a*((1:N_z)-1)));
        aprimeProbs = repmat(a2primeProbs, N_a1, N_z);
        aprimeProbs(skipinterp) = 0;
        aprimeProbs = reshape(aprimeProbs, [N_d23*N_a1, N_u, N_z]);

        % Take the expectation over the between period iid u shock
        EV1 = reshape(EV(aprimeIndex(:) + N_a*((1:N_z)-1)), [N_d23*N_a1, N_u, N_z]) .* aprimeProbs;
        EV2 = reshape(EV(aprimeplus1Index(:) + N_a*((1:N_z)-1)), [N_d23*N_a1, N_u, N_z]) .* (1 - aprimeProbs);
        EV1(isnan(EV1)) = 0; % a zero weight against an infinite node gives 0*(-Inf)=NaN, so the term contributes nothing
        EV2(isnan(EV2)) = 0;

        EV_u = sum(EV1 .* pi_u', 2) + sum(EV2 .* pi_u', 2);

        % Refine d2 out of EV before combining with ReturnFn (Retained globally for lookup)
        [EV_onlyd3, d2index] = max(reshape(EV_u, [N_d2, N_d3*N_a1, N_z]), [], 1);
        d2index_resh = reshape(d2index, [N_d3, N_a1, N_z]);
    end

    % Setup Evaluation Blocks based on lowmemory
    if vfoptions.lowmemory == 0
        z_iter = 1; special_n_z = n_z;
        midpoint_jj = zeros(N_d13, 1, N_a1, N_a2, N_z, 'gpuArray');
    else
        z_iter = 1:N_z; special_n_z = ones(1, length(n_z));
        midpoint_jj = zeros(N_d13, 1, N_a1, N_a2, 'gpuArray');
    end

    for z_c = z_iter
        if vfoptions.lowmemory == 0
            z_val = z_gridvals_J(:, :, jj);
            z_idx = 1:N_z;
            z_offset = zBind; % Strict vector matching
        else
            z_val = z_gridvals_J(z_c, :, jj);
            z_idx = z_c;
            z_offset = 0; % Flat tensor handling
        end

        % JUST-IN-TIME CALCULATION: Slices EV_onlyd3 to current z_idx (Eliminates high-memory tensor tracking)
        if ~is_terminal
            EV_slice = EV_onlyd3(:, :, z_idx);
            DiscountedEV = DiscountFactorParamsVec * reshape(EV_slice, [N_d3, N_a1, 1, 1, length(z_idx)]);
            DiscountedEVinterp = permute(interp1(a1_gridvals, permute(DiscountedEV, [2,1,3,4,5]), a1prime_grid), [2,1,3,4,5]);
        end

        % =======================================================
        % Layer 1
        % =======================================================
        ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1, n_d3, n_a1, vfoptions.level1n, n_a2, special_n_z, d13_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec, 1, 0);
        ReturnMatrix_ii = reshape(ReturnMatrix_ii, [N_d1, N_d3, N_a1, vfoptions.level1n, N_a2, length(z_idx)]);

        if is_terminal
            entireRHS_ii = reshape(ReturnMatrix_ii, [N_d13, N_a1, vfoptions.level1n, N_a2, length(z_idx)]);
        else
            % Fully unified via dynamic length(z_idx)
            DEV = reshape(DiscountedEV, [1, N_d3, N_a1, 1, 1, length(z_idx)]);
            entireRHS_ii = reshape(ReturnMatrix_ii + DEV, [N_d13, N_a1, vfoptions.level1n, N_a2, length(z_idx)]);
        end

        [~, maxindex1] = max(entireRHS_ii, [], 2);

        if vfoptions.lowmemory == 0
            midpoint_jj(:, 1, level1ii, :, :) = maxindex1;
            maxgap = squeeze(max(max(max(maxindex1(:, 1, 2:end, :, :) - maxindex1(:, 1, 1:end-1, :, :), [], 5), [], 4), [], 1));
        else
            midpoint_jj(:, 1, level1ii, :) = maxindex1;
            maxgap = squeeze(max(max(maxindex1(:, 1, 2:end, :) - maxindex1(:, 1, 1:end-1, :), [], 4), [], 1));
        end

        % =======================================================
        % Divide-and-conquer layer 2
        % =======================================================
        for ii = 1:(vfoptions.level1n-1)
            curraindex = (level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii) > 0
                if vfoptions.lowmemory == 0
                    loweredge = min(maxindex1(:, 1, ii, :, :), N_a1 - maxgap(ii));
                else
                    loweredge = min(maxindex1(:, 1, ii, :), N_a1 - maxgap(ii));
                end

                a1primeindexes = loweredge + (0:1:maxgap(ii));
                ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1, n_d3, maxgap(ii)+1, level1iidiff(ii), n_a2, special_n_z, d13_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec, 3, 0);

                if is_terminal
                    entireRHS_ii = ReturnMatrix_ii;
                else
                    % Unified array access natively handling z_offset bounds
                    d3aprimez = d3ind + N_d3*(a1primeindexes-1) + N_d3*N_a1*shiftdim(z_offset, -2);
                    entireRHS_ii = ReturnMatrix_ii + DiscountedEV(d3aprimez);
                end

                [~, maxindex] = max(entireRHS_ii, [], 2);

                if vfoptions.lowmemory == 0
                    midpoint_jj(:, 1, curraindex, :, :) = maxindex + (loweredge - 1);
                else
                    midpoint_jj(:, 1, curraindex, :) = maxindex + (loweredge - 1);
                end
            else
                if vfoptions.lowmemory == 0
                    loweredge = maxindex1(:, 1, ii, :, :);
                    midpoint_jj(:, 1, curraindex, :, :) = repelem(loweredge, 1, 1, level1iidiff(ii), 1);
                else
                    loweredge = maxindex1(:, 1, ii, :);
                    midpoint_jj(:, 1, curraindex, :) = repelem(loweredge, 1, 1, level1iidiff(ii), 1);
                end
            end
        end

        % =======================================================
        % Grid interpolation layer
        % =======================================================
        midpoint_jj = max(min(midpoint_jj, n_a1(1)-1), 2);
        a1primeindexesfine = (midpoint_jj + (midpoint_jj-1)*n2short) + (-n2short-1:1:1+n2short);

        ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1, n_d3, n2long, n_a1, n_a2, special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec, 2, 0);

        if is_terminal
            entireRHS_ii = reshape(ReturnMatrix_ii, [N_d13*n2long, N_a, length(z_idx)]);
        else
            % Fully unified via dynamic length(z_idx)
            da1primez = d3ind + N_d3*(a1primeindexesfine-1) + N_d3*N_a1prime*shiftdim(z_offset, -2);
            entireRHS_ii = reshape(reshape(ReturnMatrix_ii, [N_d13, n2long, N_a1, N_a2, length(z_idx)]) + reshape(DiscountedEVinterp(da1primez), [N_d13, n2long, N_a1, N_a2, length(z_idx)]), [N_d13*n2long, N_a, length(z_idx)]);
        end

        [Vtempii, maxindexL2] = max(entireRHS_ii, [], 1);
        V(:, z_idx, jj) = shiftdim(Vtempii, 1);

        d_ind = rem(maxindexL2-1, N_d13) + 1; % Index within d13

        if vfoptions.lowmemory == 0
            allind = d_ind + N_d13*aind + N_d13*N_a*zBind;
            linidx_lower = d_ind + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
            linidx_upper = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind + N_d13*n2long*N_a*zBind;
            ReturnMatrix_ii_resh = reshape(ReturnMatrix_ii, [N_d13, n2long, N_a1, N_a2, N_z]);
        else
            allind = d_ind + N_d13*aind;
            linidx_lower = d_ind + N_d13*n2long*aind;
            linidx_upper = d_ind + N_d13*(n2long-1) + N_d13*n2long*aind;
            ReturnMatrix_ii_resh = reshape(ReturnMatrix_ii, [N_d13, n2long, N_a1, N_a2]);
        end

        Policy(1, :, z_idx, jj) = rem(d_ind-1, N_d1) + 1; % d1
        Policy(3, :, z_idx, jj) = ceil(d_ind/N_d1);       % d3
        Policy(4, :, z_idx, jj) = midpoint_jj(allind);
        Policy(5, :, z_idx, jj) = ceil(maxindexL2/N_d13); % L2flag

        L2offset = ceil(maxindexL2/N_d13);
        isInfLower = (ReturnMatrix_ii_resh(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_resh(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);

        Policy(6, :, z_idx, jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        if is_terminal
            Policy(2, :, z_idx, jj) = 1; % Meaningless terminal value
        else
            a1mid = midpoint_jj(allind);
            if vfoptions.lowmemory == 0
                linlookup = ceil(d_ind/N_d1) + N_d3*(a1mid-1) + N_d3*N_a1*zBind;
                Policy(2, :, z_idx, jj) = d2index_resh(linlookup);
            else
                linlookup = ceil(d_ind/N_d1) + N_d3*(a1mid-1);
                Policy(2, :, z_idx, jj) = d2index_resh(linlookup + N_d3*N_a1*(z_c-1));
            end
        end
    end
end

% Switch Policy(4,:) from 'midpoint' to 'lower grid index'
adjust = (Policy(5, :, :, :) < 1 + n2short + 1);
Policy(4, :, :, :) = Policy(4, :, :, :) - adjust;
Policy(5, :, :, :) = Policy(5, :, :, :) - (n2short+1)*(~adjust);

end