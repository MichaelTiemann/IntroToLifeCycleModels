function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_DC1_GI1_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% With z.
%
% a1: standard endogenous state, this is the one divide-and-conquer (and then the grid interp layer) is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% The EV pipeline is unchanged from the DC1_GI1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset. DiscountedEV is (d3,a1prime,a2prime,-,-,-,z), no a3 term.

N_d1 = max(1, prod(n_d1(n_d1 > 0)));
N_d2 = max(1, prod(n_d2(n_d2 > 0)));
N_d3 = max(1, prod(n_d3(n_d3 > 0)));
N_a1 = max(1, prod(n_a1(n_a1 > 0)));
N_a2 = max(1, prod(n_a2(n_a2 > 0)));
N_a3 = max(1, prod(n_a3(n_a3 > 0)));
N_a  = N_a1 * N_a2 * N_a3;
N_z  = max(1, prod(n_z(n_z > 0)));
N_u  = max(1, prod(n_u(n_u > 0)));

N_a12 = N_a1 * N_a2; % the two standard assets, carried forward directly

n_d13 = [n_d1, n_d3];
N_d13 = N_d1 * N_d3;
d13_grid = [d1_grid; d3_grid];

n_d23 = [n_d2, n_d3];
N_d23 = N_d2 * N_d3;
d23_grid = [d2_grid; d3_grid];

V = zeros(N_a, N_z, N_j, 'gpuArray');
Policy = zeros(7, N_a, N_z, N_j, 'gpuArray');
% (1)=d1, (2)=d2, (3)=d3, (4)=a1prime midpoint, (5)=a2prime, (6)=a1prime L2
Policy(7, :, :, :) = 2; % We will refine away d2 out of EV before combining with ReturnFn

%% Precompute Grids and Setup DC/GI
u_grid = gpuArray(u_grid);
a2_gridvals = CreateGridvals(n_a2(n_a2>0), a2_grid, 1);
d13_gridvals = CreateGridvals(n_d13(n_d13>0), d13_grid, 1);

% Setup for DC (over a1 only)
level1ii = round(linspace(1, N_a1, vfoptions.level1n));
level1iidiff = level1ii(2:end) - level1ii(1:end-1) - 1;

% Setup for GI
n2short = vfoptions.ngridinterp;
n2long = vfoptions.ngridinterp * 2 + 3;
a1prime_grid = interp1(1:1:N_a1, a1_grid, linspace(1, N_a1, N_a1 + (N_a1 - 1) * n2short))';
N_a1fine = length(a1prime_grid);

% Precompute Arrays
aind = gpuArray(0:1:N_a-1);
zBind = shiftdim(gpuArray(0:1:N_z-1), -1); % [1,1,N_z]
d3ind = repelem(gpuArray(1:1:N_d3)', N_d1, 1); % [N_d13,1]; maps full d13-index to d3-component
a1pcol = reshape(0:1:N_a1-1, [1, N_a1]);       % [1,N_a1prime]
a2pcol = reshape(0:1:N_a2-1, [1, 1, N_a2]);    % [1,1,N_a2prime]

%% Iterate backwards through j
for jj = N_j:-1:1
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', jj, N_j)
    end

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    is_terminal = (jj == N_j) && ~isfield(vfoptions, 'V_Jplus1');
    eval_type_GI = 3 - is_terminal; % Returns 2 for terminal, 3 for backward (Legacy exact matching)

    if ~is_terminal
        DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));

        % Build a3primeIndex and a3primeProbs for RiskyAsset
        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
        [a3primeIndex, a3primeProbs] = CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec, 2);

        aprimeIndex = repelem((1:1:N_a12)', N_d23, N_u) + N_a12 * repmat(a3primeIndex-1, N_a12, 1);
        aprimeplus1Index = repelem((1:1:N_a12)', N_d23, N_u) + N_a12 * repmat(a3primeIndex, N_a12, 1);

        % Get EV in terms of next period endogenous states
        if jj == N_j
            EVnext = reshape(vfoptions.V_Jplus1, [N_a, N_z]);
            pi_z_step = pi_z_J(:, :, N_j);
        else
            EVnext = V(:, :, jj+1);
            pi_z_step = pi_z_J(:, :, jj);
        end

        EV = EVnext .* shiftdim(pi_z_step', -1);
        EV(isnan(EV)) = 0;
        EV = reshape(sum(EV, 2), [N_a, N_z]);

        % Interpolate EV onto aprime, use skipinterp to avoid numerical errors
        skipinterp = logical(EV(aprimeIndex(:) + N_a*((1:1:N_z)-1)) == EV(aprimeplus1Index(:) + N_a*((1:1:N_z)-1)));
        aprimeProbs = repmat(a3primeProbs, N_a12, N_z);
        aprimeProbs(skipinterp) = 0;
        aprimeProbs = reshape(aprimeProbs, [N_d23*N_a12, N_u, N_z]);

        % Take the expectation over the between period iid u shock
        EV1 = reshape(EV(aprimeIndex(:) + N_a*((1:1:N_z)-1)), [N_d23*N_a12, N_u, N_z]) .* aprimeProbs;
        EV2 = reshape(EV(aprimeplus1Index(:) + N_a*((1:1:N_z)-1)), [N_d23*N_a12, N_u, N_z]) .* (1 - aprimeProbs);
        EV1(isnan(EV1)) = 0; % a zero weight against an infinite node gives NaN
        EV2(isnan(EV2)) = 0;

        EV_u = sum(EV1 .* pi_u', 2) + sum(EV2 .* pi_u', 2);

        % Refine d2 out of EV before combining with ReturnFn
        [EV_onlyd3, d2index] = max(reshape(EV_u, [N_d2, N_d3*N_a12, N_z]), [], 1);
        EV_onlyd3 = reshape(EV_onlyd3, [N_d3*N_a12, N_z]);
        d2index_resh = reshape(d2index, [N_d3, N_a1, N_a2, N_z]);
    end

    % Setup Evaluation Blocks based on lowmemory
    if vfoptions.lowmemory == 0
        z_iter = 1; special_n_z = n_z;
        midpoint_jj = zeros(N_d13, 1, N_a2, N_a1, N_a2, N_a3, N_z, 'gpuArray');
    else
        z_iter = 1:N_z; special_n_z = ones(1, length(n_z));
        midpoint_jj = zeros(N_d13, 1, N_a2, N_a1, N_a2, N_a3, 'gpuArray');
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

        % JUST-IN-TIME CALCULATION: Slices EV_onlyd3 to current z_idx
        if ~is_terminal
            EV_slice = EV_onlyd3(:, z_idx);
            DiscountedEV = DiscountFactorParamsVec * reshape(EV_slice, [N_d3, N_a1, N_a2, 1, 1, 1, length(z_idx)]);
            DiscountedEVinterp = permute(interp1(a1_grid, permute(DiscountedEV, [2,1,3,4,5,6,7]), a1prime_grid), [2,1,3,4,5,6,7]);
        end

        % =======================================================
        % Layer 1
        % =======================================================
        ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);

        if is_terminal
            entireRHS_ii = ReturnMatrix_ii;
        else
            d3aprimez = d3ind + N_d3*a1pcol + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(z_offset, -4);
            entireRHS_ii = ReturnMatrix_ii + DiscountedEV(d3aprimez);
        end

        [~, maxindex1] = max(entireRHS_ii, [], 2);

        if vfoptions.lowmemory == 0
            midpoint_jj(:, 1, :, level1ii, :, :, :) = maxindex1;
            maxgap = squeeze(max(max(max(max(max( maxindex1(:, 1, :, 2:end, :, :, :) - maxindex1(:, 1, :, 1:end-1, :, :, :), [], 7), [], 6), [], 5), [], 3), [], 1));
        else
            midpoint_jj(:, 1, :, level1ii, :, :) = maxindex1;
            maxgap = squeeze(max(max(max(max( maxindex1(:, 1, :, 2:end, :, :) - maxindex1(:, 1, :, 1:end-1, :, :), [], 6), [], 5), [], 3), [], 1));
        end

        % =======================================================
        % Divide-and-conquer layer 2
        % =======================================================
        for ii = 1:(vfoptions.level1n-1)
            curra1inner = (level1ii(ii)+1:1:level1ii(ii+1)-1)';
            if maxgap(ii) > 0
                if vfoptions.lowmemory == 0
                    loweredge = min(maxindex1(:, 1, :, ii, :, :, :), N_a1 - maxgap(ii));
                else
                    loweredge = min(maxindex1(:, 1, :, ii, :, :), N_a1 - maxgap(ii));
                end

                a1primeindexes = loweredge + (0:1:maxgap(ii));
                ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);

                if is_terminal
                    entireRHS_ii = ReturnMatrix_ii;
                else
                    d3aprimez = d3ind + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(z_offset, -4);
                    entireRHS_ii = ReturnMatrix_ii + DiscountedEV(d3aprimez);
                end

                [~, maxindex_inner] = max(entireRHS_ii, [], 2);

                if vfoptions.lowmemory == 0
                    midpoint_jj(:, 1, :, curra1inner, :, :, :) = maxindex_inner + (loweredge - 1);
                else
                    midpoint_jj(:, 1, :, curra1inner, :, :) = maxindex_inner + (loweredge - 1);
                end
            else
                if vfoptions.lowmemory == 0
                    loweredge = maxindex1(:, 1, :, ii, :, :, :);
                    midpoint_jj(:, 1, :, curra1inner, :, :, :) = repelem(loweredge, 1, 1, 1, level1iidiff(ii), 1, 1, 1);
                else
                    loweredge = maxindex1(:, 1, :, ii, :, :);
                    midpoint_jj(:, 1, :, curra1inner, :, :) = repelem(loweredge, 1, 1, 1, level1iidiff(ii), 1, 1);
                end
            end
        end

        % =======================================================
        % Grid interpolation layer
        % =======================================================
        midpoint_jj = max(min(midpoint_jj, N_a1-1), 2);
        a1primeindexesfine = (midpoint_jj + (midpoint_jj-1)*n2short) + (-n2short-1:1:1+n2short);

        ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, eval_type_GI);

        if is_terminal
            entireRHS_ii = reshape(ReturnMatrix_ii, [N_d13*n2long*N_a2, N_a, length(z_idx)]);
        else
            aprimez = d3ind + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(z_offset, -4);
            entireRHS_ii = reshape(ReturnMatrix_ii + DiscountedEVinterp(aprimez), [N_d13*n2long*N_a2, N_a, length(z_idx)]);
        end

        [Vtempii, maxindexL2] = max(entireRHS_ii, [], 1);
        V(:, z_idx, jj) = shiftdim(Vtempii, 1);

        d_ind        = rem(maxindexL2-1, N_d13) + 1;
        d1_ind       = rem(d_ind-1, N_d1) + 1;
        d3_ind       = ceil(d_ind/N_d1);
        maxindexL2a1 = rem(floor((maxindexL2-1)/N_d13), n2long) + 1;
        maxindexL2a2 = floor((maxindexL2-1)/(N_d13*n2long)) + 1;

        if vfoptions.lowmemory == 0
            allind = d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zBind;
            ReturnMatrix_ii_flat = reshape(ReturnMatrix_ii, [N_d13*n2long*N_a2, N_a, N_z]);
            linidx_lower = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zBind;
            linidx_upper = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zBind;
        else
            allind = d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
            ReturnMatrix_ii_flat = reshape(ReturnMatrix_ii, [N_d13*n2long*N_a2, N_a]);
            linidx_lower = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
            linidx_upper = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
        end

        a1mid = midpoint_jj(allind);

        Policy(1, :, z_idx, jj) = d1_ind;
        Policy(3, :, z_idx, jj) = d3_ind;
        Policy(4, :, z_idx, jj) = a1mid;
        Policy(5, :, z_idx, jj) = maxindexL2a2;
        Policy(6, :, z_idx, jj) = maxindexL2a1; % L2flag

        isInfLower = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
        isInfUpper = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
        inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
        Policy(7, :, z_idx, jj) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        if is_terminal
            Policy(2, :, z_idx, jj) = ones(1, N_a, length(z_idx), 'gpuArray');
        else
            if vfoptions.lowmemory == 0
                lin = d3_ind + N_d3*(a1mid-1) + N_d3*N_a1*(maxindexL2a2-1) + N_d3*N_a1*N_a2*zBind;
                Policy(2, :, z_idx, jj) = d2index_resh(lin);
            else
                lin = d3_ind + N_d3*(a1mid-1) + N_d3*N_a1*(maxindexL2a2-1);
                Policy(2, :, z_idx, jj) = d2index_resh(lin + N_d3*N_a1*N_a2*(z_c-1));
            end
        end
    end
end

% Switch Policy(4,:) from 'midpoint' to 'lower grid index'
adjust = (Policy(6, :, :, :) < 1 + n2short + 1);
Policy(4, :, :, :) = Policy(4, :, :, :) - adjust;
Policy(6, :, :, :) = Policy(6, :, :, :) - (n2short+1)*(~adjust);

end