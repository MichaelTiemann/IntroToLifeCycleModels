function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_DC1_GI1_e_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% With z and e.
%
% a1: standard endogenous state, this is the one divide-and-conquer (and then the grid interp layer) is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% The EV pipeline is unchanged from the DC1_GI1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset. DiscountedEV is (d3,a1prime,a2prime,-,-,-,z): no a3 term
% (a3prime does not depend on a3), and no e term (the expectation is already taken over e).

% Safely calculate N_d dimensions, treating 0 as a singleton (1) for math
N_d1 = max(1, prod(n_d1(n_d1 > 0)));
N_d2 = max(1, prod(n_d2(n_d2 > 0)));
N_d3 = max(1, prod(n_d3(n_d3 > 0)));
N_a1 = max(1, prod(n_a1(n_a1 > 0)));
N_a2 = max(1, prod(n_a2(n_a2 > 0)));
N_a3 = max(1, prod(n_a3(n_a3 > 0)));
N_a  = N_a1 * N_a2 * N_a3;
N_z  = max(1, prod(n_z(n_z > 0)));
N_e  = max(1, prod(n_e(n_e > 0)));
N_u  = max(1, prod(n_u(n_u > 0)));

N_a12 = N_a1 * N_a2; % the two standard assets, carried forward directly

% For ReturnFn (d1 and d3 only)
n_d13 = [n_d1(n_d1 > 0), n_d3(n_d3 > 0)];
N_d13 = N_d1 * N_d3;
d13_grid = [d1_grid; d3_grid];

% For aprimeFn (d2 and d3)
n_d23 = [n_d2(n_d2 > 0), n_d3(n_d3 > 0)];
N_d23 = N_d2 * N_d3;
d23_grid = [d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(7,N_a,N_z,N_e,N_j,'gpuArray'); % (1)=d1, (2)=d2, (3)=d3, (4)=a1prime midpoint, (5)=a2prime, (6)=a1prime L2
Policy(7,:,:,:,:)=2;
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Setup for GI
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

% Base column vectors
a2_base = gpuArray(0:N_a2-1)';

% Precompute
aind=gpuArray(0:1:N_a-1);
zBind=shiftdim(gpuArray(0:1:N_z-1),-1);    % [1,1,N_z]
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);    % [1,1,1,N_e]
d3ind=repelem(gpuArray(1:1:N_d3)',N_d1,1); % [N_d13,1]; maps full d13-index to d3-component
a1pcol=reshape(0:1:N_a1-1,[1,N_a1]);       % [1,N_a1prime]
a2pcol=reshape(0:1:N_a2-1,[1,1,N_a2]);     % [1,1,N_a2prime]

%% Unified Time Loop
for jj = N_j:-1:1
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', jj, N_j)
    end

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    is_terminal = (jj == N_j) && ~isfield(vfoptions, 'V_Jplus1');
    eval_type_GI = 3 - is_terminal;

    %% Compute EV (Only if not terminal)
    if ~is_terminal
        DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));
        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
        [a3primeIndex, a3primeProbs] = CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec, 2);

        aprimeIndex = repelem((1:1:N_a12)', N_d23, N_u) + N_a12 * repmat(a3primeIndex-1, N_a12, 1);
        aprimeplus1Index = repelem((1:1:N_a12)', N_d23, N_u) + N_a12 * repmat(a3primeIndex, N_a12, 1);

        % 1. Get next period's V and integrate out 'e' immediately!
        if jj == N_j
            V_next = reshape(vfoptions.V_Jplus1, [N_a, N_z, N_e]);
        else
            V_next = V(:, :, :, jj+1);
        end

        % Safely get the shock distributions
        pi_e_step = pi_e_J(:, min(jj+1, size(pi_e_J, 2))); % Capped at max J
        pi_z_step = pi_z_J(:, :, jj);                      % Always safe at jj

        % shiftdim aligns pi_e to dim 3 for exact tensor broadcasting
        EVpre = sum(V_next .* shiftdim(pi_e_step, -2), 3); % Now it is just [N_a, N_z]!

        % 2. Integrate out 'z'
        EV = EVpre .* shiftdim(pi_z_step', -1);
        EV(isnan(EV)) = 0;
        EV = reshape(sum(EV, 2), [N_a, N_z]);

        % 3. Interpolate EV onto risky asset return and integrate 'u'
        skipinterp = logical(EV(aprimeIndex(:) + N_a*((1:1:N_z)-1)) == EV(aprimeplus1Index(:) + N_a*((1:1:N_z)-1)));
        aprimeProbs = repmat(a3primeProbs, N_a12, N_z);
        aprimeProbs(skipinterp) = 0;
        aprimeProbs = reshape(aprimeProbs, [N_d23*N_a12, N_u, N_z]);

        EV1 = reshape(EV(aprimeIndex(:) + N_a*((1:1:N_z)-1)), [N_d23*N_a12, N_u, N_z]) .* aprimeProbs;
        EV2 = reshape(EV(aprimeplus1Index(:) + N_a*((1:1:N_z)-1)), [N_d23*N_a12, N_u, N_z]) .* (1 - aprimeProbs);
        EV1(isnan(EV1)) = 0; EV2(isnan(EV2)) = 0;

        EV_u = sum(EV1 .* pi_u', 2) + sum(EV2 .* pi_u', 2);

        % 4. Refine d2
        [EV_onlyd3, d2index] = max(reshape(EV_u, [N_d2, N_d3*N_a12, N_z]), [], 1);
        EV_onlyd3 = reshape(EV_onlyd3, [N_d3*N_a12, N_z]);
        d2index_resh = reshape(d2index, [N_d3, N_a1, N_a2, N_z]);
    end

    %% Setup Evaluation Loops
    if vfoptions.lowmemory == 0
        z_iter = 1; special_n_z = n_z;
        e_iter = 1; special_n_e = n_e;
        midpoint = zeros(N_d13, 1, N_a2, N_a1, N_a2, N_a3, N_z, N_e, 'gpuArray');
    else
        z_iter = 1:N_z; special_n_z = ones(1, length(n_z));
        e_iter = 1:N_e; special_n_e = ones(1, length(n_e));
        midpoint = zeros(N_d13, 1, N_a2, N_a1, N_a2, N_a3, 'gpuArray');
    end

    for z_c = z_iter
        if vfoptions.lowmemory == 0
            z_val = z_gridvals_J(:, :, jj);
            z_idx = 1:N_z; z_offset = zBind;
        else
            z_val = z_gridvals_J(z_c, :, jj);
            z_idx = z_c; z_offset = 0;
        end

        % JUST-IN-TIME EV SLICING: EV only depends on z, not e!
        if ~is_terminal
            EV_slice = EV_onlyd3(:, z_idx);
            DiscountedEV = DiscountFactorParamsVec * reshape(EV_slice, [N_d3, N_a1, N_a2, 1, 1, 1, length(z_idx)]);
            DiscountedEVinterp = permute(interp1(a1_grid, permute(DiscountedEV, [2,1,3,4,5,6,7]), a1prime_grid), [2,1,3,4,5,6,7]);

            % For DC2A broadcasting, we need d1 replicated
            DiscountedEV_d13 = repelem(DiscountedEV, N_d1, 1);
            DiscountedEVinterp_d13 = repelem(DiscountedEVinterp, N_d1, 1);
        end

        for e_c = e_iter
            if vfoptions.lowmemory == 0
                e_val = e_gridvals_J(:, :, jj);
                e_idx = 1:N_e; e_offset = eBind;
            else
                e_val = e_gridvals_J(e_c, :, jj);
                e_idx = e_c; e_offset = 0;
            end

            % =======================================================
            % Layer 1
            % =======================================================
            ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);

            if is_terminal
                entireRHS_ii = ReturnMatrix_ii;
            else
                d3aprimez = d3ind + N_d3*a1pcol + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(z_offset, -4);
                entireRHS_ii = ReturnMatrix_ii + DiscountedEV_d13(d3aprimez);
            end

            [~, maxindex1] = max(entireRHS_ii, [], 2);
            midpoint(:, 1, :, level1ii, :, :, :, :) = maxindex1;

            % =======================================================
            % Layer 2: Divide and Conquer Sweep
            % =======================================================
            maxgap = squeeze(max(max(max(max(max(max(maxindex1(:, 1, :, 2:end, :, :, :, :) - maxindex1(:, 1, :, 1:end-1, :, :, :, :), [], 8), [], 7), [], 6), [], 5), [], 3), [], 1));

            for ii = 1:(vfoptions.level1n-1)
                curraindex = (level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii) > 0
                    loweredge = min(maxindex1(:, 1, :, ii, :, :, :, :), N_a1 - maxgap(ii));
                    a1primeindexes = loweredge + (0:1:maxgap(ii));
                    ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);

                    if is_terminal
                        entireRHS_ii = ReturnMatrix_ii;
                    else
                        % Exact transposed offsets for 2A
                        d3aprimez = d3ind + N_d3*(a1primeindexes-1) + N_d3*N_a1*shiftdim(a2_base,-2) + N_d3*N_a1*N_a2*shiftdim((0:length(z_idx)-1)', -6);
                        entireRHS_ii = ReturnMatrix_ii + DiscountedEV_d13(d3aprimez);
                    end

                    [~, maxindex] = max(entireRHS_ii, [], 2);
                    midpoint(:, 1, :, curraindex, :, :, :, :) = maxindex + (loweredge - 1);
                else
                    loweredge = maxindex1(:, 1, :, ii, :, :, :, :);
                    midpoint(:, 1, :, curraindex, :, :, :, :) = repelem(loweredge, 1, 1, 1, level1iidiff(ii), 1, 1, 1, 1);
                end
            end

            % =======================================================
            % Layer 3: Grid Interpolation & Final Policy Assembly
            % =======================================================
            % Define local combined ze_offset since e_offset and z_offset were split
            ze_offset = z_offset + N_z*e_offset;

            % Flatten midpoint cleanly (no squeeze) so shape perfectly feeds arrayfun
            midpoint_L2 = midpoint(:);
            a1primeindexesfine = max(1, min(N_a1fine - n2long + 1, (midpoint_L2 - 1) * n2short + 1 + shiftdim(-n2short-1:n2short+1, -1)));

            ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);

            if is_terminal
                entireRHS_ii = ReturnMatrix_ii;
            else
                % Transposed shifts push a2prime to dim 3 and z to dim 7 in the un-merged EV tensor
                aprimez = (1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim(a2_base,-2) + N_d13*N_a1fine*N_a2*shiftdim((0:length(z_idx)-1)',-6);
                entireRHS_ii = reshape(ReturnMatrix_ii + DiscountedEVinterp_d13(aprimez), [N_d13*n2long*N_a2, N_a, length(z_idx), length(e_idx)]);
            end

            entireRHS_ii = reshape(entireRHS_ii, [N_d13*n2long*N_a2, N_a, length(z_idx), length(e_idx)]);
            [Vtempii, maxindexL2] = max(entireRHS_ii, [], 1);

            V(:, z_idx, e_idx, jj) = shiftdim(Vtempii, 1);

            maxindexL2_d_a2 = rem(maxindexL2-1, N_d13*n2long*N_a2) + 1;
            d_ind = rem(maxindexL2_d_a2-1, N_d13) + 1;
            L2flag = rem(ceil(maxindexL2_d_a2/N_d13)-1, n2long) + 1;
            maxindexL2a2 = ceil(maxindexL2_d_a2/(N_d13*n2long));

            allind = d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*ze_offset;
            a1mid = midpoint(allind);

            d1_ind = rem(d_ind-1, N_d1) + 1;
            d3_ind = ceil(d_ind/N_d1);

            Policy(1, :, z_idx, e_idx, jj) = d1_ind;
            Policy(3, :, z_idx, e_idx, jj) = d3_ind;
            Policy(4, :, z_idx, e_idx, jj) = shiftdim(squeeze(a1mid), -1);
            Policy(5, :, z_idx, e_idx, jj) = maxindexL2a2;
            Policy(6, :, z_idx, e_idx, jj) = L2flag;

            if is_terminal
                Policy(2, :, z_idx, e_idx, jj) = 1;
            else
                % The exact transpose column-vector fix for 2A offsets!
                if vfoptions.lowmemory <= 1
                    zlin = shiftdim(gpuArray(0:length(z_idx)-1)', -3);
                else
                    zlin = 0;
                end
                linlookup = d3_ind + N_d3*(a1mid-1) + N_d3*N_a1*(maxindexL2a2-1) + N_d3*N_a1*N_a2*zlin;
                Policy(2, :, z_idx, e_idx, jj) = d2index_resh(linlookup);
            end

        end
    end
end

%% Switch Policy(4,:) from 'midpoint' to 'lower grid index'
adjust=(Policy(6,:,:,:,:)<1+n2short+1);
Policy(4,:,:,:,:)=Policy(4,:,:,:,:)-adjust;
Policy(6,:,:,:,:)=Policy(6,:,:,:,:)-(n2short+1)*(~adjust);

%% Shrink-wrap Policy to remove inactive choice dimensions
has_d1 = (sum(n_d1) > 0);
has_d2 = (sum(n_d2) > 0);
has_d3 = (sum(n_d3) > 0);
has_a1 = (sum(n_a1) > 0);
has_a2 = (sum(n_a2) > 0);

active_rows = [];
if has_d1, active_rows(end+1) = 1; end
if has_d2, active_rows(end+1) = 2; end
if has_d3, active_rows(end+1) = 3; end
if has_a1, active_rows(end+1) = 4; end
if has_a2, active_rows(end+1) = 5; end

% Append the GI2A flags
active_rows = [active_rows, 6, 7];

slice_idx = repmat({':'}, 1, ndims(Policy));
slice_idx{1} = active_rows;
Policy = Policy(slice_idx{:});


end
