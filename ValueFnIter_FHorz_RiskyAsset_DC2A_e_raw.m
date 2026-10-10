function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC2A_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, a3_grid, z_gridvals_J, e_gridvals_J, u_grid, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAsset_DC1_e_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% With d1, with z and e.
%
% a1: standard endogenous state, this is the one divide-and-conquer is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% d1 and d3 enter the ReturnFn, so the return matrix is over the joint d13 (d1 fastest).
% DiscountedEV only depends on d3, so it is repelem-ed up to d13 before being added.
%
% The EV pipeline is unchanged from the DC1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset.

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
Policy=zeros(5,N_a,N_z,N_e,N_j,'gpuArray'); % (1)=d1, (2)=d2, (3)=d3, (4)=a1prime, (5)=a2prime
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d13_gridvals=CreateGridvals(n_d13,d13_grid,1);

if vfoptions.lowmemory==0
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1); % [1,1,N_z]
    eBind=shiftdim(gpuArray(0:1:N_e-1),-2); % [1,1,1,N_e]
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1);
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

% Setup for DC (over a1 only)
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Precompute
a2ind=gpuArray(0:N_a2-1)';
a3ind=gpuArray(0:N_a3-1)';
d3col=repelem((1:1:N_d3)',N_d1,1);     % [N_d13,1]; maps full d13-index to d3-component
a2pcol=reshape(0:1:N_a2-1,[1,1,N_a2]); % [1,1,N_a2prime]

%% Unified Time Loop
for jj = N_j:-1:1
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', jj, N_j)
    end

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    is_terminal = (jj == N_j) && ~isfield(vfoptions, 'V_Jplus1');

    %% Compute EV (Only if not terminal)
    if ~is_terminal
        DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));
        aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);

        % 1. Get next period's V and integrate out 'e' immediately!
        if jj == N_j
            V_next = reshape(vfoptions.V_Jplus1, [N_a, N_z, N_e]);
        else
            V_next = V(:,:,:,jj+1);
        end

        % Safely get the shock distribution (cap at max J to avoid out-of-bounds)
        pi_e_step = pi_e_J(:, min(jj+1, size(pi_e_J, 2)));

        % shiftdim aligns pi_e to dim 3 for exact tensor broadcasting
        EV_base = sum(V_next .* shiftdim(pi_e_step, -2), 3); % Now it is just [N_a, N_z]!

        [a3primeIndex, a3primeProbs] = CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec, 2);
        aprimeIndex = repelem((1:1:N_a12)', N_d23, N_u) + N_a12 * repmat(a3primeIndex-1, N_a12, 1);
        aprimeplus1Index = repelem((1:1:N_a12)', N_d23, N_u) + N_a12 * repmat(a3primeIndex, N_a12, 1);
        baseProbs = repmat(a3primeProbs, N_a12, 1);
    end

    %% Setup Evaluation Loops
    if vfoptions.lowmemory == 0
        e_iter = 1; special_n_e = n_e;
        z_iter = 1; special_n_z = n_z;
    elseif vfoptions.lowmemory == 1
        e_iter = 1:N_e; special_n_e = ones(1, length(n_e));
        z_iter = 1;     special_n_z = n_z;
    else
        e_iter = 1:N_e; special_n_e = ones(1, length(n_e));
        z_iter = 1:N_z; special_n_z = ones(1, length(n_z));
    end

    for z_c = z_iter
        if vfoptions.lowmemory <= 1
            z_val = z_gridvals_J(:,:,jj);
            z_idx = 1:N_z; z_offset = zindB;
        else
            z_val = z_gridvals_J(z_c,:,jj);
            z_idx = z_c; z_offset = 0;
        end

        % JUST-IN-TIME EV SLICING
        if ~is_terminal
            if vfoptions.lowmemory <= 1
                EV_z = EV_base .* shiftdim(pi_z_J(:,:,jj)', -1);
            else
                EV_z = EV_base .* pi_z_J(z_c,:,jj);
            end
            EV_z(isnan(EV_z)) = 0;
            EV_z = sum(EV_z, 2);

            skipinterp = logical(EV_z(aprimeIndex(:)+N_a*((1:length(z_idx))-1)) == EV_z(aprimeplus1Index(:)+N_a*((1:length(z_idx))-1)));
            blockProbs = repmat(baseProbs, 1, length(z_idx));
            blockProbs(skipinterp) = 0;
            blockProbs = reshape(blockProbs, [N_d23*N_a12, N_u, length(z_idx)]);

            EV1 = reshape(EV_z(aprimeIndex(:)+N_a*((1:length(z_idx))-1)), [N_d23*N_a12, N_u, length(z_idx)]) .* blockProbs;
            EV2 = reshape(EV_z(aprimeplus1Index(:)+N_a*((1:length(z_idx))-1)), [N_d23*N_a12, N_u, length(z_idx)]) .* (1 - blockProbs);
            EV1(isnan(EV1)) = 0; EV2(isnan(EV2)) = 0;

            EV_block = sum((EV1 .* pi_u'), 2) + sum((EV2 .* pi_u'), 2);

            % Refine d2 out of EV
            [EV_onlyd3, d2index] = max(reshape(DiscountFactorParamsVec*EV_block, [N_d2, N_d3*N_a12, length(z_idx)]), [], 1);
            d2index_resh = reshape(d2index, [N_d3, N_a1, N_a2, length(z_idx)]);

            DiscountedEV = reshape(EV_onlyd3, [N_d3, N_a1, N_a2, 1, 1, 1, length(z_idx)]);
            DiscountedEV_d13 = repelem(DiscountedEV, N_d1, 1);
        end

        for e_c = e_iter
            if vfoptions.lowmemory == 0
                e_val = e_gridvals_J(:,:,jj);
                e_idx = 1:N_e; ze_offset = zindB + N_z*eBind;
                midpoint_jj = zeros(N_d13, 1, N_a2, N_a1, N_a2, N_a3, N_z, N_e, 'gpuArray');
            else
                e_val = e_gridvals_J(e_c,:,jj);
                e_idx = e_c;
                if vfoptions.lowmemory == 1
                    ze_offset = zindB; % e singular
                    midpoint_jj = zeros(N_d13, 1, N_a2, N_a1, N_a2, N_a3, N_z, 1, 'gpuArray');
                else
                    ze_offset = 0;
                    midpoint_jj = zeros(N_d13, 1, N_a2, N_a1, N_a2, N_a3, 1, 1, 'gpuArray');
                end
            end

            % Layer 1: ReturnMatrix (maxes over a1prime n-monotonicity sweep)
            ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);

            if is_terminal
                entireRHS_ii = ReturnMatrix_ii;
            else
                entireRHS_ii = ReturnMatrix_ii + DiscountedEV_d13;
            end

            [~, maxindex1] = max(entireRHS_ii, [], 2);
            midpoint_jj(:, 1, :, level1ii, :, :, :, :) = maxindex1;

            % Divide-and-conquer layer 2 (squeezes across all outer dimensions to find max gap)
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
                        % Broadcast offset for z in level 3: z is dimension 7, so shift by -6
                        d3aprimez = d3ind + N_d3*(a1primeindexes-1) + N_d3*N_a1*shiftdim((0:N_a2-1)',-2) + N_d3*N_a1*N_a2*shiftdim((0:length(z_idx)-1)', -6);
                        entireRHS_ii = ReturnMatrix_ii + DiscountedEV(d3aprimez);
                    end

                    [~, maxindex] = max(entireRHS_ii, [], 2);
                    midpoint_jj(:, 1, :, curraindex, :, :, :, :) = maxindex + (loweredge - 1);
                else
                    loweredge = maxindex1(:, 1, :, ii, :, :, :, :);
                    midpoint_jj(:, 1, :, curraindex, :, :, :, :) = repelem(loweredge, 1, 1, 1, level1iidiff(ii), 1, 1, 1, 1);
                end
            end

            % Final Policy Assembly
            ReturnMatrix_ii = CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d3, n_a2, n_a3, special_n_z, special_n_e, d13_gridvals, a1_grid(midpoint_jj(:)), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);

            if is_terminal
                entireRHS_ii = ReturnMatrix_ii;
            else
                d3aprimez = (1:1:N_d13)' + N_d13*(midpoint_jj(:)-1) + N_d13*N_a1*shiftdim((0:N_a2-1)',-2) + N_d13*N_a1*N_a2*shiftdim((0:length(z_idx)-1)', -6);
                entireRHS_ii = ReturnMatrix_ii + reshape(DiscountedEV_d13(d3aprimez(:)), [N_d13*N_a2, N_a, length(z_idx), length(e_idx)]);
            end

            entireRHS_ii = reshape(entireRHS_ii, [N_d13*N_a2, N_a, length(z_idx), length(e_idx)]);
            [Vtempii, maxindexL2] = max(entireRHS_ii, [], 1);

            V(:, z_idx, e_idx, jj) = shiftdim(Vtempii, 1);

            d_ind = rem(maxindexL2-1, N_d13) + 1;
            maxindexL2a2 = ceil(maxindexL2/N_d13);
            allind = d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*ze_offset;
            d1_ind = rem(d_ind-1, N_d1) + 1;
            d3_ind = ceil(d_ind/N_d1);

            Policy(1, :, z_idx, e_idx, jj) = d1_ind;
            Policy(3, :, z_idx, e_idx, jj) = d3_ind;
            Policy(4, :, z_idx, e_idx, jj) = shiftdim(squeeze(midpoint_jj(allind)), -1);
            Policy(5, :, z_idx, e_idx, jj) = maxindexL2a2;

            if is_terminal
                Policy(2, :, z_idx, e_idx, jj) = 1;
            else
                a1mid = midpoint_jj(allind);
                if vfoptions.lowmemory <= 1
                    zlin = shiftdim(gpuArray(0:length(z_idx)-1), -2);
                else
                    zlin = 0;
                end
                linlookup = d3_ind + N_d3*(a1mid-1) + N_d3*N_a1*(maxindexL2a2-1) + N_d3*N_a1*N_a2*zlin;
                Policy(2, :, z_idx, e_idx, jj) = d2index_resh(linlookup);
            end
        end
    end
end

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

slice_idx = repmat({':'}, 1, ndims(Policy));
slice_idx{1} = active_rows;
Policy = Policy(slice_idx{:});


end
