function [V, Policy] = ValueFnIter_FHorz_DC2A_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_uraw_FHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_d = Dims.N_d; N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
has_d2 = Dims.has_d2; has_d1 = Dims.has_d1; N_semiz_raw = Dims.N_semiz_raw;

N_a1 = n_a(1); N_a2 = n_a(2:end);
a1_grid = a_grid(1:N_a1); a2_grid = a_grid(N_a1+1:end);

level1ii = round(linspace(1, N_a1, vfoptions.level1n));
level1iidiff = level1ii(2:end) - level1ii(1:end-1) - 1;

% Preallocate outputs
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
if has_d2 && N_semiz_raw > 0
    Policy = zeros(3, N_a, N_bothz, N_e, N_j, 'gpuArray');
else
    Policy = zeros(1, N_a, N_bothz, N_e, N_j, 'gpuArray');
end

% Indexers for refinement steps
a2ind = gpuArray(0:1:N_a2-1);
a2Bind = shiftdim(gpuArray(0:1:N_a2-1), -1);
bothzind = shiftdim(gpuArray(0:1:N_bothz-1), -1);
bothzBind = shiftdim(gpuArray(0:1:N_bothz-1), -2);
semizind = shiftdim(gpuArray(0:1:N_semiz-1), -1);
semizBind = shiftdim(gpuArray(0:1:N_semiz-1), -2);
eind = shiftdim(gpuArray(0:1:N_e-1), -2);

% --- 2. Backward Induction Loop ---
for j = N_j:-1:1
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', j, N_j); end

    if j == N_j
        if isfield(vfoptions, 'V_Jplus1'); V_next = reshape(vfoptions.V_Jplus1, [N_a, N_bothz, N_e]); else; V_next = []; end
    else
        V_next = V(:, :, :, j+1);
    end

    if has_d2 && N_semiz_raw > 0
        [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period(j, V_next);
    else
        [V(:,:,:,j), Policy(1,:,:,:,j)] = solve_period(j, V_next);
    end
end

% =====================================================================
% HELPER FUNCTION: Solves a single period j
% =====================================================================
    function [V_j, Pol_j] = solve_period(j, V_next)
        ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
        has_continuation = ~isempty(V_next);

        if has_continuation
            DiscountFactorParamsVec = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, j));
            EV = sum(V_next .* Probs.pi_e_J(1, 1, :, j+1), 3);
        end

        V_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        Policy_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        % LOW MEMORY 0
        if vfoptions.lowmemory == 0
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                if has_continuation; EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz); end

                RM_ii = call_ReturnMatrix_DC2A(Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 1);

                if has_continuation
                    DiscountedEV = DiscountFactorParamsVec * reshape(EV_d2, [1, N_a1, N_a2, 1, 1, N_bothz]);
                    entireRHS_ii = RM_ii + DiscountedEV;
                else
                    entireRHS_ii = RM_ii;
                end

                [~, maxindex1] = max(entireRHS_ii, [], 2);
                [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n*N_a2, N_bothz, N_e]), [], 1);

                curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(a2ind', vfoptions.level1n, 1);
                V_ford2_jj(curraindex,:,:,d2_c) = shiftdim(Vtempii, 1);
                Policy_ford2_jj(curraindex,:,:,d2_c) = shiftdim(maxindex2, 1);

                maxgap = squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
                for ii = 1:(vfoptions.level1n-1)
                    curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)', N_a2, 1) + N_a1*repelem(a2ind', level1iidiff(ii), 1);
                    loweredge = min(maxindex1(:,1,:,ii,:,:,:), N_a1-maxgap(ii));
                    reshape_size = [N_d1*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_bothz, N_e];

                    ReturnFnHandle = @(a1p) call_ReturnMatrix_DC2A(Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 2);

                    if has_continuation
                        aprimez = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind + N_a*bothzBind;
                        EV_slice = DiscountedEV(reshape(aprimez, reshape_size));
                        if N_d1 > 1; EV_slice = repelem(EV_slice, N_d1, 1); end
                        [Vtempii, maxindex, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d1, N_a1, reshape_size, EV_slice);
                    else
                        [Vtempii, maxindex, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d1, N_a1, reshape_size);
                    end

                    V_ford2_jj(curraindex,:,:,d2_c) = shiftdim(Vtempii, 1);
                    allind = dind + N_d1*a2primeind + N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d1*N_a2*N_a2*bothzind + N_d1*N_a2*N_a2*N_bothz*eind;
                    Policy_ford2_jj(curraindex,:,:,d2_c) = shiftdim(maxindex + N_d1*(loweredge(allind)-1), 1);
                end
            end

            % LOW MEMORY 1
        elseif vfoptions.lowmemory == 1
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                if has_continuation; EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz); end

                for e_c = 1:N_e
                    e_val = Grids.e_gridvals_J(e_c,:,j);
                    RM_ii = call_ReturnMatrix_DC2A(Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 1);

                    if has_continuation
                        DiscountedEV = DiscountFactorParamsVec * reshape(EV_d2, [1, N_a1, N_a2, 1, 1, N_bothz]);
                        entireRHS_ii = RM_ii + DiscountedEV;
                    else
                        entireRHS_ii = RM_ii;
                    end

                    [~, maxindex1] = max(entireRHS_ii, [], 2);
                    [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n*N_a2, N_bothz]), [], 1);

                    curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(a2ind', vfoptions.level1n, 1);
                    V_ford2_jj(curraindex,:,e_c,d2_c) = shiftdim(Vtempii, 1);
                    Policy_ford2_jj(curraindex,:,e_c,d2_c) = shiftdim(maxindex2, 1);

                    maxgap = squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:),[],6),[],5),[],3),[],1));
                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)', N_a2, 1) + N_a1*repelem(a2ind', level1iidiff(ii), 1);
                        loweredge = min(maxindex1(:,1,:,ii,:,:), N_a1-maxgap(ii));
                        reshape_size = [N_d1*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_bothz, 1];

                        ReturnFnHandle = @(a1p) call_ReturnMatrix_DC2A(Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 2);

                        if has_continuation
                            aprimez = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind + N_a*bothzBind;
                            EV_slice = DiscountedEV(reshape(aprimez, reshape_size));
                            if N_d1 > 1; EV_slice = repelem(EV_slice, N_d1, 1); end
                            [Vtempii, maxindex, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d1, N_a1, reshape_size, EV_slice);
                        else
                            [Vtempii, maxindex, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d1, N_a1, reshape_size);
                        end

                        V_ford2_jj(curraindex,:,e_c,d2_c) = shiftdim(Vtempii, 1);
                        allind = dind + N_d1*a2primeind + N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d1*N_a2*N_a2*bothzind;
                        Policy_ford2_jj(curraindex,:,e_c,d2_c) = shiftdim(maxindex + N_d1*(loweredge(allind)-1), 1);
                    end
                end
            end

            % LOW MEMORY 2
        elseif vfoptions.lowmemory == 2
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                for z_c = 1:N_z
                    semizblock = (z_c-1)*N_semiz + (1:N_semiz);
                    if has_continuation; EV_d2z = compute_expected_value_cond(EV, pi_bothz(semizblock,:)', N_a); end

                    for e_c = 1:N_e
                        e_val = Grids.e_gridvals_J(e_c,:,j);
                        RM_ii = call_ReturnMatrix_DC2A(Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 1);

                        if has_continuation
                            DiscountedEV_z = DiscountFactorParamsVec * reshape(EV_d2z, [1, N_a1, N_a2, 1, 1, N_semiz]);
                            entireRHS_ii = RM_ii + DiscountedEV_z;
                        else
                            entireRHS_ii = RM_ii;
                        end

                        [~, maxindex1] = max(entireRHS_ii, [], 2);
                        [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n*N_a2, N_semiz]), [], 1);

                        curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(a2ind', vfoptions.level1n, 1);
                        V_ford2_jj(curraindex,semizblock,e_c,d2_c) = shiftdim(Vtempii, 1);
                        Policy_ford2_jj(curraindex,semizblock,e_c,d2_c) = shiftdim(maxindex2, 1);

                        maxgap = squeeze(max(max(max(maxindex1(:,1,:,2:end,:)-maxindex1(:,1,:,1:end-1,:),[],5),[],3),[],1));
                        for ii = 1:(vfoptions.level1n-1)
                            curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)', N_a2, 1) + N_a1*repelem(a2ind', level1iidiff(ii), 1);
                            loweredge = min(maxindex1(:,1,:,ii,:), N_a1-maxgap(ii));
                            reshape_size = [N_d1*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, N_semiz, 1];

                            ReturnFnHandle = @(a1p) call_ReturnMatrix_DC2A(Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 2);

                            if has_continuation
                                aprimez = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind + N_a*semizBind;
                                EV_slice = DiscountedEV_z(reshape(aprimez, reshape_size));
                                if N_d1 > 1; EV_slice = repelem(EV_slice, N_d1, 1); end
                                [Vtempii, maxindex, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d1, N_a1, reshape_size, EV_slice);
                            else
                                [Vtempii, maxindex, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d1, N_a1, reshape_size);
                            end

                            V_ford2_jj(curraindex,semizblock,e_c,d2_c) = shiftdim(Vtempii, 1);
                            allind = dind + N_d1*a2primeind + N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii)) + N_d1*N_a2*N_a2*semizind;
                            Policy_ford2_jj(curraindex,semizblock,e_c,d2_c) = shiftdim(maxindex + N_d1*(loweredge(allind)-1), 1);
                        end
                    end
                end
            end

            % LOW MEMORY 3
        elseif vfoptions.lowmemory == 3
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                for z_c = 1:N_bothz
                    if has_continuation; EV_d2z = compute_expected_value_cond(EV, pi_bothz(z_c,:)', N_a); end

                    for e_c = 1:N_e
                        e_val = Grids.e_gridvals_J(e_c,:,j);
                        RM_ii = call_ReturnMatrix_DC2A(Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 1);

                        if has_continuation
                            DiscountedEV_z = DiscountFactorParamsVec * reshape(EV_d2z, [1, N_a1, N_a2, 1, 1, 1]);
                            entireRHS_ii = RM_ii + DiscountedEV_z;
                        else
                            entireRHS_ii = RM_ii;
                        end

                        [~, maxindex1] = max(entireRHS_ii, [], 2);
                        [Vtempii, maxindex2] = max(reshape(entireRHS_ii, [N_d1*N_a1*N_a2, vfoptions.level1n*N_a2]), [], 1);

                        curraindex = repmat(level1ii', N_a2, 1) + N_a1*repelem(a2ind', vfoptions.level1n, 1);
                        V_ford2_jj(curraindex,z_c,e_c,d2_c) = shiftdim(Vtempii, 1);
                        Policy_ford2_jj(curraindex,z_c,e_c,d2_c) = shiftdim(maxindex2, 1);

                        maxgap = squeeze(max(max(maxindex1(:,1,:,2:end)-maxindex1(:,1,:,1:end-1),[],3),[],1));
                        for ii = 1:(vfoptions.level1n-1)
                            curraindex = repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)', N_a2, 1) + N_a1*repelem(a2ind', level1iidiff(ii), 1);
                            loweredge = min(maxindex1(:,1,:,ii), N_a1-maxgap(ii));
                            reshape_size = [N_d1*(maxgap(ii)+1)*N_a2, level1iidiff(ii)*N_a2, 1, 1];

                            ReturnFnHandle = @(a1p) call_ReturnMatrix_DC2A(Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_grid(a1p), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 2);

                            if has_continuation
                                aprimez = repelem(loweredge+(0:1:maxgap(ii)),1,1,1,level1iidiff(ii),1,1) + N_a1*a2Bind;
                                EV_slice = DiscountedEV_z(reshape(aprimez, reshape_size));
                                if N_d1 > 1; EV_slice = repelem(EV_slice, N_d1, 1); end
                                [Vtempii, maxindex, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d1, N_a1, reshape_size, EV_slice);
                            else
                                [Vtempii, maxindex, dind, a2primeind] = RefineSearch_DC2A(ReturnFnHandle, loweredge, maxgap(ii), N_d1, N_a1, reshape_size);
                            end

                            V_ford2_jj(curraindex,z_c,e_c,d2_c) = shiftdim(Vtempii, 1);
                            allind = dind + N_d1*a2primeind + N_d1*N_a2*repelem(a2ind,1,level1iidiff(ii));
                            Policy_ford2_jj(curraindex,z_c,e_c,d2_c) = shiftdim(maxindex + N_d1*(loweredge(allind)-1), 1);
                        end
                    end
                end
            end
        end

        % Resolve max over d2
        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_d2(Policy_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = shiftdim(Policy_ford2_jj, -1);
        end
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function RM = call_ReturnMatrix_DC2A(n_d_in, n_bothz_in, n_e_in, d_val, a1p_val, a2p_val, a1_val, a2_val, z_val, e_val, ParamsVec, Refine)
        if Dims.N_e_raw == 0
            RM = CreateReturnFnMatrix_Disc_DC2A(ReturnFn, n_d_in, n_bothz_in, d_val, a1p_val, a2p_val, a1_val, a2_val, z_val, ParamsVec, Refine, 0);
        else
            RM = CreateReturnFnMatrix_Disc_DC2A_e(ReturnFn, n_d_in, n_bothz_in, n_e_in, d_val, a1p_val, a2p_val, a1_val, a2_val, z_val, e_val, ParamsVec, Refine, 0);
        end
    end

    function d_slice = get_d_slice(d12_gridvals, d2_c, has_d2, has_d1)
        if has_d2 && has_d1; d_slice = d12_gridvals(:,:,d2_c); else; d_slice = d12_gridvals; end
    end

    function Pol3 = extract_policy_d2(Policy_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1)
        Pol3 = zeros(3, N_a, N_bothz, N_e, 'gpuArray');
        Pol3(2,:,:,:) = shiftdim(maxindex_d2, -1);
        linear_d2 = reshape(maxindex_d2, [N_a * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_d2 - 1);
        d1aprime_ind = reshape(Policy_ford2_jj(lin_idx), [1, N_a, N_bothz, N_e]);
        Pol3(1,:,:,:) = shiftdim(rem(d1aprime_ind - 1, N_d1) + 1, -1);
        Pol3(3,:,:,:) = shiftdim(ceil(d1aprime_ind / N_d1), -1);
    end

    function EV_out = compute_expected_value(EV, pi_bothz, N_a, N_bothz)
        EVinf = (EV == -Inf); EV_val = EV; EV_val(EVinf) = -1e250;
        EV_val = EV_val * pi_bothz'; EV_val(EVinf * (pi_bothz' > 0) > 0) = -Inf;
        EV_out = reshape(EV_val, [N_a, 1, N_bothz]);
    end

    function EV_out = compute_expected_value_cond(EV, pi_slice, N_a)
        EV_val = EV .* shiftdim(pi_slice, -1); EV_val(isnan(EV_val)) = 0;
        EV_out = sum(EV_val, 2);
    end


end
