function [V, Policy] = ValueFnIter_FHorz_GI1_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_uraw_FHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e;
N_d = Dims.N_d; N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
has_d2 = Dims.has_d2; has_d1 = Dims.has_d1; N_semiz_raw = Dims.N_semiz_raw;
has_d = Dims.has_d1 || Dims.has_d2;

% Grid interpolation setup
n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
aprime_grid = interp1(1:1:N_a, a_grid, linspace(1, N_a, N_a+(N_a-1)*n2short));
n2aprime = length(aprime_grid);

% Preallocate outputs
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
if has_d2 && N_semiz_raw > 0
    Policy = zeros(5, N_a, N_bothz, N_e, N_j, 'gpuArray');
    Policy(5,:,:,:,:) = 2;
else
    Policy = zeros(4, N_a, N_bothz, N_e, N_j, 'gpuArray');
    Policy(4,:,:,:,:) = 2;
end

% Indexers for refinement steps
aind = gpuArray(0:1:N_a-1);
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

    [V(:,:,:,j), Policy(:,:,:,:,j)] = solve_period(j, V_next);
end

% Adjust Policy Flags and Shrink-Wrap
if has_d2 && N_semiz_raw > 0
    adjust = (Policy(4,:,:,:,:) < 1+n2short+1);
    Policy(3,:,:,:,:) = Policy(3,:,:,:,:) - adjust;
    Policy(4,:,:,:,:) = Policy(4,:,:,:,:) - (n2short+1).*(~adjust);
    if ~has_d; Policy = Policy(2:end, :, :, :, :); end
else
    adjust = (Policy(3,:,:,:,:) < 1+n2short+1);
    Policy(2,:,:,:,:) = Policy(2,:,:,:,:) - adjust;
    Policy(3,:,:,:,:) = Policy(3,:,:,:,:) - (n2short+1).*(~adjust);
    if ~has_d; Policy = Policy(2:end, :, :, :, :); end
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
        maxidx_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        midpt_ford2_jj = zeros(N_a, N_bothz, N_e, N_d2, 'gpuArray');
        flag_ford2_jj = 2 * ones(N_a, N_bothz, N_e, N_d2, 'gpuArray');

        % LOW MEMORY 0
        if vfoptions.lowmemory == 0
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);

                if has_continuation
                    EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                    EVinterp = interp1(a_grid, EV_d2, aprime_grid);
                end

                RM_L1 = call_RM_L1(Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec);
                RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, N_bothz, N_e]);

                if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2, -1); else; entireRHS_L1 = RM_L1; end

                [~, maxindex1] = max(entireRHS_L1, [], 2);
                midpoint = max(min(maxindex1, N_a-1), 2);
                aprimeindexes = (midpoint + (midpoint-1)*n2short) + (-n2short-1:1:1+n2short);

                RM_L2 = call_RM_L2(Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, reshape(aprime_grid(aprimeindexes), size(aprimeindexes)), a_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec);
                RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, N_bothz, N_e]);

                if has_continuation
                    aprimez = aprimeindexes + n2aprime * bothzBind;
                    entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp(aprimez(:)), [N_d1*n2long, N_a, N_bothz, N_e]);
                else
                    entireRHS_L2 = RM_L2;
                end

                [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                V_ford2_jj(:,:,:,d2_c) = shiftdim(Vtemp, 1);
                maxidx_ford2_jj(:,:,:,d2_c) = shiftdim(maxindexL2, 1);

                d1_ind = rem(maxindexL2-1, N_d1) + 1;
                L2offset = ceil(maxindexL2 / N_d1);
                linidx_lower = d1_ind + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;
                linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind + N_d1*n2long*N_a*N_bothz*eind;

                isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                allind = d1_ind + N_d1*aind + N_d1*N_a*bothzind + N_d1*N_a*N_bothz*eind;
                midpt_ford2_jj(:,:,:,d2_c) = squeeze(midpoint(allind));
            end

            % LOW MEMORY 1
        elseif vfoptions.lowmemory == 1
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
                if has_continuation; EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz); EVinterp = interp1(a_grid, EV_d2, aprime_grid); end

                for e_c = 1:N_e
                    e_val = Grids.e_gridvals_J(e_c,:,j);
                    RM_L1 = call_RM_L1(Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec);
                    RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, N_bothz, 1]);

                    if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2, -1); else; entireRHS_L1 = RM_L1; end

                    [~, maxindex1] = max(entireRHS_L1, [], 2);
                    midpoint = max(min(maxindex1, N_a-1), 2);
                    aprimeindexes = (midpoint + (midpoint-1)*n2short) + (-n2short-1:1:1+n2short);

                    RM_L2 = call_RM_L2(Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, reshape(aprime_grid(aprimeindexes), size(aprimeindexes)), a_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec);
                    RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, N_bothz, 1]);

                    if has_continuation
                        aprimez = aprimeindexes + n2aprime * bothzBind;
                        entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp(aprimez(:)), [N_d1*n2long, N_a, N_bothz, 1]);
                    else
                        entireRHS_L2 = RM_L2;
                    end

                    [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                    V_ford2_jj(:,:,e_c,d2_c) = shiftdim(Vtemp, 1);
                    maxidx_ford2_jj(:,:,e_c,d2_c) = shiftdim(maxindexL2, 1);

                    d1_ind = rem(maxindexL2-1, N_d1) + 1;
                    L2offset = ceil(maxindexL2 / N_d1);
                    linidx_lower = d1_ind + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;
                    linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*bothzind;

                    isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford2_jj(:,:,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    allind = d1_ind + N_d1*aind + N_d1*N_a*bothzind;
                    midpt_ford2_jj(:,:,e_c,d2_c) = squeeze(midpoint(allind));
                end
            end

            % LOW MEMORY 2
        elseif vfoptions.lowmemory == 2
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
                for z_c = 1:N_z
                    semizblock = (z_c-1)*N_semiz + (1:N_semiz);
                    if has_continuation; EV_d2z = compute_expected_value_cond(EV, pi_bothz(semizblock,:)', N_a); EVinterp_z = interp1(a_grid, EV_d2z, aprime_grid); end

                    for e_c = 1:N_e
                        e_val = Grids.e_gridvals_J(e_c,:,j);
                        RM_L1 = call_RM_L1(Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec);
                        RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, N_semiz, 1]);

                        if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2z, -1); else; entireRHS_L1 = RM_L1; end

                        [~, maxindex1] = max(entireRHS_L1, [], 2);
                        midpoint = max(min(maxindex1, N_a-1), 2);
                        aprimeindexes = (midpoint + (midpoint-1)*n2short) + (-n2short-1:1:1+n2short);

                        RM_L2 = call_RM_L2(Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, reshape(aprime_grid(aprimeindexes), size(aprimeindexes)), a_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec);
                        RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, N_semiz, 1]);

                        if has_continuation
                            aprimez = aprimeindexes + n2aprime * semizBind;
                            entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp_z(aprimez(:)), [N_d1*n2long, N_a, N_semiz, 1]);
                        else
                            entireRHS_L2 = RM_L2;
                        end

                        [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                        V_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(Vtemp, 1);
                        maxidx_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(maxindexL2, 1);

                        d1_ind = rem(maxindexL2-1, N_d1) + 1;
                        L2offset = ceil(maxindexL2 / N_d1);
                        linidx_lower = d1_ind + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
                        linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;

                        isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                        inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                        flag_ford2_jj(:,semizblock,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                        allind = d1_ind + N_d1*aind + N_d1*N_a*semizind;
                        midpt_ford2_jj(:,semizblock,e_c,d2_c) = squeeze(midpoint(allind));
                    end
                end
            end

            % LOW MEMORY 3
        elseif vfoptions.lowmemory == 3
            for d2_c = 1:N_d2
                pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j));
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
                for z_c = 1:N_bothz
                    if has_continuation; EV_d2z = compute_expected_value_cond(EV, pi_bothz(z_c,:)', N_a); EVinterp_z = interp1(a_grid, EV_d2z, aprime_grid); end

                    for e_c = 1:N_e
                        e_val = Grids.e_gridvals_J(e_c,:,j);
                        RM_L1 = call_RM_L1(Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec);
                        RM_L1 = reshape(RM_L1, [N_d1, N_a, N_a, 1, 1]);

                        if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * shiftdim(EV_d2z, -1); else; entireRHS_L1 = RM_L1; end

                        [~, maxindex1] = max(entireRHS_L1, [], 2);
                        midpoint = max(min(maxindex1, N_a-1), 2);
                        aprimeindexes = (midpoint + (midpoint-1)*n2short) + (-n2short-1:1:1+n2short);

                        RM_L2 = call_RM_L2(Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, reshape(aprime_grid(aprimeindexes), size(aprimeindexes)), a_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec);
                        RM_L2 = reshape(RM_L2, [N_d1 * n2long, N_a, 1, 1]);

                        if has_continuation
                            entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp_z(aprimeindexes(:)), [N_d1*n2long, N_a, 1, 1]);
                        else
                            entireRHS_L2 = RM_L2;
                        end

                        [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                        V_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(Vtemp, 1);
                        maxidx_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(maxindexL2, 1);

                        d1_ind = rem(maxindexL2-1, N_d1) + 1;
                        L2offset = ceil(maxindexL2 / N_d1);
                        linidx_lower = d1_ind + N_d1*n2long*aind;
                        linidx_upper = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;

                        isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                        inLowerStrict = (L2offset >= 2) & (L2offset <= n2short+1); inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                        flag_ford2_jj(:,z_c,e_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                        allind = d1_ind + N_d1*aind;
                        midpt_ford2_jj(:,z_c,e_c,d2_c) = squeeze(midpoint(allind));
                    end
                end
            end
        end

        % Resolve max over d2
        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);

        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_GI_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1);
        else
            Pol_j = extract_policy_GI_no_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, N_d1);
        end
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function RM = call_RM_L1(n_d_in, n_bothz_in, n_e_in, d_val, a_val, z_val, e_val, ParamsVec)
        if Dims.N_e_raw == 0
            RM = CreateReturnFnMatrix_Disc(ReturnFn, n_d_in, n_a, n_bothz_in, d_val, a_val, z_val, ParamsVec, 1);
        else
            RM = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d_in, n_a, n_bothz_in, n_e_in, d_val, a_val, z_val, e_val, ParamsVec, 1);
        end
    end

    function RM = call_RM_L2(n_d_in, n_bothz_in, n_e_in, d_val, aprime_val, a_val, z_val, e_val, ParamsVec)
        if Dims.N_e_raw == 0
            RM = CreateReturnFnMatrix_Disc_DC1(ReturnFn, n_d_in, n_bothz_in, d_val, aprime_val, a_val, z_val, ParamsVec, 2);
        else
            RM = CreateReturnFnMatrix_Disc_DC1_e(ReturnFn, n_d_in, n_bothz_in, n_e_in, d_val, aprime_val, a_val, z_val, e_val, ParamsVec, 2);
        end
    end

    function d_slice = get_d_slice(d12_gridvals, d2_c, has_d2, has_d1)
        if has_d2 && has_d1; d_slice = d12_gridvals(:,:,d2_c); else; d_slice = d12_gridvals; end
    end

    function Pol4 = extract_policy_GI_no_d2(maxidx, midpoint, flag, N_d1)
        Pol4 = zeros(4, size(maxidx, 2), size(maxidx, 3), size(maxidx, 4), 'gpuArray');
        d_ind = shiftdim(rem(maxidx - 1, N_d1) + 1, -1);
        Pol4(1,:,:,:) = d_ind; % d
        Pol4(2,:,:,:) = shiftdim(midpoint, -1); % midpoint
        Pol4(3,:,:,:) = shiftdim(ceil(maxidx / N_d1), -1); % aprimeL2ind
        Pol4(4,:,:,:) = shiftdim(flag, -1); % flag
    end

    function Pol5 = extract_policy_GI_d2(maxidx_ford2, midpoint_ford2, flag_ford2, maxindex_d2, N_a, N_bothz, N_e, N_d1)
        Pol5 = zeros(5, N_a, N_bothz, N_e, 'gpuArray');
        Pol5(2,:,:,:) = shiftdim(maxindex_d2, -1); % d2
        linear_d2 = reshape(maxindex_d2, [N_a * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_d2 - 1);

        maxidx = reshape(maxidx_ford2(lin_idx), [1, N_a, N_bothz, N_e]);
        Pol5(1,:,:,:) = rem(maxidx - 1, N_d1) + 1; % d1
        Pol5(4,:,:,:) = ceil(maxidx / N_d1); % aprimeL2ind
        Pol5(3,:,:,:) = reshape(midpoint_ford2(lin_idx), [1, N_a, N_bothz, N_e]); % midpoint
        Pol5(5,:,:,:) = reshape(flag_ford2(lin_idx), [1, N_a, N_bothz, N_e]); % flag
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
