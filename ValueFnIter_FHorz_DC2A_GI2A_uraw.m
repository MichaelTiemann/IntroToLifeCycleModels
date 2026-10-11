function [V, Policy] = ValueFnIter_FHorz_DC2A_GI2A_uraw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Universal Setup ---
[Dims, Grids, Probs] = Setup_uraw_FHorz(n_d, n_a, n_z, N_j, d_gridvals, z_gridvals_J, pi_z_J, vfoptions);

N_a = Dims.N_a; N_bothz = Dims.N_bothz; N_e = Dims.N_e; 
N_d = Dims.N_d; N_d1 = Dims.N_d1; N_d2 = Dims.N_d2; N_z = Dims.N_z; N_semiz = Dims.N_semiz;
has_d2 = Dims.has_d2; has_d1 = Dims.has_d1; N_semiz_raw = Dims.N_semiz_raw;
has_d = Dims.has_d1 || Dims.has_d2;

N_a1 = n_a(1); N_a2 = n_a(2:end);
a1_grid = a_grid(1:N_a1); a2_grid = a_grid(N_a1+1:end);

level1ii = round(linspace(1, N_a1, vfoptions.level1n));
level1iidiff = level1ii(2:end) - level1ii(1:end-1) - 1;

% Grid interpolation setup
n2short = vfoptions.ngridinterp;
n2long = n2short*2 + 3;
a1prime_grid = interp1(1:1:N_a1, a1_grid, linspace(1, N_a1, N_a1+(N_a1-1)*n2short))';
N_a1fine = length(a1prime_grid);

% Preallocate outputs
V = zeros(N_a, N_bothz, N_e, N_j, 'gpuArray');
if has_d2 && N_semiz_raw > 0
    Policy = zeros(6, N_a, N_bothz, N_e, N_j, 'gpuArray');
    Policy(6,:,:,:,:) = 2;
else
    Policy = zeros(5, N_a, N_bothz, N_e, N_j, 'gpuArray');
    Policy(5,:,:,:,:) = 2;
end

% Explicitly reshaped indexers to guarantee safe implicit expansion in L2 refinement
a12ind = reshape(gpuArray(0:N_a-1), [1, N_a1, N_a2]);
bothzind = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, N_bothz]);
semizind = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, N_semiz]);
eind = reshape(gpuArray(0:N_e-1), [1, 1, 1, 1, N_e]);

a2Bind = reshape(gpuArray(0:N_a2-1), [1, 1, N_a2]);
bothzBind = reshape(gpuArray(0:N_bothz-1), [1, 1, 1, 1, 1, N_bothz]);
semizBind = reshape(gpuArray(0:N_semiz-1), [1, 1, 1, 1, 1, N_semiz]);

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
    adjust = (Policy(5,:,:,:,:) < 1+n2short+1);
    Policy(3,:,:,:,:) = Policy(3,:,:,:,:) - adjust;
    Policy(5,:,:,:,:) = Policy(5,:,:,:,:) - (n2short+1).*(~adjust);
    if ~has_d; Policy = Policy(2:end, :, :, :, :); end
else
    adjust = (Policy(4,:,:,:,:) < 1+n2short+1);
    Policy(2,:,:,:,:) = Policy(2,:,:,:,:) - adjust;
    Policy(4,:,:,:,:) = Policy(4,:,:,:,:) - (n2short+1).*(~adjust);
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
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
                
                if has_continuation
                    if N_semiz_raw > 0; pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bothz = Probs.pi_z_J(:,:,j); end
                    EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz);
                    EV_reshaped = reshape(EV_d2, [N_a1, N_a2, 1, 1, N_bothz]);
                    EVinterp = interp1(a1_grid, EV_reshaped, a1prime_grid);
                end
                
                RM_L1 = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 1, 0);

                if has_continuation
                    DiscountedEV = DiscountFactorParamsVec * reshape(EV_d2, [1, N_a1, N_a2, 1, 1, N_bothz, 1]);
                    entireRHS_L1 = RM_L1 + DiscountedEV; 
                else
                    entireRHS_L1 = RM_L1; 
                end

                [~, maxindex1] = max(entireRHS_L1, [], 2);
                midpoints_jj = zeros(N_d1, 1, N_a2, N_a1, N_a2, N_bothz, N_e, 'gpuArray');
                midpoints_jj(:,1,:,level1ii,:,:,:) = maxindex1;

                maxgap = squeeze(max(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],7),[],6),[],5),[],3),[],1));
                
                for ii = 1:(vfoptions.level1n-1)
                    curraindex = level1ii(ii)+1:level1ii(ii+1)-1;
                    if maxgap(ii) > 0
                        loweredge = min(maxindex1(:,1,:,ii,:,:,:), N_a1-maxgap(ii));
                        a1primeindexes = loweredge + (0:1:maxgap(ii));
                        
                        RM_ref = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 3, 0);
                        
                        if has_continuation
                            aprimez = a1primeindexes + N_a1*a2Bind + N_a*bothzBind;
                            entireRHS_ref = RM_ref + DiscountFactorParamsVec * reshape(EV_d2(aprimez(:)), [N_d1, maxgap(ii)+1, N_a2, 1, N_a2, N_bothz, N_e]);
                        else
                            entireRHS_ref = RM_ref;
                        end

                        [~, maxindex] = max(entireRHS_ref, [], 2);
                        midpoints_jj(:,1,:,curraindex,:,:,:) = maxindex + (loweredge - 1);
                    else
                        loweredge = maxindex1(:,1,:,ii,:,:,:);
                        midpoints_jj(:,1,:,curraindex,:,:,:) = repelem(loweredge, 1, 1, 1, length(curraindex), 1, 1, 1);
                    end
                end

                midpoints_jj = max(min(midpoints_jj, N_a1-1), 2);
                a1primeindexes = (midpoints_jj + (midpoints_jj-1)*n2short) + (-n2short-1:1:1+n2short);
                
                RM_L2 = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.n_e, d_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, Grids.bothz_gridvals_J(:,:,j), Grids.e_gridvals_J(:,:,j), ReturnFnParamsVec, 2, 0);

                if has_continuation
                    aprimez = a1primeindexes + N_a1fine * a2Bind + N_a1fine * N_a2 * bothzBind;
                    entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp(aprimez(:)), [N_d1*n2long*N_a2, N_a1, N_a2, N_bothz, N_e]);
                else
                    entireRHS_L2 = RM_L2;
                end

                [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                
                maxindexL2d = rem(maxindexL2-1, N_d1) + 1;
                maxindexL2a = ceil(maxindexL2 / N_d1);
                maxindexL2a1 = rem(maxindexL2a-1, n2long) + 1;
                maxindexL2a2 = ceil(maxindexL2a / n2long);

                V_ford2_jj(:,:,:,d2_c) = reshape(Vtemp, [N_a, N_bothz, N_e]);
                maxidx_ford2_jj(:,:,:,d2_c) = reshape(maxindexL2, [N_a, N_bothz, N_e]);

                linidx_lower = maxindexL2d + N_d1*n2long*(maxindexL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*bothzind + N_d1*n2long*N_a2*N_a*N_bothz*eind;
                linidx_upper = maxindexL2d + N_d1*(n2long-1) + N_d1*n2long*(maxindexL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*bothzind + N_d1*n2long*N_a2*N_a*N_bothz*eind;
                
                isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2) & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                
                flag_ford2_jj(:,:,:,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), [N_a, N_bothz, N_e]);
                
                allind = maxindexL2d + N_d1*(maxindexL2a2-1) + N_d1*N_a2*a12ind + N_d1*N_a2*N_a*bothzind + N_d1*N_a2*N_a*N_bothz*eind;
                midpt_ford2_jj(:,:,:,d2_c) = reshape(midpoints_jj(allind), [N_a, N_bothz, N_e]);
            end

        % LOW MEMORY 1
        elseif vfoptions.lowmemory == 1
            for d2_c = 1:N_d2
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
                if has_continuation
                    if N_semiz_raw > 0; pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bothz = Probs.pi_z_J(:,:,j); end
                    EV_d2 = compute_expected_value(EV, pi_bothz, N_a, N_bothz); 
                    EV_reshaped = reshape(EV_d2, [N_a1, N_a2, 1, 1, N_bothz]);
                    EVinterp = interp1(a1_grid, EV_reshaped, a1prime_grid); 
                end
                
                for e_c = 1:N_e
                    e_val = Grids.e_gridvals_J(e_c,:,j);
                    RM_L1 = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 1, 0);

                    if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * reshape(EV_d2, [1, N_a1, N_a2, 1, 1, N_bothz]); else; entireRHS_L1 = RM_L1; end

                    [~, maxindex1] = max(entireRHS_L1, [], 2);
                    midpoints_jj = zeros(N_d1, 1, N_a2, N_a1, N_a2, N_bothz, 1, 'gpuArray');
                    midpoints_jj(:,1,:,level1ii,:,:,:) = maxindex1;

                    maxgap = squeeze(max(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],6),[],5),[],3),[],1));
                    
                    for ii = 1:(vfoptions.level1n-1)
                        curraindex = level1ii(ii)+1:level1ii(ii+1)-1;
                        if maxgap(ii) > 0
                            loweredge = min(maxindex1(:,1,:,ii,:,:,:), N_a1-maxgap(ii));
                            a1primeindexes = loweredge + (0:1:maxgap(ii));
                            
                            RM_ref = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 3, 0);
                            
                            if has_continuation
                                aprimez = a1primeindexes + N_a1*a2Bind + N_a*bothzBind;
                                entireRHS_ref = RM_ref + DiscountFactorParamsVec * reshape(EV_d2(aprimez(:)), [N_d1, maxgap(ii)+1, N_a2, 1, N_a2, N_bothz, 1]);
                            else
                                entireRHS_ref = RM_ref;
                            end

                            [~, maxindex] = max(entireRHS_ref, [], 2);
                            midpoints_jj(:,1,:,curraindex,:,:,:) = maxindex + (loweredge - 1);
                        else
                            loweredge = maxindex1(:,1,:,ii,:,:,:);
                            midpoints_jj(:,1,:,curraindex,:,:,:) = repelem(loweredge, 1, 1, 1, length(curraindex), 1, 1);
                        end
                    end

                    midpoints_jj = max(min(midpoints_jj, N_a1-1), 2);
                    a1primeindexes = (midpoints_jj + (midpoints_jj-1)*n2short) + (-n2short-1:1:1+n2short);
                    
                    RM_L2 = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.n_bothz, Dims.special_n_e, d_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, Grids.bothz_gridvals_J(:,:,j), e_val, ReturnFnParamsVec, 2, 0);

                    if has_continuation
                        aprimez = a1primeindexes + N_a1fine * a2Bind + N_a1fine * N_a2 * bothzBind;
                        entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp(aprimez(:)), [N_d1*n2long*N_a2, N_a1, N_a2, N_bothz, 1]);
                    else
                        entireRHS_L2 = RM_L2;
                    end

                    [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                    
                    maxindexL2d = rem(maxindexL2-1, N_d1) + 1;
                    maxindexL2a = ceil(maxindexL2 / N_d1);
                    maxindexL2a1 = rem(maxindexL2a-1, n2long) + 1;
                    maxindexL2a2 = ceil(maxindexL2a / n2long);

                    V_ford2_jj(:,:,e_c,d2_c) = reshape(Vtemp, [N_a, N_bothz]);
                    maxidx_ford2_jj(:,:,e_c,d2_c) = reshape(maxindexL2, [N_a, N_bothz]);

                    linidx_lower = maxindexL2d + N_d1*n2long*(maxindexL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*bothzind;
                    linidx_upper = maxindexL2d + N_d1*(n2long-1) + N_d1*n2long*(maxindexL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*bothzind;
                    
                    isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                    inLowerStrict = (maxindexL2a1 >= 2) & (maxindexL2a1 <= n2short+1);
                    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                    
                    flag_ford2_jj(:,:,e_c,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), [N_a, N_bothz]);
                    
                    allind = maxindexL2d + N_d1*(maxindexL2a2-1) + N_d1*N_a2*a12ind + N_d1*N_a2*N_a*bothzind;
                    midpt_ford2_jj(:,:,e_c,d2_c) = reshape(midpoints_jj(allind), [N_a, N_bothz]);
                end
            end

        % LOW MEMORY 2
        elseif vfoptions.lowmemory == 2
            for d2_c = 1:N_d2
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
                if has_continuation
                    if N_semiz_raw > 0; pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bothz = Probs.pi_z_J(:,:,j); end
                end
                for z_c = 1:N_z
                    semizblock = (z_c-1)*N_semiz + (1:N_semiz);
                    if has_continuation
                        EV_d2z = compute_expected_value_cond(EV, pi_bothz(semizblock,:)', N_a); 
                        EV_reshaped = reshape(EV_d2z, [N_a1, N_a2, 1, 1, N_semiz]);
                        EVinterp_z = interp1(a1_grid, EV_reshaped, a1prime_grid); 
                    end
                    
                    for e_c = 1:N_e
                        e_val = Grids.e_gridvals_J(e_c,:,j);
                        RM_L1 = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 1, 0);

                        if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * reshape(EV_d2z, [1, N_a1, N_a2, 1, 1, N_semiz]); else; entireRHS_L1 = RM_L1; end

                        [~, maxindex1] = max(entireRHS_L1, [], 2);
                        midpoints_jj = zeros(N_d1, 1, N_a2, N_a1, N_a2, N_semiz, 1, 'gpuArray');
                        midpoints_jj(:,1,:,level1ii,:,:,:) = maxindex1;

                        maxgap = squeeze(max(max(max(maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:),[],5),[],3),[],1));
                        
                        for ii = 1:(vfoptions.level1n-1)
                            curraindex = level1ii(ii)+1:level1ii(ii+1)-1;
                            if maxgap(ii) > 0
                                loweredge = min(maxindex1(:,1,:,ii,:,:,:), N_a1-maxgap(ii));
                                a1primeindexes = loweredge + (0:1:maxgap(ii));
                                
                                RM_ref = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 3, 0);
                                
                                if has_continuation
                                    aprimez = a1primeindexes + N_a1*a2Bind + N_a*semizBind;
                                    entireRHS_ref = RM_ref + DiscountFactorParamsVec * reshape(EV_d2z(aprimez(:)), [N_d1, maxgap(ii)+1, N_a2, 1, N_a2, N_semiz, 1]);
                                else
                                    entireRHS_ref = RM_ref;
                                end

                                [~, maxindex] = max(entireRHS_ref, [], 2);
                                midpoints_jj(:,1,:,curraindex,:,:,:) = maxindex + (loweredge - 1);
                            else
                                loweredge = maxindex1(:,1,:,ii,:,:,:);
                                midpoints_jj(:,1,:,curraindex,:,:,:) = repelem(loweredge, 1, 1, 1, length(curraindex), 1, 1);
                            end
                        end

                        midpoints_jj = max(min(midpoints_jj, N_a1-1), 2);
                        a1primeindexes = (midpoints_jj + (midpoints_jj-1)*n2short) + (-n2short-1:1:1+n2short);
                        
                        RM_L2 = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, [Dims.n_semiz, Dims.special_n_z], Dims.special_n_e, d_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, Grids.bothz_gridvals_J(semizblock,:,j), e_val, ReturnFnParamsVec, 2, 0);

                        if has_continuation
                            aprimez = a1primeindexes + N_a1fine * a2Bind + N_a1fine * N_a2 * semizBind;
                            entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp_z(aprimez(:)), [N_d1*n2long*N_a2, N_a1, N_a2, N_semiz, 1]);
                        else
                            entireRHS_L2 = RM_L2;
                        end

                        [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                        
                        maxindexL2d = rem(maxindexL2-1, N_d1) + 1;
                        maxindexL2a = ceil(maxindexL2 / N_d1);
                        maxindexL2a1 = rem(maxindexL2a-1, n2long) + 1;
                        maxindexL2a2 = ceil(maxindexL2a / n2long);

                        V_ford2_jj(:,semizblock,e_c,d2_c) = reshape(Vtemp, [N_a, N_semiz]);
                        maxidx_ford2_jj(:,semizblock,e_c,d2_c) = reshape(maxindexL2, [N_a, N_semiz]);

                        linidx_lower = maxindexL2d + N_d1*n2long*(maxindexL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind;
                        linidx_upper = maxindexL2d + N_d1*(n2long-1) + N_d1*n2long*(maxindexL2a2-1) + N_d1*n2long*N_a2*a12ind + N_d1*n2long*N_a2*N_a*semizind;
                        
                        isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                        inLowerStrict = (maxindexL2a1 >= 2) & (maxindexL2a1 <= n2short+1);
                        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                        
                        flag_ford2_jj(:,semizblock,e_c,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), [N_a, N_semiz]);
                        
                        allind = maxindexL2d + N_d1*(maxindexL2a2-1) + N_d1*N_a2*a12ind + N_d1*N_a2*N_a*semizind;
                        midpt_ford2_jj(:,semizblock,e_c,d2_c) = reshape(midpoints_jj(allind), [N_a, N_semiz]);
                    end
                end
            end
            
        % LOW MEMORY 3
        elseif vfoptions.lowmemory == 3
            for d2_c = 1:N_d2
                d_val = get_d_slice(Grids.d12_gridvals, d2_c, has_d2, has_d1);
                if has_continuation
                    if N_semiz_raw > 0; pi_bothz = kron(Probs.pi_z_J(:,:,j), Probs.pi_semiz_J(:,:,d2_c,j)); else; pi_bothz = Probs.pi_z_J(:,:,j); end
                end
                for z_c = 1:N_bothz
                    if has_continuation
                        EV_d2z = compute_expected_value_cond(EV, pi_bothz(z_c,:)', N_a); 
                        EV_reshaped = reshape(EV_d2z, [N_a1, N_a2]);
                        EVinterp_z = interp1(a1_grid, EV_reshaped, a1prime_grid); 
                    end
                    
                    for e_c = 1:N_e
                        e_val = Grids.e_gridvals_J(e_c,:,j);
                        RM_L1 = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 1, 0);

                        if has_continuation; entireRHS_L1 = RM_L1 + DiscountFactorParamsVec * reshape(EV_d2z, [1, N_a1, N_a2, 1, 1]); else; entireRHS_L1 = RM_L1; end

                        [~, maxindex1] = max(entireRHS_L1, [], 2);
                        midpoints_jj = zeros(N_d1, 1, N_a2, N_a1, N_a2, 1, 1, 'gpuArray');
                        midpoints_jj(:,1,:,level1ii,:,1,1) = maxindex1;

                        maxgap = squeeze(max(max(max(maxindex1(:,1,:,2:end,:,1,1)-maxindex1(:,1,:,1:end-1,:,1,1),[],5),[],3),[],1));
                        
                        for ii = 1:(vfoptions.level1n-1)
                            curraindex = level1ii(ii)+1:level1ii(ii+1)-1;
                            if maxgap(ii) > 0
                                loweredge = min(maxindex1(:,1,:,ii,:,1,1), N_a1-maxgap(ii));
                                a1primeindexes = loweredge + (0:1:maxgap(ii));
                                
                                RM_ref = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1_grid(a1primeindexes), a2_grid, a1_grid(curraindex), a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 3, 0);
                                
                                if has_continuation
                                    aprimez = a1primeindexes + N_a1*a2Bind;
                                    entireRHS_ref = RM_ref + DiscountFactorParamsVec * reshape(EV_d2z(aprimez(:)), [N_d1, maxgap(ii)+1, N_a2, 1, N_a2, 1, 1]);
                                else
                                    entireRHS_ref = RM_ref;
                                end

                                [~, maxindex] = max(entireRHS_ref, [], 2);
                                midpoints_jj(:,1,:,curraindex,:,1,1) = maxindex + (loweredge - 1);
                            else
                                loweredge = maxindex1(:,1,:,ii,:,1,1);
                                midpoints_jj(:,1,:,curraindex,:,1,1) = repelem(loweredge, 1, 1, 1, length(curraindex), 1, 1, 1);
                            end
                        end

                        midpoints_jj = max(min(midpoints_jj, N_a1-1), 2);
                        a1primeindexes = (midpoints_jj + (midpoints_jj-1)*n2short) + (-n2short-1:1:1+n2short);
                        
                        RM_L2 = CreateReturnFnMatrix_Disc_DC2A_uraw(ReturnFn, Dims.special_n_d, Dims.special_n_bothz, Dims.special_n_e, d_val, a1prime_grid(a1primeindexes), a2_grid, a1_grid, a2_grid, Grids.bothz_gridvals_J(z_c,:,j), e_val, ReturnFnParamsVec, 2, 0);

                        if has_continuation
                            aprimez = a1primeindexes + N_a1fine * a2Bind;
                            entireRHS_L2 = RM_L2 + DiscountFactorParamsVec * reshape(EVinterp_z(aprimez(:)), [N_d1*n2long*N_a2, N_a1, N_a2, 1, 1]);
                        else
                            entireRHS_L2 = RM_L2;
                        end

                        [Vtemp, maxindexL2] = max(entireRHS_L2, [], 1);
                        
                        maxindexL2d = rem(maxindexL2-1, N_d1) + 1;
                        maxindexL2a = ceil(maxindexL2 / N_d1);
                        maxindexL2a1 = rem(maxindexL2a-1, n2long) + 1;
                        maxindexL2a2 = ceil(maxindexL2a / n2long);

                        V_ford2_jj(:,z_c,e_c,d2_c) = reshape(Vtemp, [N_a, 1]);
                        maxidx_ford2_jj(:,z_c,e_c,d2_c) = reshape(maxindexL2, [N_a, 1]);

                        linidx_lower = maxindexL2d + N_d1*n2long*(maxindexL2a2-1) + N_d1*n2long*N_a2*a12ind;
                        linidx_upper = maxindexL2d + N_d1*(n2long-1) + N_d1*n2long*(maxindexL2a2-1) + N_d1*n2long*N_a2*a12ind;
                        
                        isInfLower = (RM_L2(linidx_lower) == -Inf); isInfUpper = (RM_L2(linidx_upper) == -Inf);
                        inLowerStrict = (maxindexL2a1 >= 2) & (maxindexL2a1 <= n2short+1);
                        inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                        
                        flag_ford2_jj(:,z_c,e_c,d2_c) = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), [N_a, 1]);
                        
                        allind = maxindexL2d + N_d1*(maxindexL2a2-1) + N_d1*N_a2*a12ind;
                        midpt_ford2_jj(:,z_c,e_c,d2_c) = reshape(midpoints_jj(allind), [N_a, 1]);
                    end
                end
            end
        end

        % Resolve max over d2
        [V_j, maxindex_d2] = max(V_ford2_jj, [], 4);
        
        if has_d2 && N_semiz_raw > 0
            Pol_j = extract_policy_GI2A_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, maxindex_d2, N_a, N_bothz, N_e, N_d1, n2long);
        else
            Pol_j = extract_policy_GI2A_no_d2(maxidx_ford2_jj, midpt_ford2_jj, flag_ford2_jj, N_a, N_bothz, N_e, N_d1, n2long);
        end
    end

% =====================================================================
% SUB-HELPERS
% =====================================================================
    function d_slice = get_d_slice(d12_gridvals, d2_c, has_d2, has_d1)
        if has_d2 && has_d1; d_slice = d12_gridvals(:,:,d2_c); else; d_slice = d12_gridvals; end
    end

    function Pol5 = extract_policy_GI2A_no_d2(maxidx, midpoint, flag, N_a, N_bothz, N_e, N_d1, n2long)
        Pol5 = zeros(5, N_a, N_bothz, N_e, 'gpuArray');
        Pol5(1,:,:,:) = reshape(rem(maxidx(:,:,:,1) - 1, N_d1) + 1, [1, N_a, N_bothz, N_e]); % d
        maxidxL2a = ceil(maxidx(:,:,:,1) / N_d1);
        Pol5(4,:,:,:) = reshape(rem(maxidxL2a - 1, n2long) + 1, [1, N_a, N_bothz, N_e]); % a1primeL2ind
        Pol5(3,:,:,:) = reshape(ceil(maxidxL2a / n2long), [1, N_a, N_bothz, N_e]); % a2prime
        Pol5(2,:,:,:) = reshape(midpoint(:,:,:,1), [1, N_a, N_bothz, N_e]); % midpoint
        Pol5(5,:,:,:) = reshape(flag(:,:,:,1), [1, N_a, N_bothz, N_e]); % flag
    end

    function Pol6 = extract_policy_GI2A_d2(maxidx_ford2, midpoint_ford2, flag_ford2, maxindex_d2, N_a, N_bothz, N_e, N_d1, n2long)
        Pol6 = zeros(6, N_a, N_bothz, N_e, 'gpuArray');
        Pol6(2,:,:,:) = reshape(maxindex_d2, [1, N_a, N_bothz, N_e]); % d2
        linear_d2 = reshape(maxindex_d2, [N_a * N_bothz * N_e, 1]);
        lin_idx = (1:(N_a * N_bothz * N_e))' + (N_a * N_bothz * N_e) * (linear_d2 - 1);
        
        maxidx = reshape(maxidx_ford2(lin_idx), [1, N_a, N_bothz, N_e]);
        Pol6(1,:,:,:) = reshape(rem(maxidx - 1, N_d1) + 1, [1, N_a, N_bothz, N_e]); % d1
        
        maxidxL2a = ceil(maxidx / N_d1);
        Pol6(5,:,:,:) = reshape(rem(maxidxL2a - 1, n2long) + 1, [1, N_a, N_bothz, N_e]); % a1primeL2ind
        Pol6(4,:,:,:) = reshape(ceil(maxidxL2a / n2long), [1, N_a, N_bothz, N_e]); % a2prime
        
        Pol6(3,:,:,:) = reshape(midpoint_ford2(lin_idx), [1, N_a, N_bothz, N_e]); % midpoint
        Pol6(6,:,:,:) = reshape(flag_ford2(lin_idx), [1, N_a, N_bothz, N_e]); % flag
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
