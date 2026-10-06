function [V, Policy] = ValueFnIter_FHorz_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
N_d = prod(n_d);
has_d = (N_d > 0);

N_a = prod(n_a);

N_z = prod(n_z);
if N_z == 0
    pi_z_J = ones(1, 1, N_j);
    z_gridvals_J = zeros(1, 1, N_j);
    Nz_eff = 1;
    if vfoptions.lowmemory > 0
        special_n_z = 0;
    end
else
    Nz_eff = N_z;
    if vfoptions.lowmemory > 0
        special_n_z = ones(1, length(n_z));
    end
end

V = zeros(N_a, Nz_eff, N_j, 'gpuArray');
Policy = zeros(1, N_a, Nz_eff, N_j, 'gpuArray');

%% j = N_j
ReturnFnParamsVec_J = CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions, 'V_Jplus1')
    if vfoptions.lowmemory == 0
        ReturnMatrix = CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec_J, 0);
        [Vtemp, maxindex] = max(ReturnMatrix, [], 1);
        V(:,:,N_j) = shiftdim(Vtemp, 1);
        Policy(1,:,:,N_j) = shiftdim(maxindex, 1);
    elseif vfoptions.lowmemory == 1
        for z_c = 1:Nz_eff
            z_val = z_gridvals_J(z_c, :, N_j);
            ReturnMatrix_z = CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec_J, 0);
            [Vtemp, maxindex] = max(ReturnMatrix_z, [], 1);
            V(:,z_c,N_j) = shiftdim(Vtemp, 1);
            Policy(1,:,z_c,N_j) = shiftdim(maxindex, 1);
        end
    end
else
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j);
    DiscountFactorParamsVec = prod(DiscountFactorParamsVec);

    EV = reshape(vfoptions.V_Jplus1, [N_a, Nz_eff]);
    EVinf = (EV == -Inf);
    EV(EVinf) = -1e250;
    EV = EV * pi_z_J(:,:,N_j)';
    EV(EVinf * (pi_z_J(:,:,N_j)' > 0) > 0) = -Inf;

    EV = reshape(EV, [N_a, 1, Nz_eff]);

    if vfoptions.lowmemory == 0
        entireEV = EV;
        if has_d; entireEV = repelem(entireEV, N_d, 1, 1); end

        ReturnMatrix = CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec_J, 0);
        entireRHS = ReturnMatrix + DiscountFactorParamsVec * entireEV;
        [Vtemp, maxindex] = max(entireRHS, [], 1);
        V(:,:,N_j) = shiftdim(Vtemp, 1);
        Policy(1,:,:,N_j) = shiftdim(maxindex, 1);
    elseif vfoptions.lowmemory == 1
        for z_c = 1:Nz_eff
            z_val = z_gridvals_J(z_c, :, N_j);
            entireEV_z = EV(:,:,z_c);
            if has_d; entireEV_z = repelem(entireEV_z, N_d, 1, 1); end
            ReturnMatrix_z = CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec_J, 0);
            entireRHS_z = ReturnMatrix_z + DiscountFactorParamsVec * entireEV_z;
            [Vtemp, maxindex] = max(entireRHS_z, [], 1);
            V(:,z_c,N_j) = shiftdim(Vtemp, 1);
            Policy(1,:,z_c,N_j) = shiftdim(maxindex, 1);
        end
    end
end

%% Iterate backwards through j.
for reverse_j = 1:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', jj, N_j)
    end

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    DiscountFactorParamsVec = prod(DiscountFactorParamsVec);

    EV = V(:,:,jj+1);
    EVinf = (EV == -Inf);
    EV(EVinf) = -1e250;
    EV = EV * pi_z_J(:,:,jj)';
    EV(EVinf * (pi_z_J(:,:,jj)' > 0) > 0) = -Inf;

    EV = reshape(EV, [N_a, 1, Nz_eff]);

    if vfoptions.lowmemory == 0
        entireEV = EV;
        if has_d; entireEV = repelem(entireEV, N_d, 1, 1); end

        ReturnMatrix = CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec, 0);
        entireRHS = ReturnMatrix + DiscountFactorParamsVec * entireEV;
        [Vtemp, maxindex] = max(entireRHS, [], 1);
        V(:,:,jj) = shiftdim(Vtemp, 1);
        Policy(1,:,:,jj) = shiftdim(maxindex, 1);
    elseif vfoptions.lowmemory == 1
        for z_c = 1:Nz_eff
            z_val = z_gridvals_J(z_c, :, jj);
            entireEV_z = EV(:,:,z_c);
            if has_d; entireEV_z = repelem(entireEV_z, N_d, 1, 1); end
            ReturnMatrix_z = CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec, 0);
            entireRHS_z = ReturnMatrix_z + DiscountFactorParamsVec * entireEV_z;
            [Vtemp, maxindex] = max(entireRHS_z, [], 1);
            V(:,z_c,jj) = shiftdim(Vtemp, 1);
            Policy(1,:,z_c,jj) = shiftdim(maxindex, 1);
        end
    end
end


end