function [V, Policy] = ValueFnIter_FHorz_e_raw(n_d, n_a, n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);
N_e = prod(n_e);

has_d = (N_d > 0);
has_z = (N_z > 0);
Nz_eff = max(N_z, 1);

V = zeros(N_a, Nz_eff, N_e, N_j, 'gpuArray');
Policy = zeros(1, N_a, Nz_eff, N_e, N_j, 'gpuArray');

if vfoptions.lowmemory > 0
    special_n_e = ones(1, length(n_e));
end
if vfoptions.lowmemory > 1 && has_z
    special_n_z = ones(1, length(n_z));
end

pi_e_J = shiftdim(pi_e_J, -2);

%% j = N_j
ReturnFnParamsVec_J = CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

z_gridvals_N_j = [];
if has_z; z_gridvals_N_j = z_gridvals_J(:,:,N_j); end

if ~isfield(vfoptions, 'V_Jplus1')
    if vfoptions.lowmemory == 0
        ReturnMatrix = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals_N_j, e_gridvals_J(:,:,N_j), ReturnFnParamsVec_J, 0);
        if ~has_z
            sz = size(ReturnMatrix);
            if length(sz) == 2; ReturnMatrix = reshape(ReturnMatrix, [sz(1), sz(2), 1, 1]); else; ReturnMatrix = reshape(ReturnMatrix, [sz(1), sz(2), 1, sz(3:end)]); end
        end
        [Vtemp, maxindex] = max(ReturnMatrix, [], 1);
        V(:,:,:,N_j) = shiftdim(Vtemp, 1);
        Policy(1,:,:,:,N_j) = shiftdim(maxindex, 1);
    elseif vfoptions.lowmemory == 1
        for e_c = 1:N_e
            e_val = e_gridvals_J(e_c, :, N_j);
            ReturnMatrix_e = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals_N_j, e_val, ReturnFnParamsVec_J, 0);
            if ~has_z
                sz = size(ReturnMatrix_e);
                if length(sz) == 2; ReturnMatrix_e = reshape(ReturnMatrix_e, [sz(1), sz(2), 1, 1]); else; ReturnMatrix_e = reshape(ReturnMatrix_e, [sz(1), sz(2), 1, sz(3:end)]); end
            end
            [Vtemp, maxindex] = max(ReturnMatrix_e, [], 1);
            V(:,:,e_c,N_j) = shiftdim(Vtemp, 1);
            Policy(1,:,:,e_c,N_j) = shiftdim(maxindex, 1);
        end
    elseif vfoptions.lowmemory == 2 && has_z
        for z_c = 1:N_z
            z_val = z_gridvals_J(z_c, :, N_j);
            for e_c = 1:N_e
                e_val = e_gridvals_J(e_c, :, N_j);
                ReturnMatrix_ze = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec_J, 0);
                [Vtemp, maxindex] = max(ReturnMatrix_ze, [], 1);
                V(:,z_c,e_c,N_j) = shiftdim(Vtemp, 1);
                Policy(1,:,z_c,e_c,N_j) = shiftdim(maxindex, 1);
            end
        end
    end
else
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j);
    DiscountFactorParamsVec = prod(DiscountFactorParamsVec);

    EV = reshape(vfoptions.V_Jplus1, [N_a, Nz_eff, N_e]);
    EV = sum(EV .* pi_e_J(1, 1, :, N_j+1), 3);

    if has_z
        EVinf = (EV == -Inf);
        EV(EVinf) = -1e250;
        EV = EV * pi_z_J(:,:,N_j)';
        EV(EVinf * (pi_z_J(:,:,N_j)' > 0) > 0) = -Inf;
    end

    EV = reshape(EV, [N_a, 1, Nz_eff]);

    if vfoptions.lowmemory == 0
        if has_d; entireEV = repelem(entireEV, N_d, 1); else; entireEV = EV; end
        ReturnMatrix = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals_N_j, e_gridvals_J(:,:,N_j), ReturnFnParamsVec_J, 0);
        if ~has_z
            sz = size(ReturnMatrix);
            if length(sz) == 2; ReturnMatrix = reshape(ReturnMatrix, [sz(1), sz(2), 1, 1]); else; ReturnMatrix = reshape(ReturnMatrix, [sz(1), sz(2), 1, sz(3:end)]); end
        end
        entireRHS = ReturnMatrix + DiscountFactorParamsVec*entireEV;
        [Vtemp, maxindex] = max(entireRHS, [], 1);
        V(:,:,:,N_j) = shiftdim(Vtemp, 1);
        Policy(1,:,:,:,N_j) = shiftdim(maxindex, 1);
    elseif vfoptions.lowmemory == 1
        for e_c = 1:N_e
            e_val = e_gridvals_J(e_c, :, N_j);
            ReturnMatrix_e = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals_N_j, e_val, ReturnFnParamsVec_J, 0);
            if ~has_z
                sz = size(ReturnMatrix_e);
                if length(sz) == 2; ReturnMatrix_e = reshape(ReturnMatrix_e, [sz(1), sz(2), 1, 1]); else; ReturnMatrix_e = reshape(ReturnMatrix_e, [sz(1), sz(2), 1, sz(3:end)]); end
            end
            entireRHS = ReturnMatrix_e + DiscountFactorParamsVec*entireEV;
            [Vtemp, maxindex] = max(entireRHS, [], 1);
            V(:,:,e_c,N_j) = shiftdim(Vtemp, 1);
            Policy(1,:,:,e_c,N_j) = shiftdim(maxindex, 1);
        end
    elseif vfoptions.lowmemory == 2 && has_z
        for z_c = 1:N_z
            entireEV_z = EV(:,:,z_c);
            z_val = z_gridvals_J(z_c, :, N_j);
            for e_c = 1:N_e
                e_val = e_gridvals_J(e_c, :, N_j);
                ReturnMatrix_ze = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec_J, 0);
                entireRHS_ze = ReturnMatrix_ze + DiscountFactorParamsVec*entireEV_z;
                [Vtemp, maxindex] = max(entireRHS_ze, [], 1);
                V(:,z_c,e_c,N_j) = shiftdim(Vtemp, 1);
                Policy(1,:,z_c,e_c,N_j) = shiftdim(maxindex, 1);
            end
        end
    end
end

%% Iterate backwards through j
for reverse_j = 1:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose == 1
        fprintf('Finite horizon: %i of %i \n', jj, N_j)
    end

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    DiscountFactorParamsVec = prod(DiscountFactorParamsVec);

    EV = V(:,:,:,jj+1);
    EV = sum(EV .* pi_e_J(1, 1, :, jj+1), 3);

    if has_z
        EVinf = (EV == -Inf);
        EV(EVinf) = -1e250;
        EV = EV * pi_z_J(:,:,jj)';
        EV(EVinf * (pi_z_J(:,:,jj)' > 0) > 0) = -Inf;
    end

    EV = reshape(EV, [N_a, 1, Nz_eff]);
    if has_d; entireEV = repelem(EV, N_d, 1, 1); else; entireEV = EV; end

    z_gridvals_jj = [];
    if has_z; z_gridvals_jj = z_gridvals_J(:,:,jj); end

    if vfoptions.lowmemory == 0
        ReturnMatrix = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals_jj, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 0);
        if ~has_z
            sz = size(ReturnMatrix);
            if length(sz) == 2; ReturnMatrix = reshape(ReturnMatrix, [sz(1), sz(2), 1, 1]); else; ReturnMatrix = reshape(ReturnMatrix, [sz(1), sz(2), 1, sz(3:end)]); end
        end
        entireRHS = ReturnMatrix + DiscountFactorParamsVec*entireEV;
        [Vtemp, maxindex] = max(entireRHS, [], 1);
        V(:,:,:,jj) = shiftdim(Vtemp, 1);
        Policy(1,:,:,:,jj) = shiftdim(maxindex, 1);
    elseif vfoptions.lowmemory == 1
        for e_c = 1:N_e
            e_val = e_gridvals_J(e_c, :, jj);
            ReturnMatrix_e = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals_jj, e_val, ReturnFnParamsVec, 0);
            if ~has_z
                sz = size(ReturnMatrix_e);
                if length(sz) == 2; ReturnMatrix_e = reshape(ReturnMatrix_e, [sz(1), sz(2), 1, 1]); else; ReturnMatrix_e = reshape(ReturnMatrix_e, [sz(1), sz(2), 1, sz(3:end)]); end
            end
            entireRHS = ReturnMatrix_e + DiscountFactorParamsVec*entireEV;
            [Vtemp, maxindex] = max(entireRHS, [], 1);
            V(:,:,e_c,jj) = shiftdim(Vtemp, 1);
            Policy(1,:,:,e_c,jj) = shiftdim(maxindex, 1);
        end
    elseif vfoptions.lowmemory == 2 && has_z
        for z_c = 1:N_z
            entireEV_z = entireEV(:,:,z_c);
            z_val = z_gridvals_J(z_c, :, jj);
            for e_c = 1:N_e
                e_val = e_gridvals_J(e_c, :, jj);
                ReturnMatrix_ze = CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec, 0);
                entireRHS_ze = ReturnMatrix_ze + DiscountFactorParamsVec*entireEV_z;
                [Vtemp, maxindex] = max(entireRHS_ze, [], 1);
                V(:,z_c,e_c,jj) = shiftdim(Vtemp, 1);
                Policy(1,:,z_c,e_c,jj) = shiftdim(maxindex, 1);
            end
        end
    end
end

if ~has_z
    V = reshape(V, [N_a, N_e, N_j]);
    Policy = reshape(Policy, [1, N_a, N_e, N_j]);
end


end