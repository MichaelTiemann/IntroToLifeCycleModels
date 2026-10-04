function [V, Policy] = ValueFnIter_FHorz_noz_raw(n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
N_d = prod(n_d);
N_a = prod(n_a);
has_d = (N_d > 0);

V = zeros(N_a, N_j, 'gpuArray');
Policy = zeros(1, N_a, N_j, 'gpuArray');

%% j = N_j
ReturnFnParamsVec_J = CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions, 'V_Jplus1')
    ReturnMatrix = CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec_J, 0);
    [Vtemp, maxindex] = max(ReturnMatrix, [], 1);
    V(:, N_j) = shiftdim(Vtemp, 1);
    Policy(1, :, N_j) = shiftdim(maxindex, 1);
else
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j);
    DiscountFactorParamsVec = prod(DiscountFactorParamsVec);
    EV = reshape(vfoptions.V_Jplus1, [N_a, 1]);

    entireEV = EV;
    if has_d
        entireEV = repelem(EV, N_d, 1);
    end

    ReturnMatrix = CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec_J, 0);
    entireRHS = ReturnMatrix + DiscountFactorParamsVec * entireEV;

    [Vtemp, maxindex] = max(entireRHS, [], 1);
    V(:, N_j) = shiftdim(Vtemp, 1);
    Policy(1, :, N_j) = shiftdim(maxindex, 1);
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

    EV = V(:, jj+1);

    entireEV = EV;
    if has_d
        entireEV = repelem(EV, N_d, 1);
    end

    ReturnMatrix = CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec, 0);
    entireRHS = ReturnMatrix + DiscountFactorParamsVec * entireEV;

    [Vtemp, maxindex] = max(entireRHS, [], 1);
    V(:, jj) = shiftdim(Vtemp, 1);
    Policy(1, :, jj) = shiftdim(maxindex, 1);
end


end