function [V, Policy, Valt, Policyalt] = UnKronUFHorz(VKron, PolicyKron, ValtKron, PolicyaltKron, n_d, n_a_choice, n_a, n_z, N_j, vfoptions, is_gi)

[V, Policy] = unkron_single_local(VKron, PolicyKron, n_d, n_a_choice, n_a, n_z, N_j, vfoptions, is_gi);

if nargout > 2 && ~isempty(ValtKron)
    if nargout > 3 && ~isempty(PolicyaltKron)
        [Valt, Policyalt] = unkron_single_local(ValtKron, PolicyaltKron, n_d, n_a_choice, n_a, n_z, N_j, vfoptions, is_gi);
    else
        Valt = unkron_v_only_local(ValtKron, n_a, n_z, N_j, vfoptions);
        Policyalt = [];
    end
else
    Valt = [];
    Policyalt = [];
end

end

function V = unkron_v_only_local(VKron, n_a, n_z, N_j, vfoptions)
if vfoptions.outputkron == 1
    V = VKron;
    return;
end
state_dims = n_a;
if prod(vfoptions.n_semiz) > 0; state_dims = [state_dims, vfoptions.n_semiz]; end
if prod(n_z) > 0;               state_dims = [state_dims, n_z];               end
if prod(vfoptions.n_e) > 0;     state_dims = [state_dims, vfoptions.n_e];     end
V = reshape(VKron, [state_dims, N_j]);
end

function [V, Policy] = unkron_single_local(VKron, PolicyKron, n_d, n_a_choice, n_a, n_z, N_j, vfoptions, is_gi)
N_d = max(prod(n_d), 1);
l_d = length(n_d); if isempty(n_d) || n_d(1) == 0; l_d = 0; end
l_a_c = length(n_a_choice); if isempty(n_a_choice) || n_a_choice(1) == 0; l_a_c = 0; end

if is_gi
    n2short = vfoptions.ngridinterp;
    adjust = (PolicyKron(3,:,:,:,:) < 1 + n2short + 1);
    PolicyKron(2,:,:,:,:) = PolicyKron(2,:,:,:,:) - adjust;
    PolicyKron(3,:,:,:,:) = PolicyKron(3,:,:,:,:) - (n2short + 1) * (~adjust);

    if vfoptions.outputkron == 1
        if l_d == 0
            Policy = PolicyKron(2:4, :, :, :, :);
        else
            Policy = PolicyKron;
        end
        V = VKron;
        return;
    end

    s2 = size(PolicyKron, 2);
    s3 = size(PolicyKron, 3);
    s4 = size(PolicyKron, 4);
    s5 = size(PolicyKron, 5);
    n_rows = l_d + l_a_c + 2;
    Policy = zeros(n_rows, s2, s3, s4, s5, 'gpuArray');

    if l_d > 0
        d_idx = PolicyKron(1,:,:,:,:);
        d_cum = 1;
        for i = 1:l_d
            Policy(i,:,:,:,:) = rem(ceil(d_idx / d_cum) - 1, n_d(i)) + 1;
            d_cum = d_cum * n_d(i);
        end
    end

    if l_a_c > 0
        a_idx = PolicyKron(2,:,:,:,:);
        a_cum = 1;
        for i = 1:l_a_c
            Policy(l_d + i,:,:,:,:) = rem(ceil(a_idx / a_cum) - 1, n_a_choice(i)) + 1;
            a_cum = a_cum * n_a_choice(i);
        end
    end

    Policy(n_rows - 1,:,:,:,:) = PolicyKron(3,:,:,:,:);
    Policy(n_rows,:,:,:,:)     = PolicyKron(4,:,:,:,:);
end

state_dims = n_a;
if prod(vfoptions.n_semiz) > 0; state_dims = [state_dims, vfoptions.n_semiz]; end
if prod(n_z) > 0;               state_dims = [state_dims, n_z];               end
if prod(vfoptions.n_e) > 0;     state_dims = [state_dims, vfoptions.n_e];     end

V      = reshape(VKron,  [state_dims, N_j]);
Policy = reshape(Policy, [n_rows, state_dims, N_j]);
end
