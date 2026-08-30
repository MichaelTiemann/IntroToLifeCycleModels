function F = arrayfun_expand(ReturnFn, varargin)

% Parse inputs and find sizes
n_inputs = length(varargin);
input_sizes = cellfun(@size, varargin, 'UniformOutput', false);

% Determine maximum dimension length across all inputs for broadcasting
max_nd = max(cellfun(@ndims, varargin));
full_sz = ones(1, max_nd);

last_arr_idx=0;
param_idx=0;
for i = 1:n_inputs
    sz = input_sizes{i};
    if length(sz)==2 && all(sz==[1,1])
        % We are in the scalars, aka Parameters (not state space) for ReturnFn
        last_arr_idx=i-1;
        param_idx=i;
        break
    end
    sz(end+1:max_nd) = 1; % pad with singleton dimensions
    for d = 1:max_nd
        if full_sz(d) == 1
            full_sz(d) = sz(d);
        elseif sz(d) ~= 1 && sz(d) ~= full_sz(d)
            error('arrayfun: non-conformant array dimensions for implicit expansion');
        end
    end
end
if last_arr_idx==0
    last_arr_idx=n_inputs;
    param_idx=n_inputs+1;
end

% Expand inputs virtually/explicitly to match full_sz
expanded_args = cell(1, last_arr_idx);
for i = 1:last_arr_idx
    expanded_args{i} = bsxfun(@times, varargin{i}, builtin('ones', full_sz));
end

F = builtin('arrayfun', ReturnFn, expanded_args{:}, varargin{param_idx:end});


end

