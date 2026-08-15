function F = arrayfun(ReturnFn, varargin)

# These functions we pass directly to the built-in arrayfun
C = {@ones, @zeros, @nan};
if any(cellfun(@(x) isequal(x, ReturnFn), C));
    F=builtin('arrayfun', ReturnFn, varargin{:});
    return;
endif

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
    endif
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
endif

% Expand inputs virtually/explicitly to match full_sz
expanded_args = cell(1, last_arr_idx);
for i = 1:last_arr_idx
    expanded_args{i} = bsxfun(@times, varargin{i}, ones(full_sz));
end

% Call native arrayfun on identically sized/expanded arrays
if isfield(evalin('base', 'vfoptions'),'n_proc')
    if isfield(evalin('base', 'vfoptions'),'ChunksPerProc')
        chunks_per_proc=evalin('base', 'vfoptions.ChunksPerProc');
    else
        chunks_per_proc=1;
    endif
    F = builtin('pararrayfun', evalin('base','vf_options.n_proc'), ReturnFn, expanded_args{:}, varargin{param_idx:end}, "ChunksPerProc", chunks_per_proc);
else
    F = builtin('arrayfun', ReturnFn, expanded_args{:}, varargin{param_idx:end});
endif


endfunction


