function varargout = pararrayfun_vfi(ncores, func, is_vectorized, varargin)

% Determine how many outputs were requested by the calling function
num_outs = nargout;
if num_outs == 0
    num_outs = 1;
end

num_args = length(varargin);
num_grid_args = num_args - 2; % Shield the final TWO arguments (parameter arrays)

% =========================================================================
% 1. IDENTIFY WORKLOAD & TARGET DIMENSIONS
% =========================================================================
is_map_reduce = ischar(func) && strcmp(func, 'local_max_worker_FHorz_ExpAssetz_nod1_raw');

if is_map_reduce
    % Explicitly read the candidate dimensions for the map-reduce payload
    len_a2 = size(varargin{7}, 1);
    len_z  = size(varargin{8}, 3);

    if ncores > len_z && len_a2 > len_z
        split_arg = 7;
        ncores = min(ncores, len_a2);
        concat_dim = 2; % Output tensor stitches horizontally along a2
        N = len_a2;
    else
        split_arg = 8;
        ncores = min(ncores, max(1, len_z));
        concat_dim = 3; % Output tensor stitches into depth along z
        N = len_z;
    end
else
    % Fallback generic slicing for standard VFIToolkit functions
    max_ndims = max(cellfun(@ndims, varargin(1:num_grid_args)));
    sz = ones(1, max_ndims);

    for i = 1:num_grid_args
        if isnumeric(varargin{i}) || islogical(varargin{i})
            arg_sz = size(varargin{i});
            sz(1:length(arg_sz)) = max(sz(1:length(arg_sz)), arg_sz);
        end
    end

    ncores_sz = ncores - sz;
    ncores_sz(sz <= 1) = Inf;
    ncores_sz(ncores_sz < 0) = Inf;
    [~, split_dim] = min(ncores_sz);

    if isempty(split_dim) || isinf(ncores_sz(split_dim))
        [~, split_dim] = max(sz);
    end

    ncores = min(ncores, sz(split_dim));
    concat_dim = split_dim;
    N = sz(split_dim);
end

% =========================================================================
% 2. SEQUENTIAL FALLBACK (Zero overhead for debugging / small grids)
% =========================================================================
if ncores <= 1
    if ~exist('is_vectorized', 'var')
        is_vectorized = false;
    end
    varargout = cell(1, num_outs);
    if is_vectorized
        [varargout{:}] = feval(func, varargin{:});
    else
        [varargout{:}] = arrayfun_expand(func, varargin{:});
    end
    return
end

% =========================================================================
% 3. FUSED SLICE & DISPATCH PIPELINE (Zero intermediate memory retention)
% =========================================================================
edges = round(linspace(0, N, ncores + 1));

for c = 1:ncores
    idx = (edges(c) + 1):edges(c + 1);
    args = cell(1, num_args);

    if is_map_reduce
        for a = 1:num_args
            if a == 7 && split_arg == 7
                args{a} = varargin{7}(idx, :); % Slice a2
            elseif a == 8 && split_arg == 8
                args{a} = varargin{8}(1, 1, idx, :); % Slice z
            elseif a == 9 && ~isempty(varargin{9})
                % Align DiscountedEV slice to match the chosen state dimension
                if split_arg == 7
                    args{a} = varargin{9}(:, idx, :);
                else
                    args{a} = varargin{9}(:, :, idx);
                end
            else
                args{a} = varargin{a}; % Pass through un-sliced
            end
        end
    else
        for a = 1:num_grid_args
            if isscalar(varargin{a}) || ~(isnumeric(varargin{a}) || islogical(varargin{a}))
                args{a} = varargin{a}; % Protect scalars and strings
            elseif size(varargin{a}, split_dim) > 1
                S.type = '()';
                S.subs = repmat({':'}, 1, ndims(varargin{a}));
                S.subs{split_dim} = idx;
                args{a} = subsref(varargin{a}, S);
            else
                args{a} = varargin{a};
            end
        end
        % Bypass the slicer entirely for the parameter arrays
        args{num_args - 1} = varargin{num_args - 1};
        args{num_args}     = varargin{num_args};
    end

    % Write to RAM disk immediately. 'args' is overwritten on the next
    % iteration, instantly releasing the slice from master RAM.
    temp_in = sprintf('/Volumes/VFIRAM/vfi_worker_%d_temp.mat', c);
    in_file = sprintf('/Volumes/VFIRAM/vfi_worker_%d_in.mat', c);
    save('-v6', temp_in, 'func', 'args', 'is_vectorized', 'num_outs');
    rename(temp_in, in_file);
end

% =========================================================================
% 4. POLL & COLLECT RESULTS
% =========================================================================
results = cell(ncores, num_outs);
core_completed = false(ncores, 1);
completed = 0;

while completed < ncores
    for c = 1:ncores
        if ~core_completed(c)
            out_file = sprintf('/Volumes/VFIRAM/vfi_worker_%d_out.mat', c);
            if exist(out_file, 'file')
                try
                    data = load(out_file, 'result');
                    delete(out_file);

                    % Robust error propagation from workers
                    if isstruct(data.result) && isfield(data.result, 'message')
                        error('Worker %d exception: %s', c, data.result.message);
                    elseif isa(data.result, 'MException')
                        rethrow(data.result);
                    end

                    if iscell(data.result) && length(data.result) == num_outs
                        for o = 1:num_outs
                            results{c, o} = data.result{o};
                        end
                    else
                        results{c, 1} = data.result;
                    end

                    core_completed(c) = true;
                    completed = completed + 1;
                catch
                    % Catch partial read exceptions when file is mid-write
                end
            end
        end
    end
    if completed < ncores
        pause(0.005);
    end
end

% =========================================================================
% 5. REASSEMBLE TENSORS (Memory cleanly released on return)
% =========================================================================
varargout = cell(1, num_outs);
for o = 1:num_outs
    varargout{o} = cat(concat_dim, results{:, o});
end


end
