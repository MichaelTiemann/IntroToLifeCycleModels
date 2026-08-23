function out = pararrayfun_vfi(ncores, func, is_vectorized, varargin)

% 1. Determine the target dimensions based on implicit expansion rules
num_args = length(varargin);
max_ndims = max(cellfun(@ndims, varargin));
sz = ones(1, max_ndims);

for i = 1:num_args
    arg_sz = size(varargin{i});
    [max_d, max_i] = max(arg_sz);
    sz(max_i) = max(sz(max_i), max_d);
end

% 2. Find the best dimension to parallelize over (minimizes overhead)
ncores_sz=ncores-sz;
ncores_sz(sz==1)=Inf;
ncores_sz(ncores_sz<0)=Inf;
[~, split_dim] = min(ncores_sz);
if ~isempty(split_dim) && sz(split_dim) <= ncores
    % We are going to split into exactly split_dim pieces
    ncores=sz(split_dim);
    inputs_split = cell(ncores, num_args);
    indices = repmat({':'}, 1, max_ndims);
    for c = 1:ncores
        indices{split_dim} = c;
        for a = 1:num_args
            A=varargin{a};
            if isscalar(A)
                % Pass scalars directly; Octave's local arrayfun handles them perfectly
                inputs_split{c, a} = A;
            elseif size(A, split_dim) > 1
                % Slice the matrix along the split dimension
                inputs_split{c, a} = A(indices{:});
            else
                % Pass singleton dimensions (e.g., 1x201) directly to the worker
                inputs_split{c, a} = A;
            end
        end
    end
else
    [~, split_dim] = max(sz);
    N = sz(split_dim);

    % Since we are called by arrayfun, we cannot fall back if N < ncores

    % 3. Partition the inputs to avoid pre-expansion
    % Create chunk boundaries
    edges = round(linspace(0, N, ncores + 1));
    inputs_split = cell(ncores, num_args);

    for c = 1:ncores
        idx = (edges(c) + 1):edges(c + 1);
        for a = 1:num_args
            if isscalar(varargin{a})
                % Pass scalars directly; Octave's local arrayfun handles them perfectly
                inputs_split{c, a} = varargin{a};
            elseif size(varargin{a}, split_dim) > 1
                % Slice the large arrays along the split dimension
                S.type = '()';
                S.subs = repmat({':'}, 1, ndims(varargin{a}));
                S.subs{split_dim} = idx;
                inputs_split{c, a} = subsref(varargin{a}, S);
            else
                % Pass singleton dimensions (e.g., 1x201) directly to the worker
                inputs_split{c, a} = varargin{a};
            end
        end
    end
end

% 4. Rearrange our split cell array into separate columns
args_for_parcellfun = cell(1, num_args);
for a = 1:num_args
    args_for_parcellfun{a} = inputs_split(:, a);
end

% 5. Dispatch chunks to the persistent workers
for c = 1:ncores
    args = cellfun(@(x) x(c), args_for_parcellfun);
    temp_in = sprintf('/tmp/vfi_worker_%d_temp.mat', c);
    in_file = sprintf('/tmp/vfi_worker_%d_in.mat', c);

    % SAVE the new is_vectorized flag along with func and args
    save('-binary', temp_in, 'func', 'args', 'is_vectorized');
    rename(temp_in, in_file);
end

% 6. Wait for all workers to finish their slice
results = cell(ncores, 1);
completed = 0;
while completed < ncores
    for c = 1:ncores
        out_file = sprintf('/tmp/vfi_worker_%d_out.mat', c);

        if isempty(results{c}) && exist(out_file, 'file')
            data = load(out_file, 'result');
            delete(out_file);

            if isa(data.result, 'MException')
                rethrow(data.result); % Crash gracefully if worker failed
            end

            results{c} = data.result;
            completed = completed + 1;
        end
    end
    if completed < ncores
        pause(0.005); % Yield briefly
    end
end

% 7. Reassemble the final array
out = cat(split_dim, results{:});


end
