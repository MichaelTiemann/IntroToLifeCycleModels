function varargout = pararrayfun_vfi(ncores, func, is_vectorized, varargin)

% Determine how many outputs were requested by the calling function
num_outs = nargout;
if num_outs == 0
    num_outs = 1;
end

num_args = length(varargin);
num_grid_args = num_args - 2; % Shield the final TWO arguments (ReturnFnParamNames & ReturnFnParamsVec) from the slicer!

% 1. Determine target dimensions using ONLY grid/state arguments
max_ndims = max(cellfun(@ndims, varargin(1:num_grid_args)));
sz = ones(1, max_ndims);

for i = 1:num_grid_args
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
    % Perfect match logic
    ncores=sz(split_dim);
    inputs_split = cell(ncores, num_args);
    indices = repmat({':'}, 1, max_ndims);
    for c = 1:ncores
        indices{split_dim} = c;
        for a = 1:num_grid_args
            A = varargin{a};
            if isscalar(A)
                inputs_split{c, a} = A;
            elseif size(A, split_dim) > 1
                inputs_split{c, a} = A(indices{:});
            else
                inputs_split{c, a} = A;
            end
        end
        % Bypassing the slicer completely for the parameter names and values vectors!
        inputs_split{c, num_args - 1} = varargin{num_args - 1};
        inputs_split{c, num_args}     = varargin{num_args};
    end
else
    % Fallback logic (slicing largest dimension)
    [~, split_dim] = max(sz);
    N = sz(split_dim);
    edges = round(linspace(0, N, ncores + 1));
    inputs_split = cell(ncores, num_args);

    for c = 1:ncores
        idx = (edges(c) + 1):edges(c + 1);
        for a = 1:num_grid_args
            if isscalar(varargin{a})
                inputs_split{c, a} = varargin{a};
            elseif size(varargin{a}, split_dim) > 1
                S.type = '()';
                S.subs = repmat({':'}, 1, ndims(varargin{a}));
                S.subs{split_dim} = idx;
                inputs_split{c, a} = subsref(varargin{a}, S);
            else
                inputs_split{c, a} = varargin{a};
            end
        end
        % Bypassing the slicer completely for the parameter names and values vectors!
        inputs_split{c, num_args - 1} = varargin{num_args - 1};
        inputs_split{c, num_args}     = varargin{num_args};
    end
end

% 4. Rearrange our split cell array into separate columns
args_for_parcellfun = cell(1, num_args);
for a = 1:num_args
    args_for_parcellfun{a} = inputs_split(:, a);
end

% 5. Dispatch chunks to the persistent workers
for c = 1:ncores
    args = cell(1, num_args);
    for a = 1:num_args
        args{a} = args_for_parcellfun{a}{c};
    end

    temp_in = sprintf('/tmp/vfi_worker_%d_temp.mat', c);
    in_file = sprintf('/tmp/vfi_worker_%d_in.mat', c);

    save('-binary', temp_in, 'func', 'args', 'is_vectorized', 'num_outs');
    rename(temp_in, in_file);
end

% Clear the split structures to free master RAM during the compute wait
clear inputs_split args_for_parcellfun args A S;

% 6. Wait for all workers to finish their slice
results = cell(ncores, num_outs);
core_completed = false(ncores, 1);
completed = 0;

while completed < ncores
    for c = 1:ncores
        if ~core_completed(c)
            out_file = sprintf('/tmp/vfi_worker_%d_out.mat', c);
            if exist(out_file, 'file')
                try
                    data = load(out_file, 'result');
                    delete(out_file);

                    if isa(data.result, 'MException')
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
                    % File might be mid-write; catch error and try again
                end
            end
        end
    end
    if completed < ncores
        pause(0.005);
    end
end

% 7. Reassemble the final arrays for varargout
varargout = cell(1, num_outs);
for o = 1:num_outs
    varargout{o} = cat(split_dim, results{:, o});
end

% Clear the raw results cell array before returning to the main toolkit loop
clear results data;

end
