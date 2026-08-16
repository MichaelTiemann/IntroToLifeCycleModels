function out = pararrayfun_vfi(ncores, func, varargin)
    % 1. Determine the target dimensions based on implicit expansion rules
    num_args = length(varargin);
    max_ndims = max(cellfun(@ndims, varargin));
    sz = ones(1, max_ndims);

    for i = 1:num_args
        arg_sz = size(varargin{i});
        sz(1:length(arg_sz)) = max(sz(1:length(arg_sz)), arg_sz);
    end

    % 2. Find the largest dimension to parallelize over (minimizes overhead)
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

    if true
        % 4. Rearrange our split cell array into separate columns
        args_for_parcellfun = cell(1, num_args);
        for a = 1:num_args
            args_for_parcellfun{a} = inputs_split(:, a);
        end

        % 5. Distribute the string-extracted target function
        func_str = func2str(func);
        func_cells = repmat({func_str}, ncores, 1);

        % 6. Execute using parcellfun mapped to safe_worker
        results = parcellfun(ncores, @safe_worker, func_cells, args_for_parcellfun{:}, ...
            'UniformOutput', false, 'VerboseLevel', 1);

        % 7. Reassemble the final array
        out = cat(split_dim, results{:});
    else
        % 4. Rearrange our split cell array into separate columns for parcellfun
        args_for_parcellfun = cell(1, num_args);
        for a = 1:num_args
            args_for_parcellfun{a} = inputs_split(:, a);
        end

        % 5. Distribute the 'builtin' command and the target function handle
        % Create column cells replicated for each core
        builtin_cmd_cells = repmat({'arrayfun'}, ncores, 1);
        func_cells = repmat({regexprep(regexprep(func2str(func),'^.*\) ',''),' \(.*$','')}, ncores, 1);

        % 6. Execute using parcellfun mapped to @builtin
        % The worker natively executes: builtin('arrayfun', func, chunked_arg1, chunked_arg2, ...)
        results = parcellfun(ncores, @builtin, builtin_cmd_cells, func_cells, args_for_parcellfun{:}, 'UniformOutput', false);

        % 7. Reassemble the final array
        out = cat(split_dim, results{:});
    endif
end
