function persistent_worker(worker_id)

ram_dir = '/mnt/VFIRAM';
in_file    = sprintf('%s/vfi_worker_%d_in.mat', ram_dir, worker_id);
out_file   = sprintf('%s/vfi_worker_%d_out.mat', ram_dir, worker_id);
temp_out   = sprintf('%s/vfi_worker_%d_temp.mat', ram_dir, worker_id);
ready_file = sprintf('%s/vfi_worker_%d_ready', ram_dir, worker_id);
exit_file  = sprintf('%s/vfi_worker_%d_exited', ram_dir, worker_id);

% Signal ready barrier to the pool coordinator
fclose(fopen(ready_file, 'w'));

while true
    if exist(in_file, 'file')
        % Delegate to sub-function for deterministic scope-exit memory release
        should_exit = process_task(in_file, out_file, temp_out, exit_file);
        if should_exit
            exit(0);
        end
    else
        pause(0.005);
    end
end


end

% --- LOCAL SUB-FUNCTION FOR SCOPE ISOLATION ---
function should_exit = process_task(in_file, out_file, temp_out, exit_file)

should_exit = false;
try
    % 1. Load payload (allocates to this temporary scope)
    load(in_file, 'func', 'args', 'is_vectorized', 'num_outs');
    delete(in_file);

    % 2. Check for the kill signal
    if ischar(func) && strcmp(func, 'EXIT')
        fclose(fopen(exit_file, 'w'));
        should_exit = true;
        return;
    end

    % 3. Execute the math
    if ~exist('is_vectorized', 'var'), is_vectorized = false; end
    if ~exist('num_outs', 'var'), num_outs = 1; end

    result = cell(1, num_outs);
    if is_vectorized
        [result{:}] = feval(func, args{:});
    else
        [result{:}] = arrayfun_expand(func, args{:});
    end

    save('-v6', temp_out, 'result');
    rename(temp_out, out_file);

catch err
    if exist(in_file, 'file')
        delete(in_file);
    end
    result = err;
    save('-v6', temp_out, 'result');
    rename(temp_out, out_file);
end
% When this function returns, 'args', 'result', and 'func' are automatically
% and cleanly destroyed by Octave. No 'clear' commands needed!


end
