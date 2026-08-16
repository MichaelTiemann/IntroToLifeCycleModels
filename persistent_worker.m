function persistent_worker(worker_id)
    in_file = sprintf('/tmp/vfi_worker_%d_in.mat', worker_id);
    out_file = sprintf('/tmp/vfi_worker_%d_out.mat', worker_id);
    temp_out = sprintf('/tmp/vfi_worker_%d_temp.mat', worker_id);

    while true
        if exist(in_file, 'file')
            % 1. Load the task and delete the trigger file
            load(in_file, 'func_str', 'args');
            delete(in_file);

            % 2. Check for the kill signal
            if strcmp(func_str, 'EXIT')
                break;
            end

            % 3. Execute the math
            try
                func_handle = str2func(func_str);
                result = arrayfun_expand(func_handle, args{:}); % Target your expander!

                % 4. Save using fast binary format, then atomically rename
                save('-binary', temp_out, 'result');
                rename(temp_out, out_file);
            catch err
                result = err;
                save('-binary', temp_out, 'result');
                rename(temp_out, out_file);
            end
        else
            % Idle yield so we don't cook the CPU while waiting for the next jj period
            pause(0.005);
        end
    end
end
