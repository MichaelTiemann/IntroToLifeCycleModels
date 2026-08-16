function persistent_worker(worker_id)
    in_file = sprintf('/tmp/vfi_worker_%d_in.mat', worker_id);
    out_file = sprintf('/tmp/vfi_worker_%d_out.mat', worker_id);
    temp_out = sprintf('/tmp/vfi_worker_%d_temp.mat', worker_id);

    while true
        if exist(in_file, 'file')
            % 1. Load the raw handle and args
            load(in_file, 'func', 'args');
            delete(in_file);

            % 2. Check for the kill signal (which we pass as a string)
            if ischar(func) && strcmp(func, 'EXIT')
                break;
            end

            % 3. Execute the math using the loaded handle!
            try
                % The handle 'func' brought its closure (like 'psi') with it!
                result = arrayfun_expand(func, args{:});

                save('-binary', temp_out, 'result');
                rename(temp_out, out_file);
            catch err
                result = err;
                save('-binary', temp_out, 'result');
                rename(temp_out, out_file);
            end
        else
            pause(0.005);
        end
    end
end
