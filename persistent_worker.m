function persistent_worker(worker_id)

in_file    = sprintf('/Volumes/VFIRAM/vfi_worker_%d_in.mat', worker_id);
out_file   = sprintf('/Volumes/VFIRAM/vfi_worker_%d_out.mat', worker_id);
temp_out   = sprintf('/Volumes/VFIRAM/vfi_worker_%d_temp.mat', worker_id);
ready_file = sprintf('/Volumes/VFIRAM/vfi_worker_%d_ready', worker_id);
exit_file  = sprintf('/Volumes/VFIRAM/vfi_worker_%d_exited', worker_id);

% 1. Signal boot completion to the pool coordinator
fclose(fopen(ready_file, 'w'));

while true
    if exist(in_file, 'file')
        % Load the exact payload matching pararrayfun_vfi
        load(in_file, 'func', 'args', 'is_vectorized', 'num_outs');
        delete(in_file);

        % 2. Check for the kill signal
        if ischar(func) && strcmp(func, 'EXIT')
            fclose(fopen(exit_file, 'w'));
            break;
        end

        % 3. Execute the math!
        try
            if ~exist('is_vectorized', 'var')
                is_vectorized = false;
            end
            if ~exist('num_outs', 'var')
               num_outs = 1;
            end

            result = cell(1, num_outs);
            if is_vectorized
                % THE FAST PATH: Pure feval
                [result{:}] = feval(func, args{:});
            else
                % THE SCALAR PATH: Fall back to element-by-element
                [result{:}] = arrayfun_expand(func, args{:});
            end

            save('-v6', temp_out, 'result');
            rename(temp_out, out_file);

        catch err
            result = err;
            save('-v6', temp_out, 'result');
            rename(temp_out, out_file);
        end

        % Purge local memory
        clear args func result is_vectorized num_outs;
    else
        pause(0.001);
    end
end


end
