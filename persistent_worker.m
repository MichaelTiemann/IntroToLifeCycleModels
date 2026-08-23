function persistent_worker(worker_id)
in_file = sprintf('/tmp/vfi_worker_%d_in.mat', worker_id);
out_file = sprintf('/tmp/vfi_worker_%d_out.mat', worker_id);
temp_out = sprintf('/tmp/vfi_worker_%d_temp.mat', worker_id);

while true
    if exist(in_file, 'file')
        % 1. Load the raw handle, args, AND the flag
        load(in_file, 'func', 'args', 'is_vectorized');
        delete(in_file);

        % 2. Check for the kill signal
        if ischar(func) && strcmp(func, 'EXIT')
            break;
        end

        % 3. Execute the math!
        try
            % If it's an older worker file without the flag, default to false
            if ~exist('is_vectorized', 'var')
                is_vectorized = false;
            end

            if is_vectorized
                % THE FAST PATH: Direct evaluation of the chunked arrays!
                result = func(args{:});
            else
                % THE SCALAR PATH: Fall back to element-by-element
                result = arrayfun_expand(func, args{:});
            end

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
