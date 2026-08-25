function persistent_worker(worker_id)

in_file = sprintf('/Volumes/VFIRAM/vfi_worker_%d_in.mat', worker_id);
out_file = sprintf('/Volumes/VFIRAM/vfi_worker_%d_out.mat', worker_id);
temp_out = sprintf('/Volumes/VFIRAM/vfi_worker_%d_temp.mat', worker_id);

while true
    if exist(in_file, 'file')
        % 1. Load the raw payload
        load(in_file, 'func', 'args', 'is_vectorized', 'num_outs');
        delete(in_file);

        % 2. Check for the kill signal
        if ischar(func) && strcmp(func, 'EXIT')
            break;
        end

        % 3. Execute the math!
        try
            if ~exist('is_vectorized', 'var')
                is_vectorized = false;
            end

            result = cell(1, num_outs);

            if is_vectorized
                % THE FAST PATH: Pure feval, no workspace hacks needed!
                [result{:}] = feval(func, args{:});
            else
                [result{:}] = arrayfun_expand(func, args{:});
            end

            save('-v6', temp_out, 'result');
            rename(temp_out, out_file);
        catch err
            result = err;
            save('-v6', temp_out, 'result');
            rename(temp_out, out_file);
        end

        % Purge the payload from local memory
        clear args func result is_vectorized num_outs;
    else
        pause(0.005);
    end
end


end
