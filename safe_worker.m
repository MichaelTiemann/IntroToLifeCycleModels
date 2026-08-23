function out = safe_worker(func_str, varargin)

try
    % Convert the string back to a handle
    % Octave's str2func handles both 'my_func' and '@(x,y) x+y'
    func_handle = str2func(func_str);

    % Target your explicit expander to handle the bsxfun logic
    out = arrayfun_expand(func_handle, varargin{:});
catch err
    % Grab the Process ID so multiple failing workers don't overwrite the same log
    pid = getpid();
    log_name = sprintf('worker_crash_pid_%d.log', pid);
    fid = fopen(log_name, 'w');

    fprintf(fid, '--- CRASH IN WORKER PID %d ---\n', pid);
    fprintf(fid, 'Target Function: %s\n\n', func_str);
    fprintf(fid, 'Error Message:\n%s\n\n', err.message);
    fprintf(fid, 'Stack Trace:\n');

    for k = 1:length(err.stack)
        fprintf(fid, '  Line %d in %s (%s)\n', ...
            err.stack(k).line, err.stack(k).name, err.stack(k).file);
    end

    % Log the sizes of the inputs that caused the crash
    fprintf(fid, '\nInput Dimensions:\n');
    for i = 1:length(varargin)
        fprintf(fid, '  Arg %d: [%s]\n', i, num2str(size(varargin{i})));
    end

    fclose(fid);

    % Rethrow the error so parcellfun knows this thread died
    rethrow(err);
end


end
