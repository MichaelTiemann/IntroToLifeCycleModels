function vfi_pool(action, ncores)

ram_dir = '/mnt/VFIRAM';

switch lower(action)
    case 'stop'
        fprintf('Shutting down %d workers...\n', ncores);

        % 1. Send EXIT payload to all workers
        for i = 1:ncores
            func = 'EXIT';
            args = {};
            is_vectorized = false;
            num_outs = 1;
            in_file = sprintf('%s/vfi_worker_%d_in.mat', ram_dir, i);
            save('-v6', in_file, 'func', 'args', 'is_vectorized', 'num_outs');
        end

        % 2. Shutdown Barrier: Block until every worker signals exit
        for i = 1:ncores
            exit_file = sprintf('%s/vfi_worker_%d_exited', ram_dir, i);
            while ~exist(exit_file, 'file')
                pause(0.001);
            end
            delete(exit_file);
        end

        % 3. Clean up stale worker files
        system(sprintf('rm -f %s/vfi_worker_*', ram_dir));

    case 'start'
        fprintf('Starting %d persistent workers...\n', ncores);
        system(sprintf('rm -f %s/vfi_worker_*', ram_dir));

        % Launch headless worker processes in the background
        for i = 1:ncores
            system(sprintf('octave --no-gui --eval "persistent_worker(%d)" &', i));
        end

        % Startup Barrier: Block until all workers confirm initialization
        for i = 1:ncores
            ready_file = sprintf('%s/vfi_worker_%d_ready', ram_dir, i);
            while ~exist(ready_file, 'file')
                pause(0.001);
            end
            delete(ready_file);
        end

    case 'restart'
        vfi_pool('stop', ncores);
        vfi_pool('start', ncores);
end


end
