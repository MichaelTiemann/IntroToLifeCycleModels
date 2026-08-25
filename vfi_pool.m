function vfi_pool(action, ncores)

if strcmp(action, 'start')
    fprintf('Starting %d persistent workers...\n', ncores);
    system('rm -f /Volumes/VFIRAM/vfi_worker_*'); % Clean stale files
    for i = 1:ncores
        % Launch headless octave processes in the background
        system(sprintf('octave --no-gui --eval "persistent_worker(%d)" &', i));
    end
    pause(2); % Give them a second to boot up and idle

elseif strcmp(action, 'stop')
    fprintf('Shutting down workers...\n');
    for i = 1:ncores
        func = 'EXIT'; % Save it as 'func' to match the worker's load command
        args = {};
        in_file = sprintf('/Volumes/VFIRAM/vfi_worker_%d_in.mat', i);
        save('-v6', in_file, 'func', 'args');
    end
end


end
