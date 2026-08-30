function z = ones(varargin)
% Remove 'gpuArray' from the input argument list if present
idx = strcmp(varargin, 'gpuArray');
filtered_args = varargin(~idx);

% Call original built-in ones function
z = builtin('ones', filtered_args{:});


end

