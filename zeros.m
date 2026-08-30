function z = zeros(varargin)

% Remove 'gpuArray' from the input argument list if present
idx = strcmp(varargin, 'gpuArray');
filtered_args = varargin(~idx);

% Call original built-in zeros function
z = builtin('zeros', filtered_args{:});


end

