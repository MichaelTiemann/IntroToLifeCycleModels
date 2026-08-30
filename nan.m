function z = nan(varargin)
% Remove 'gpuArray' from the input argument list if present
idx = strcmp(varargin, 'gpuArray');
filtered_args = varargin(~idx);

% Call original built-in ones function
z = builtin('nan', filtered_args{:});


end
