function F = arrayfun(ReturnFn, varargin)

% These functions we pass directly to the built-in arrayfun
C = {@ones, @zeros, @nan, @true, @false};
if length(varargin)<2 || any(cellfun(@(x) isequal(x, ReturnFn), C));
    F = builtin('arrayfun', ReturnFn, varargin{:});
    return;
endif

func_str = func2str(ReturnFn);

% Get vfoptions
vfoptions_local = struct();
if evalin('base', "exist('vfoptions', 'var')")
    vfoptions_local = evalin('base', 'vfoptions');
endif

% 1. Check if the function is vectorized
is_vectorized = false;
if isfield(vfoptions_local, 'vectorizedarrayfunnames')
    vectorized_names = vfoptions_local.vectorizedarrayfunnames;
    if ischar(vectorized_names)
        vectorized_names = {vectorized_names};
    endif

    for ii = 1:numel(vectorized_names)
        if !isempty(vectorized_names{ii}) && ...
                !isempty(strfind(func_str, vectorized_names{ii}))
            is_vectorized = true;
            break;
        endif
    endfor
endif

% 2. Check if parallel processing is requested
use_parallel = isfield(vfoptions_local, 'n_proc') && vfoptions_local.n_proc > 1;

% 3. Route to the appropriate execution path
if is_vectorized
    if use_parallel
        % Pass 'true' for is_vectorized
        F = pararrayfun_vfi(vfoptions_local.n_proc, ReturnFn, true, varargin{:});
    else
        F = ReturnFn(varargin{:});
    endif
    return;
endif

% 4. Fallbacks for standard scalar return functions
if use_parallel && func_str(end)==')'
    % Pass 'false' for is_vectorized
    F = pararrayfun_vfi(vfoptions_local.n_proc, ReturnFn, false, varargin{:});
else
    F = arrayfun_expand(ReturnFn, varargin{:});
endif

endfunction
