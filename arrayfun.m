function F = arrayfun(ReturnFn, varargin)

% These functions we pass directly to the built-in arrayfun
C = {@ones, @zeros, @nan, @true, @false};
if length(varargin)<2 || any(cellfun(@(x) isequal(x, ReturnFn), C))
    F = builtin('arrayfun', ReturnFn, varargin{:});
    return;
end

func_str = func2str(ReturnFn);

% Get vfoptions
vfoptions_local = struct();
if evalin('base', "exist('vfoptions', 'var')")
    vfoptions_local = evalin('base', 'vfoptions');
end

% 1. Check if the function is vectorized
is_vectorized = false;
if isfield(vfoptions_local, 'vectorizedarrayfunnames')
    vectorized_names = vfoptions_local.vectorizedarrayfunnames;
    if ischar(vectorized_names)
        vectorized_names = {vectorized_names};
    end

    % HARD OVERRIDE FOR WORKERS: Bypass string matching completely!
    if any(strcmp(vectorized_names, 'FORCE_VECTORIZED'))
        is_vectorized = true;
    else
        for ii = 1:numel(vectorized_names)
            if ~isempty(vectorized_names{ii}) && ~isempty(strfind(func_str, vectorized_names{ii}))
                is_vectorized = true;
                break;
            end
        end
    end
end

% 2. Route directly!
if is_vectorized
    % The fast vectorized path (executes locally on whichever node calls it)
    F = ReturnFn(varargin{:});
else
    % The fallback for scalar functions
    F = arrayfun_expand(ReturnFn, varargin{:});
end

end
