function F = arrayfun(ReturnFn, varargin)

# These functions we pass directly to the built-in arrayfun
C = {@ones, @zeros, @nan, @true, @false};
if length(varargin)<2 || any(cellfun(@(x) isequal(x, ReturnFn), C));
    F=builtin('arrayfun', ReturnFn, varargin{:});
    return;
endif

func_str = func2str(ReturnFn);

% Call native arrayfun on identically sized/expanded arrays
if isfield(evalin('base', 'vfoptions'),'n_proc') && func_str(end)==')'
    F = pararrayfun_vfi(evalin('base','vfoptions.n_proc'), ReturnFn, varargin{:});
else
    F = arrayfun_expand(ReturnFn, varargin{:});
endif


endfunction


