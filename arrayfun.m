function F = arrayfun(ReturnFn, varargin)
    if ReturnFn==@ones
        F=builtin('arrayfun', ReturnFn, varargin{:});
        return;
    endif
    paramidx=1;
    lastarridx=1;
    for idx=1:length(varargin)
        if isscalar(varargin{idx})
            paramidx=idx;
            lastarridx=idx-1;
            break
        endif
    endfor
    assert(lastarridx>1);
    nd_arrays=cell(1,lastarridx);

    vectors=cellfun(@squeeze, varargin(1:lastarridx), 'UniformOutput', false);
    % Call built-in ndgrid dynamically using comma-separated expansion
    [nd_arrays{1:lastarridx}] = ndgrid (vectors{:});

    % Call original built-in zeros function
    F = builtin('pararrayfun', 6, ReturnFn, nd_arrays{1:lastarridx}, varargin{paramidx:end}, "ChunksPerProc", 4);
endfunction

