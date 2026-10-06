function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d, n_a2, d_gridvals, a2_grid, aprimeFnParams, aprimeIndexAsColumn, n_z, z_gridvals, n_e, e_gridvals)
% For experienceasset: enumerate a2prime=aprimeFn(d, a2) over ALL d (used
% during value-function iteration). Because the true value of a2prime will
% (almost always) lie between two consecutive points in a2_grid, it is
% linearly interpolated back on to a2_grid. Thus the continuous a2prime is
% represented by (index of lower grid point in a2primeIndexes, probability
% of lower grid point in a2primeProbs) on a2_grid; the upper index is
% implicitly lower+1 with prob 1-minus-prob-of-lower.
%
% Output sizes:
%   l_a2==1 (legacy):
%     a2primeIndexes - col=1 => [N_all, 1]; col=2 => [Nd_eff, N_a2]
%     a2primeProbs   - [Nd_eff, N_a2]; upper idx = lower+1, prob upper = 1-prob lower
%   l_a2==2 (multi-dim, per-dim factored -- NOT Kron-folded corners):
%     a2primeIndexes - col=1 => [l_a2, N_all]; col=2 => [l_a2, Nd_eff, N_a2]
%     a2primeProbs   - [l_a2, Nd_eff, N_a2] ALWAYS (unlike a2primeIndexes, its shape does not
%                      depend on aprimeIndexAsColumn)
%     Row k is the a2_k dimension on its own: a2primeIndexes(k,...) is the lower-grid index
%     within that dimension (1..n_a2(k)) and a2primeProbs(k,...) the probability of that lower
%     point. These are per-dimension indices, NOT linear indices in N_a2=prod(n_a2) space.
%     The caller combines them into the four corners itself and does a nested 2-corner interp
%     with skipinterp at each level (bit-exact when V is flat in a dimension).

% Catch omitted trailing arguments for backward compatibility
if nargin < 8; n_z = 0; z_gridvals = []; end
if nargin < 10; n_e = 0; e_gridvals = []; end

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if size(aprimeFnParams(ii))~=[1,1]
        error('Using experienceasset does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_d=prod(n_d);
Nd_eff = max(N_d, 1);

N_a2=prod(n_a2);

N_z = prod(n_z);
Nz_eff = max(N_z, 1);

N_e = prod(n_e);
Ne_eff = max(N_e, 1);

N_all = Nd_eff * N_a2 * Nz_eff * Ne_eff; % The new dynamic total number of elements

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
if l_d>4
    error('experienceasset does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a2>2
    error('experienceasset currently supports length(n_a2) in {1,2}')
end

l_z = 0; if N_z > 0; l_z = length(n_z); end
l_e = 0; if N_e > 0; l_e = length(n_e); end

if nargin(aprimeFn) ~= l_d + l_a2 + (l_a2 >= 2) + l_z + l_e + length(aprimeFnParams)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

% --- ELEGANT DYNAMIC GRID PACKING ---

% 1. Pack 'd' variables (Dimension 1)
d_vals = cell(1, l_d);
for i = 1:l_d
    if l_d == 1; v = d_gridvals; else; v = d_gridvals(:, i); end
    d_vals{i} = v;
end

% 2. Pack Exogenous States (z, e)
% The shift offset for exogenous states must start after a2.
% If l_a2=1, z starts at dim 3 (shift -2). If l_a2=2, z starts at dim 4 (shift -3).
shift_offset = -(1 + l_a2);

z_vals_cell = {};
if ~isempty(z_gridvals)
    l_z = length(n_z);
    z_vals_cell = cell(1, l_z);
    if l_z == 1; z_vals_cell{1} = shiftdim(z_gridvals, shift_offset);
    else
        for i = 1:l_z
            z_vals_cell{i} = shiftdim(z_gridvals(:, i), shift_offset);
        end
    end
    shift_offset = shift_offset - 1;
end

e_vals_cell = {};
if ~isempty(e_gridvals)
    l_e = length(n_e);
    e_vals_cell = cell(1, l_e);
    if l_e == 1; e_vals_cell{1} = shiftdim(e_gridvals, shift_offset);
    else
        for i = 1:l_e
            e_vals_cell{i} = shiftdim(e_gridvals(:, i), shift_offset);
        end
    end
    shift_offset = shift_offset - 1; % Ready for u_gridvals later
end

% 3. Evaluate arrayfun seamlessly for both 1D and 2D Experience Assets
if l_a2 == 1
    a2vals_cell = {shiftdim(a2_grid(1:n_a2(1)), -1)};
    
    % Combine all inputs positionally: [d, a2, z, e, Params]
    % Note: ParamCell' transposes the column cell to a row cell for horizontal concatenation
    GridParamsCell = [d_vals, a2vals_cell, z_vals_cell, e_vals_cell, ParamCell'];
    
    a2primeVals = arrayfun(aprimeFn, GridParamsCell{:});
    
elseif l_a2 == 2
    n_a2_1 = n_a2(1);
    n_a2_2 = n_a2(2);
    a2_grid_1 = a2_grid(1:n_a2_1);
    a2_grid_2 = a2_grid(n_a2_1+1:n_a2_1+n_a2_2);
    
    a2vals_cell = {shiftdim(a2_grid_1, -1), shiftdim(a2_grid_2, -2)};
    
    % For l_a2 == 2, aprimeFn requires a 'whicha' selector (1 or 2) injected immediately after the a2 inputs
    GridParamsCell_1 = [d_vals, a2vals_cell, {1}, z_vals_cell, e_vals_cell, ParamCell'];
    GridParamsCell_2 = [d_vals, a2vals_cell, {2}, z_vals_cell, e_vals_cell, ParamCell'];
    
    a2primeVals_1 = arrayfun(aprimeFn, GridParamsCell_1{:});
    a2primeVals_2 = arrayfun(aprimeFn, GridParamsCell_2{:});
end

if l_a2==1

    %% Calculate grid indexes and probs from the values
    a2primeVals=reshape(a2primeVals,[1,N_all]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

    % For small aprimeVals and a_grid, max() is faster than discretize()
    if N_all*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1);
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;
        a2primeIndexes=reshape(a2primeIndexes,[N_all,1]);

        aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        a2primeProbs(offTopOfGrid)=0;
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid);
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1 % value fn codes want column when no z
        a2primeIndexes=a2primeIndexes';
    elseif aprimeIndexAsColumn==3 % value fn with another asset uses 3
        a2primeIndexes=reshape(a2primeIndexes,[N_all,Nz_eff,Ne_eff]);
    else % aprimeIndexAsColumn==2 % value fn with z, or simulation, want matrix
        a2primeIndexes=reshape(a2primeIndexes,[Nd_eff,N_a2,Nz_eff,Ne_eff]);
    end
    a2primeProbs=reshape(a2primeProbs,[Nd_eff,N_a2,Nz_eff,Ne_eff]);

elseif l_a2==2
    %% Multi-dim a2 (l_a2=2): bilinear interp, returned PER-DIM FACTORED (the caller folds the 4 corners)
    n_a2_1=n_a2(1); n_a2_2=n_a2(2);
    a2_grid_1=a2_grid(1:n_a2_1);
    a2_grid_2=a2_grid(n_a2_1+1:n_a2_1+n_a2_2);

    %% Per-dim grid indexes and probs (inlined 1D linear-interp; mirrors l_a2==1 above)
    a2primeVals_1=reshape(a2primeVals_1,[1,N_all]);
    a2primeVals_2=reshape(a2primeVals_2,[1,N_all]);
    a2_griddiff_1=a2_grid_1(2:end)-a2_grid_1(1:end-1);
    a2_griddiff_2=a2_grid_2(2:end)-a2_grid_2(1:end-1);

    % --- a2 dim 1 ---
    if N_all*n_a2_1<1000000
        [~,loIdx_1]=max((a2_grid_1>a2primeVals_1),[],1);
        loIdx_1=loIdx_1-1;
        loIdx_1(loIdx_1==0)=1;
        aprime_residual_1=a2primeVals_1'-a2_grid_1(loIdx_1);
        prob_1=1-aprime_residual_1./a2_griddiff_1(loIdx_1);
        offTopOfGrid_1=(a2primeVals_1>=a2_grid_1(end));
        loIdx_1(offTopOfGrid_1)=n_a2_1-1;
        prob_1(offTopOfGrid_1)=0;
        offBottomOfGrid_1=(a2primeVals_1<=a2_grid_1(1));
        prob_1(offBottomOfGrid_1)=1;
    else
        loIdx_1=discretize(a2primeVals_1,a2_grid_1);
        offBottomOfGrid_1=(a2primeVals_1<=a2_grid_1(1));
        loIdx_1(offBottomOfGrid_1)=1;
        offTopOfGrid_1=(a2primeVals_1>=a2_grid_1(end));
        loIdx_1(offTopOfGrid_1)=n_a2_1-1;
        aprime_residual_1=a2primeVals_1'-a2_grid_1(loIdx_1);
        prob_1=1-aprime_residual_1./a2_griddiff_1(loIdx_1);
        prob_1(offBottomOfGrid_1)=1;
        prob_1(offTopOfGrid_1)=0;
    end

    % --- a2 dim 2 ---
    if N_all*n_a2_2<1000000
        [~,loIdx_2]=max((a2_grid_2>a2primeVals_2),[],1);
        loIdx_2=loIdx_2-1;
        loIdx_2(loIdx_2==0)=1;
        aprime_residual_2=a2primeVals_2'-a2_grid_2(loIdx_2);
        prob_2=1-aprime_residual_2./a2_griddiff_2(loIdx_2);
        offTopOfGrid_2=(a2primeVals_2>=a2_grid_2(end));
        loIdx_2(offTopOfGrid_2)=n_a2_2-1;
        prob_2(offTopOfGrid_2)=0;
        offBottomOfGrid_2=(a2primeVals_2<=a2_grid_2(1));
        prob_2(offBottomOfGrid_2)=1;
    else
        loIdx_2=discretize(a2primeVals_2,a2_grid_2);
        offBottomOfGrid_2=(a2primeVals_2<=a2_grid_2(1));
        loIdx_2(offBottomOfGrid_2)=1;
        offTopOfGrid_2=(a2primeVals_2>=a2_grid_2(end));
        loIdx_2(offTopOfGrid_2)=n_a2_2-1;
        aprime_residual_2=a2primeVals_2'-a2_grid_2(loIdx_2);
        prob_2=1-aprime_residual_2./a2_griddiff_2(loIdx_2);
        prob_2(offBottomOfGrid_2)=1;
        prob_2(offTopOfGrid_2)=0;
    end

    % Per-dim factored output (NOT Kron-folded):
    %   a2primeIndexes(k,:) = lower-grid index in a2_k dim (1..n_a2(k))
    %   a2primeProbs(k,:)   = probability of lower grid point in a2_k dim
    % Caller does nested 2-corner interp with skipinterp at each level (bit-exact when V is flat).
    a2primeIndexes=zeros(l_a2,N_all,'gpuArray');
    a2primeProbs=zeros(l_a2,N_all,'gpuArray');
    a2primeIndexes(1,:)=loIdx_1(:);
    a2primeIndexes(2,:)=loIdx_2(:);
    a2primeProbs(1,:)=prob_1(:);
    a2primeProbs(2,:)=prob_2(:);

    if aprimeIndexAsColumn==1 % column-flat layout
        % already [l_a2, N_all]
    else % aprimeIndexAsColumn==2 % matrix layout
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,Nd_eff,N_a2,Nz_eff,Ne_eff]);
    end
    a2primeProbs=reshape(a2primeProbs,[l_a2,Nd_eff,N_a2,Nz_eff,Ne_eff]);
end


end
