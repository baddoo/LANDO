function [eVals, eVecs, varargout] = linopLANDO(Xdic, Wtilde, kernel, varargin)
%linopLANDO   Extract the linear operator from a LANDO model.
%              
%   [EVALS, EVECS] = LINOPLANDO(XDIC, WTILDE, KERNEL) extracts the
%   eigenvalues (EVALS) and eigenvectors (EVECS) of the local linear 
%   operator of the LANDO model for zero base state defined by the dictionary XDIC,
%   weight matrix WTILDE and kernel structure KERNEL. The matrices XDIC and
%   WTILDE should be learned from TRAINLANDO script. The input kernel class
%   should be determined by the DEFINEKERNEL script.
%
%   [EVALS, EVECS, LINOP] = LINOPLANDO(XDIC, WTILDE, KERNEL) also extracts the
%   linear operator defined by the LANDO model.
%
%   [EVALS, EVECS, LINOP] = LINOPLANDO(XDIC, WTILDE, KERNEL, VALUE) sets the followig parameters:
%   - 'xBar', XBAR: the local base state to linearise about. The default
%   base state is XBAR = 0.
%   - 'nModes', NMODES: the number of modes used in the PCA projection. 
%   - 'xScl', XSCL: a matrix that rescales the X features to improve the
%   conditioning. This must be the same scaling matrix used to train the
%   original LANDO model on whic the XDIC and WTILDE are based.
%
%   Reference:
%   Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon and Steven L. Brunton,
%   "Kernel Learning for Robust Dynamic Mode Decomposition: Linear and  Nonlinear 
%   Disambiguation Optimization (LANDO)", arXiv:2106.01510.
%
%See also trainLANDO, predictLANDO, defineKernel, lorenzExample
%

nx = size(Xdic,1);
[nModes, xScl, xBar] = parseInputs(nx,varargin{:});

sXdic = xScl*Xdic; sxBar = xScl*xBar;
evalKernelDeriv = kernel{2};

% Project model onto principal components of the kernel gradient
[Ux,Sx,Vx] = svd((evalKernelDeriv(sXdic,sxBar)*xScl)',0);
if isempty(nModes); nModes = size(Ux,2); end
nModes = min(nModes,size(Ux,2));
Ux = Ux(:,1:nModes); Sx = Sx(1:nModes,1:nModes); Vx = Vx(:,1:nModes);
LTilde = (Ux'*Wtilde)*(evalKernelDeriv(sXdic,sxBar)*xScl)*Ux;

% Do eigendecomposition of reduced operator
[PsiHat,eVals] = eig(LTilde);
eVals = diag(eVals);

% Sort eigenvalues
[~,idx] = sort(abs(eVals),'descend');
eVals = eVals(idx);
PsiHat = PsiHat(:,idx); % Sort projected eigenvectors similarly

% Project eigenvectors back onto full space
eVecs = Wtilde*Vx*Sx*PsiHat*diag(1./eVals);

% If requested, output the full linear operator
if nargout>2
   
linop = Wtilde*kernel{2}(sXdic,sxBar)*xScl;
varargout{1} = linop;

end
    
%% Extract optional inputs
function [nModes, xScl, xBar] = parseInputs(nx,varargin)

% Defaults
nModes = nx; xScl = 1; xBar = zeros(nx,1);

% Extract optional arguments
j = 0;
while j < nargin-1
   j = j+1;
   v = varargin{j};
   if strcmp(v,'nModes'), j = j+1; nModes = varargin{j};
   elseif strcmp(v,'xScl'), j = j+1; xScl = varargin{j};
   elseif strcmp(v,'xBar'), j = j+1; xBar = varargin{j};
   elseif isempty(v), break
   else
       error('linopLANDO:parseinputs','Unrecognized input')
   end
end
end   % end of parseInputs
end