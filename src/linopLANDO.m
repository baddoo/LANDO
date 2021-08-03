function [eVals,eVecs,varargout] = linopLANDO(Xdic,Wtilde,kernel,varargin)
% LANDO Linear and nonlinear disambiguation optimization
%           dic = LANDO(X,Y,KERNEL,NU,SCL) returns the dictionary defined
%           by the kernel, nu and the scaling SCL.
%               
%
% Inputs: X = x samples
%         Y = y samples
% kernel = kernel structure
% xBar = linearised state
% further optional inputs 
% 'eigenModes', number of eigenmodes about 
% 
% Examples
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