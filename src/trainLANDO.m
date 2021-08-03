function [model, Xdic, Wtilde, recErr] = trainLANDO(X, Y, nu, kernel, varargin)
%trainLANDO   Learn a LANDO model.
%              
%   MODEL = TRAINLANDO(X, Y, NU, KERNEL) trains a kernel model (MODEL) 
%           in batch based on input data matrices X and Y. The dictionary
%           sparsification parameter is set by NU. The input KERNEL
%           class should be determined by the DEFINEKERNEL script.
%
%   [MODEL, XDIC, WTILDE] = TRAINLANDO(X, Y, NU, KERNEL) also returns the sparsified
%   dictionary XDIC on which the model is based, and the learned weight
%   matrix WTILDE.
%
%   MODEL = TRAINLANDO(X, Y, NU, KERNEL, VALUE) sets the followig parameters:
%   - 'online', 0 or 1: whether the algorithm operates online, (default
%   ONLINE = 0).
%   - 'xScl', XSCL: a matrix that rescales the X features to improve the
%   conditioning
%   - 'backslash', 0 or 1: determines whether to use MATLAB's backslash operator 
%   to solve the least-squares problem. This is the default option. Cannot be used 
%   if the online option is enabled.
%   - 'psinv', 0 or 1: determines whether to use the pseudoinverse to
%   sovle the least-squares problem. Usually slower than the backslash operator but 
%   provides further opportunities for (Tikhinov) regularisation. Cannot be used 
%   if the online option is enabled
%   - 'display', 0 or 1: determines whether to display the results of LANDO. 
%   Turned off by default.
%
%   Reference:
%   Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon and Steven L. Brunton,
%   "Kernel Learning for Robust Dynamic Mode Decomposition: Linear and  Nonlinear 
%   Disambiguation Optimization (LANDO)", arXiv:2106.01510.
%
%See also linopLANDO, predictLANDO, defineKernel, lorenzExample
%

% Parse inputs:
[online, xScl, backslash, psinv, displ] = parseInputs(varargin{:});
tic % Begin timer

evalKernel = kernel{1}; % Extract the kernel evaluation function
sX = xScl*X; % Rescale the X data

% Initialise Cholesky factorisation
ktt = evalKernel(sX(:,1),sX(:,1));
C = sqrt(ktt);

wr = 1; % Marker for whether the Cholesky warning has been triggered

if online; Pt = 1; Wtilde = Y(:,1)./ktt; end

m = 1; dIdx = 1; % Define indices

for t = 2:size(X,2)
    kTildeT = evalKernel(sX(:,dIdx),sX(:,t));
    % Almost linearly-dependent test
    pit = C'\(C\kTildeT);
    ktt = evalKernel(sX(:,t),sX(:,t));
    delta = ktt - kTildeT'*pit;
    if abs(delta)>nu
        dIdx = [dIdx; t]; % Update the dictionary index
        % Update Cholesky factor
        C12 = kTildeT'/C';
        if ktt<=norm(C12)^2 && wr
            warning(['The Cholesky factor is ill-conditioned.'...
            'Consider increasing the sparsity parameter or changing the kernel hyperparameters.'])
            wr = 0;
        end
            C = [C, zeros(m,1); C12, max(sqrt(ktt - norm(C12)^2),0)];
            if online % Update the model if the online option is enabled
                Pt = [Pt , zeros(m,1); zeros(1,m) 1];
                YminWKD = (Y(:,t) - Wtilde*kTildeT)/delta;
                Wtilde = [Wtilde - YminWKD*pit', YminWKD];
            end
            m = m+1;
    else
       if online % Update the model if the online option is enabled
           ht = pit'*Pt/(1 + pit'*Pt*pit);
           Pt = Pt - (Pt*pit)*ht;
           Wtilde = Wtilde + (((Y(:,t) - Wtilde*kTildeT)*ht)/C')/C;
       end
    end
end

% Form the dictionary and scaled dictionary
Xdic = X(:,dIdx); sXdic = xScl*Xdic;

% Solve the least squares problem in batch if the online option is not enabled
if ~online
    if backslash
        Wtilde = Y/evalKernel(sXdic,sX);
    elseif psinv 
        [Uk,Sk,Vk] = svd(evalKernel(sXdic,sX),0);
        kTrunc = find(diag(Sk)<eps*Sk(1),1); if isempty(kTrunc); kTrunc = size(Sk,1); end
        Wtilde = Y*Vk(:,1:kTrunc)*pinv(Sk(1:kTrunc,1:kTrunc))*Uk(:,1:kTrunc)';
    end
end

% Construct the model
model = @(x) Wtilde*kernel{1}(sXdic,xScl*x);

% Calculate the reconstruction error of the model
recErr = mean(vecnorm(Y - model(X))./vecnorm(Y));

if displ
    fprintf([
    '------- LANDO completed ------- \n',...
    'Training error:     %4.3f%% \n',...    
    'Time taken:         %4.2f secs\n',...
    'Number of samples:  %d\n',...
    'Size of dictionary: %d\n\n'], 100*recErr,toc,size(X,2), m);
end

%% Extract optional inputs
function [online, xScl, backslash, psinv, displ] = parseInputs(varargin)

% Defaults
online = 0; xScl = 1; displ = 0;
backslash = 1; psinv = 0;

% Extract optional arguments
j = 0;
while j < nargin-1
   j = j+1;
   v = varargin{j};
   if strcmp(v,'xScl'), j = j+1; xScl = varargin{j};
   elseif strcmp(v,'online'), online = 1; 
   elseif strcmp(v,'display'), displ = 1; 
   elseif strcmp(v,'backslash'), backslash = 1; psinv = 0;
   elseif strcmp(v,'psinv'), psinv = 1; backslash = 0;
   elseif isempty(v), break
   else
       error('trainLANDO:parseinputs','Unrecognized input')
   end
end
end   % end of parseInputs
end