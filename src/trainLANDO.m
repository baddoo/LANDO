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
%   conditioning.
%   - 'backslash', 0 or 1: determines whether to use MATLAB's backslash operator 
%   to solve the least-squares problem. This is the default option. Cannot be used 
%   if the online option is enabled.
%   - 'psinv', 0 or 1: determines whether to use the pseudoinverse to
%   sovle the least-squares problem. Usually slower than the backslash operator but 
%   provides further opportunities for (Tikhinov) regularisation. Cannot be used 
%   if the online option is enabled.
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
sX = xScl.*X; % Rescale the X data

% Initialise Cholesky factorisation
ktt = kernel{1}(sX(:,1),sX(:,1));
C = sqrt(ktt);

wr = 1; % Marker for whether the Cholesky warning has been triggered

if online; Pt = 1; Wtilde = Y(:,1)./ktt; end

m = 1; sXdic = sX(:,1); % Initialize index and dictionary
for t = 2:size(X,2)
    sXt = sX(:,t);
    kTildeT = evalKernel(sXdic,sXt);
    % Almost linearly-dependent test
    pit = C'\(C\kTildeT);
    ktt = evalKernel(sXt,sXt);
    delta = ktt - kTildeT'*pit;
    if abs(delta)>nu
        sXdic(:,end+1) = sXt; % Update the dictionary
        % Update Cholesky factor
        C12 = kTildeT'/C';
        if ktt<=norm(C12)^2 && wr
            warning(['The Cholesky factor is ill-conditioned.'...
            'Consider increasing the sparsity parameter or changing the kernel hyperparameters.'])
            wr = 0;
        end
            C(end+1,:) = C12; C(:,end+1) = [zeros(m,1); max(abs(sqrt(ktt - norm(C12)^2)),0)];
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

% Form the unscaled dictionary
Xdic = 1./xScl.*sXdic;

% Solve the least squares problem in batch if the online option is not enabled
if ~online
    if backslash
        Wtilde = Y/evalKernel(sXdic,sX);
    elseif psinv 
        Wtilde = Y*pinv(evalKernel(sXdic,sX),1e-10);
    end
end

% Construct the model
model = @(x) Wtilde*kernel{1}(sXdic,xScl.*x);

% Calculate the reconstruction error of the model
recErr = mean(vecnorm(Y - model(X))./vecnorm(Y));

if displ
    fprintf([
    '------- LANDO completed ------- \n',...
    'Training error:     %4.3f%% \n',...    
    'Time taken:         %4.2f secs\n',...
    'Number of samples:  %d\n',...
    'Size of dictionary: %d\n\n'], 100*recErr, toc, size(X,2), m);
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