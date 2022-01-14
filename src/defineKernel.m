function k = defineKernel(varargin)
%defineKernel   Define the kernel function and gradient based on provided
%hyperparameters.
%              
%   K = DEFINEKERNEL(VALUE) builds a kernel cell based on parameters
%   specified by VALUE. K consists of two function handles. The first
%   evaluates the kernel itself whereas the second evaluated the gradient
%   of the kernel at a given point. 
%
%   To specify the type of kernel, use any combination of:
%   'gaussian', GAUSSPARAMS: a two-element vector, the first element
%   determines the coefficient of the Gaussian component whereas the second
%   defines bandwidth (sigma in the paper).
%   'polynomial', POLYPARAMS: a n-element vector for a degree n-1 polynomial
%   kernel. The vector entires correspond to the coefficient of each degree of
%   polynomial. For example if POLYPARAMS = [1, 0, 3] then the kernel takes
%   the form k(x,y) = 1 + 0*(x'*y) + 3*(x'*y).^2;
%
%   This file provides significant flexilibty for automatically defining kernels.
%   However, you can sometimes reduce the overhead time of evaluating the kernel
%   by defining the kernel manually and writing it "inline".
%
%   Reference:
%   Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon and Steven L. Brunton,
%   "Kernel Learning for Robust Dynamic Mode Decomposition: Linear and  Nonlinear 
%   Disambiguation Optimization (LANDO)", arXiv:2106.01510.
%
%See also trainLANDO, linopLANDO, predictLANDO, lorenzExample
%

% Extract arguments
gaussParams = [0,nan];
polyParams = nan;
j = 1;
while j < nargin
   j = j+1;
   v = varargin{j-1};
   if strcmp(v,'gaussian'), j = j+1; gaussParams = varargin{j-1};
   elseif strcmp(v,'polynomial'), j = j+1; polyParams = varargin{j-1};
   else
       error('defineKernel:parseinputs','Unrecognized input')
   end
end
  
% Combine the functions into a cell
k = cell(2,1);
k{1} = @(x,y) evalKernel(x, y, gaussParams, polyParams);
k{2} = @(x,y) evalKernelDeriv(x, y, gaussParams, polyParams);

% Evaluate the kernel
function k = evalKernel(x, y, gaussParams, polyParams)
        polyPart = 0;
    if ~isnan(polyParams) % Polynomial component
        XY = x'*y;
        for jj = 1:numel(polyParams)
            if polyParams(jj)~=0
                polyPart = polyPart + polyParams(jj).*XY.^(jj-1);
            end
        end    
    end
    if isnan(gaussParams(2)) % Gaussian component
        gaussPart = 0;
    else
        gaussPart = gaussParams(1)*gaussKernel(x,y,gaussParams(2));
    end
    k =  polyPart + gaussPart; % Combine the components
end

% Evaluate the gradient of the kernel
function k = evalKernelDeriv(x, y, gaussParams, polyParams)
    polyPart = 0;
    if ~isnan(polyParams) % Polynomial component
        XY = x'*y;
        for jj = 2:numel(polyParams)
            if polyParams(jj)~=0
                polyPart = polyPart + (jj-1).*polyParams(jj).*XY.^(jj-2).*x';
            end
        end    
    end
    if isnan(gaussParams(2)) % Gaussian component
        gaussPart = 0;
    else
        gaussPart = gaussParams(1)/gaussParams(2)^2*gaussKernel(x,y,gaussParams(2)).*(x-y)';
    end
    k =  polyPart + gaussPart; % Combine the components
end

end 

% Evaluate a Gaussian kernel
function K = gaussKernel(X1,X2,Sigma)
    if isnan(Sigma)
        K = 0;
    else
        dim1 = size(X1,2);
        dim2 = size(X2,2);

        norms1 = sum(X1.^2,1);
        norms2 = sum(X2.^2,1);

        mat1 = repmat(norms1',1,dim2);
        mat2 = repmat(norms2,dim1,1);

        distmat = mat1 + mat2 - 2*X1'*X2;	% full distance matrix
        K = exp(-distmat/(2*Sigma^2));
    end
end