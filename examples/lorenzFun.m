function dxdt = lorenzFun(t, y, Sigma, Beta, Rho)
%lorenzFun   Evaluates the right-hand-side of the Lorenz system for given
%parameters
%
%   Reference:
%   Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon and Steven L. Brunton,
%   "Kernel Learning for Robust Dynamic Mode Decomposition: Linear and  Nonlinear 
%   Disambiguation Optimization (LANDO)", arXiv:2106.01510.
%

dxdt = [Sigma*(y(2) - y(1));
        y(1)*(Rho - y(3)) - y(2);
        y(1)*y(2) - Beta*y(3)];