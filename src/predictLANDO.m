function out = predictLANDO(model, tend, x0, type, options)
%predictLANDO   Form predictions based on a LANDO model.
%              
%   out = PREDICTLANDO(MODEL, TEND, X0, TYPE, OPTIONS) performs a prediction of a
%   LANDO model (MODEL) for initial condition X0 up to time TEND. TYPE
%   determines whether the model is defined for discrete time ('disc') or
%   continuous time ('cont'). The OPTIONS should be set using odeset to
%   refine the integration used in ode45.
%
%   Reference:
%   Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon and Steven L. Brunton,
%   "Kernel Learning for Robust Dynamic Mode Decomposition: Linear and  Nonlinear 
%   Disambiguation Optimization (LANDO)", arXiv:2106.01510.
%
%See also trainLANDO, linopLANDO, defineKernel, lorenzExample
%

if strcmp(type,'disc') % Discrete-time integration
    
    out = zeros(numel(x0),tend);
    out(:,1) = x0;
    for t = 1:tend-1
        out(:,t+1) = model(out(:,t));
    end
        
elseif strcmp(type,'cont') % Continuous-time integration
    out = ode45(@(t,x) model(x),[0,tend],x0,options);
end

end