addpath('../src/'); % Add the source files to the path

% Lorenz's parameters (chaotic)
Sigma = 10; Beta = 8/3; rho = 28;

% Define the true linear operator for comparison
trueLinOp = @(xBar) [-Sigma Sigma 0;...
                  rho-xBar(3) -1 -xBar(1);...
                     xBar(2) xBar(1) -Beta];

x0=[-8; 8; 27]; % Initial condition
dt = 1e-3; tspan= dt:dt:10; % Sampling rate and length of training data
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,3)); % Define options for integration
[t,x]=ode45(@(t,x) lorenzFun(t,x,Sigma,Beta,rho),tspan,x0,options); % Generate data
data = [x(:,1) x(:,2) x(:,3)];

% Compute the time-derivative of the data
dxdt = zeros(length(x),3);
for i=1:length(x)
    dxdt(i,:) = lorenzFun(0,x(i,:),Sigma,Beta,rho);
end

X = x'; Y = dxdt'; % Arrange the data into data matrices

%% LANDO procedure

% Rescale the data to improve the condition number
xScl = diag(1./max(abs(X),[],2));

% Shuffle the data to improve the quality of the dictionary
ranp = randperm(size(X,2)); Xr = X(:,ranp); Yr = Y(:,ranp);

% Sparsification parameter. Increase for a sparser dictionary
nu = 1e-6;

% Choose equilibrium base state to linearise about
xBar = [-sqrt(Beta*(rho-1)),-sqrt(Beta*(rho-1)),rho-1]';

% Define three different kernels
linKernel   = defineKernel('polynomial', [0,1]);   % Linear kernel
gaussKernel = defineKernel('gaussian',   [1,1.1]); % Gaussian kernel
quadKernel  = defineKernel('polynomial', [1,1,1]); % Quadratic kernel

% Train LANDO for the three kernels
trainopts = {'display','backslash','xScl',xScl}; % Define training options
[linModel,   linXdic,   linWtilde]   = trainLANDO(Xr, Yr, nu, linKernel,   trainopts{:});
[gaussModel, gaussXdic, gaussWtilde] = trainLANDO(Xr, Yr, nu, gaussKernel, trainopts{:});
[quadModel,  quadXdic,  quadWtilde]  = trainLANDO(Xr, Yr, nu, quadKernel,  trainopts{:});

% Dynamic mode decomposition of the learned linear operator at xBar
linopts = {'xScl',xScl,'nModes',3,'xBar',xBar}; % Define linear operator options
[linEVals,   linEVecs]   = linopLANDO(linXdic,   linWtilde,   linKernel,   linopts{:});
[gaussEVals, gaussEVecs] = linopLANDO(gaussXdic, gaussWtilde, gaussKernel, linopts{:});
[quadEVals,  quadEVecs]  = linopLANDO(quadXdic,  quadWtilde,  quadKernel,  linopts{:});

% Compute reconstructions    
recLin   = predictLANDO(linModel,   10, x0, 'cont', options);    
recGauss = predictLANDO(gaussModel, 10, x0, 'cont', options);
recQuad  = predictLANDO(quadModel,  10, x0, 'cont', options);

% Compute predictions
x0Pred = [10; 14; 10];  % New initial condition
[~,predData] = ode45(@(t,x) lorenzFun(t,x,Sigma,Beta,rho),tspan,x0Pred,options); 
predLin   = predictLANDO(linModel,   10, x0Pred, 'cont', options);    
predGauss = predictLANDO(gaussModel, 10, x0Pred, 'cont', options);
predQuad  = predictLANDO(quadModel,  10, x0Pred, 'cont', options);

%% Plots
LW = 'LineWidth'; IN  = 'Interpreter'; LT = 'Latex'; FS = 'FontSize';
cols = lines;

% Plot reconstructions
f1 = figure(1); subplot(1,4,1)
plot3(data(:,1),data(:,2),data(:,3),'Color',cols(1,:),LW,1);
niceFig; title({'training' 'data'},IN,LT,FS,15);
subplot(1,4,2)
plot3(recLin.y(1,:),recLin.y(2,:),recLin.y(3,:),'Color',cols(2,:),LW,1);
niceFig; title({'linear', 'kernel'},IN,LT,FS,15)
subplot(1,4,3)
plot3(recGauss.y(1,:),recGauss.y(2,:),recGauss.y(3,:),'Color',cols(4,:),LW,1);
niceFig; title({'Gaussian', 'kernel'},IN,LT,FS,15)
subplot(1,4,4)
plot3(recQuad.y(1,:),recQuad.y(2,:),recQuad.y(3,:),'Color',cols(3,:),LW,1);
niceFig; title({'quadratic', 'kernel'},IN,LT,FS,15)
sgtitle('Reconstructed trajectories',IN,LT,FS,20)
f1.Position(3:4) = [800,300];

% Plot predictions
f2 = figure(2); subplot(1,4,1)
plot3(predData(:,1),predData(:,2),predData(:,3),'Color',cols(1,:),LW,1);
niceFig; title({'test', 'data'},IN,LT,FS,15);
subplot(1,4,2)
plot3(predLin.y(1,:),predLin.y(2,:),predLin.y(3,:),'Color',cols(2,:),LW,1);
niceFig; title({'linear', 'kernel'},IN,LT,FS,15)
subplot(1,4,3)
plot3(predGauss.y(1,:),predGauss.y(2,:),predGauss.y(3,:),'Color',cols(4,:),LW,1);
niceFig; title({'Gaussian', 'kernel'},IN,LT,FS,15)
subplot(1,4,4)
plot3(predQuad.y(1,:),predQuad.y(2,:),predQuad.y(3,:),'Color',cols(3,:),LW,1);
niceFig; title({'quadratic', 'kernel'},IN,LT,FS,15);
sgtitle('Predicted trajectories',IN,LT,FS,20)
f2.Position(3:4) = [800,300];

function niceFig
% Script that makes the Lorenz figures look nice
axis equal; view(27,16)
axis([-20 20 -30 30 3 48])
ax = gca; ax.TickLabelInterpreter = 'Latex';
ax.FontSize = 10; zticks([0,20,40])
set(gca,'Color','w'); set(gcf,'Color','w')
grid on; 
end