%% Learning the Lorenz equation with noise

addpath('../src/'); % Add the source files to the path

% Lorenz's parameters (chaotic)
Sigma = 10; Beta = 8/3; rho = 28;

x0=[-8; 8; 27]; % Initial condition
dt = 1e-3; tspan= dt:dt:50; % Sampling rate and sample times
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,3)); % ode45 options
[~,xClean]=ode45(@(t,x) lorenzFun(t,x,Sigma,Beta,rho),tspan,x0,options);

% Contaminate the data with noise
noiseMag = .05; % Magnitude of noise
xNoisy = xClean + noiseMag*randn(size(xClean))*sqrt(diag(var(xClean)));

%% Smooth the data

% Initialise zeroed smoothed data matrices
dxdt = zeros(numel(tspan)+1,3);
xt = zeros(numel(tspan)+1,3);

% Perform total variation regularised differentation
disp('Started TVRegDiff.')
dxdt(:,1) = TVRegDiff( xNoisy(:,1), 10, .00002, [], 'small', 1e12, dt, 0, 0 );
dxdt(:,2) = TVRegDiff( xNoisy(:,2), 10, .00002, [], 'small', 1e12, dt, 0, 0 );
dxdt(:,3) = TVRegDiff( xNoisy(:,3), 10, .00002, [], 'small', 1e12, dt, 0, 0 );
disp('Finished TVRegDiff.')

% Integrate the derivative of the data to get an approximation for X
xt(:,1) = cumsum(dxdt(:,1))*dt;
xt(:,2) = cumsum(dxdt(:,2))*dt;
xt(:,3) = cumsum(dxdt(:,3))*dt;

% Give X the same mean as the orignal data
xt = xt - mean(xt) + mean(xNoisy);

% Arrange the data into data matrices
X = xt'; Y = dxdt';

%% Begin LANDO procedure

% Rescale the data to improve the condition number
xScl = diag(1./max(abs(X),[],2));

% Shuffle the data to improve the quality of the dictionary
ranp = randperm(size(X,2)); Xr = X(:,ranp); Yr = Y(:,ranp);

% Sparsification parameter. Increase for a sparser dictionary
nu = 1e-6;

% Choose equilibrium base state to linearise about
xBar = [-sqrt(Beta*(rho-1)),-sqrt(Beta*(rho-1)),rho-1]';

% Define three different kernels
quadKernel  = defineKernel('polynomial', [1,1,1]); % Quadratic kernel

% Train LANDO for the three kernels
[quadModel, quadXdic, quadWtilde]  = trainLANDO(Xr, Yr, nu, quadKernel, ...
                                    'display', 'backslash', 'xScl', xScl);
                       
% Dynamic mode decomposition of the learned linear operator at xBar
[quadEVals, quadEVecs, learnedLinop]  = linopLANDO(quadXdic,  quadWtilde,  quadKernel, ...
                            'xScl', xScl, 'nModes', 3, 'xBar', xBar);
                        
% Display true linear operator
trueLinOp = [-Sigma Sigma 0;...
              rho-xBar(3) -1 -xBar(1);...
             xBar(2) xBar(1) -Beta];
fprintf('The true linear operator is: \n\n')         
disp(trueLinOp)

% Display learned linear operator at xBar
fprintf('The learned linear operator is: \n\n')         
disp(learnedLinop)

% Compute reconstructions    
rec   = predictLANDO(quadModel, 50, x0, 'cont', options);    

%% Plots
LW = 'LineWidth'; IN  = 'Interpreter'; LT = 'Latex'; FS = 'FontSize';

% Generate data for plotting purposes only. It's the same as the training
% data but sampled at points adaptively selected by ode45: the point is to
% get "tPlot", which allows us to color the curve nicely
[tPlot,xCPlot]=ode45(@(t,x) lorenzFun(t,x,Sigma,Beta,rho),[0, 50],x0,options);
xNPlot = xCPlot + noiseMag*randn(size(xCPlot))*sqrt(diag(var(xCPlot)));

% Plot the clean data, noisy training data, and reconstructed trajectory
% with the color indicating the integration time-step at each point
f1 = figure(1); clf;
subplot(1,3,1)
color_line3(xCPlot(:,1),xCPlot(:,2),xCPlot(:,3),[0; diff(tPlot)],LW,1);
niceFig; title({'clean data'},IN,LT,FS,15)
subplot(1,3,2)
color_line3(xNPlot(:,1),xNPlot(:,2),xNPlot(:,3),[0; diff(tPlot)],LW,1);
niceFig; title({'noisy', 'training data'},IN,LT,FS,15)
subplot(1,3,3)
color_line3(rec.y(1,:),rec.y(2,:),rec.y(3,:),[0, diff(rec.x)],LW,1);
niceFig; title({'reconstructed', 'tracjectory'},IN,LT,FS,15)
f1.Position(3:4) = [800,300];

function niceFig
% Script that makes the Lorenz figures look nice
axis equal; view(27,16)
axis([-20 20 -30 30 3 48])
ax = gca; ax.TickLabelInterpreter = 'Latex';
ax.FontSize = 10; zticks([0,20,40])
set(gca,'Color','w'); set(gcf,'Color','w')
grid on; 
end