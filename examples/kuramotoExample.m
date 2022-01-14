%% LANDO on the Kuramoto oscillator system
% This example applies LANDO to the Kuramoto oscillator system to learn the 
% fundamental frequencies. Note that you need to
% download the simulation data from the Dropbox link on the Github
% readme. Alternatively, you can generate the data manually using the code
% below.

addpath('../src/'); % Add the source files to the path

nt = 1e3; % Number of samples
nOsc = 2e3; % Number of oscillators
tspan = linspace(0,20,nt);

%% Load the data
load('kuramotoData'); % You need to download the data here first

% You can also generate the data yourself, or change the parameters by
% commenting out the text below
%
% rng(12);
% omegaTrue = 10*sort(rand(nOsc,1)); % True natural frequencies
% h =  2; % Forcing amplitude
% kappa = 15*ones(nOsc)+sqrt(5)*randn(nOsc); % Random coupling matrix
% theta0 = 2*pi*rand(nOsc,1); % Initial condition
% fprintf('Generating data... \n')
% Y = zeros(nOsc,nt);
% [t,theta] = ode45(@(t,x) kuramoto(t,x,omegaTrue,kappa,h),tspan,theta0);
% X = theta';
% for j = 1:nt
% Y(:,j) = kuramoto(0,X(:,j),omegaTrue,kappa,h);
% end
% fprintf('Finished generating data... \n')
%% Plot the data
figure(1)
IN = 'Interpreter'; LT = 'Latex'; FS = 'FontSize'; LW = 'LineWidth';
cols = brighten(hsv(nOsc),0);
[~,s] = sort(X(:,1)); Xp = X(s,:); % Sort the trajectories by initial condition
% Plot every 20th trajectory, each with a different color
for jj = 1:20:nOsc
plot3(tspan,cos(Xp(jj,1:nt)),sin(Xp(jj,1:nt)),'Color',cols(jj,:),'LineWidth',1);
hold on
end
plot3(tspan,0*tspan,0*tspan,'k','LineWidth',3) % Block the black horizontal line
th = linspace(0,2*pi,nOsc);  patch(0*th,cos(th),sin(th),th,'LineWidth',5,'FaceColor','none','EdgeColor','interp')
colormap(brighten(hsv,0)); title('training data',IN,LT,FS,20);
axis([0,2,-1.2,1.2,-1.2,1.2]); view([-60 10])
xlabel('$t$',IN,LT); ylabel('$\cos(\theta)$',IN,LT); zlabel('$\sin(\theta)$',IN,LT);
hold off; set(gcf,'color','w'); pbaspect([4 1 1]); grid on; box on
set(gca,'TickLabelInterpreter','Latex',FS,20); xticks([0,.5,1,1.5,2])
drawnow;
%%
% Shuffle the data to improve the quality of the dictionary
ranp = randperm(size(X,2)); Xr = X(:,ranp); Yr = Y(:,ranp);

% Sparsification parameter. Increase for a sparser dictionary
nu = 1e-4;

% Define quadratic kernel
quadKernel  = defineKernel('polynomial', [2,2,2e-2]);

% Change variable of kernel to sines and cosines. Note that, if you want
% the spectrum, you'll need to define the gradient of this new kernel.
trigVar = @(x) [sin(x); cos(x)];
trigKernel = cell(1);
trigKernel{1} = @(x,y) quadKernel{1}(trigVar(x), trigVar(y));

% Train the LANDO model
disp('Training the model...')
model = trainLANDO(Xr, Yr, nu, trigKernel, 'display', 'psinv', 1);

% Extract the natural frequencies
omegaApprox = model(zeros(nOsc,1));

%% Compute the normalised error
err = norm((omegaTrue - omegaApprox)./(omegaTrue))/nOsc;

fprintf('The error in the learned natural frequencies is %4.3f%%. \n', 100*err);

%% Plot the learned and true natural frequencies
figure(2)
loc = 1:100:nOsc; % Select every 100th natural frequency
p1 = scatter(loc,omegaTrue(loc),100,'o',LW,2,'MarkerEdgeColor',[1 1 1]*.5); % Ground truth
hold on;
p2 = scatter(loc,omegaApprox(loc),100,'x',LW,2,'MarkerEdgeColor',[0 0 1]); % Learned frequencies
% Make the plot look nice
hold off; grid on; box on; set(gca,'TickLabelInterpreter','Latex',FS,15)
axis tight; xlim([-10,nOsc]);  ylim([-.1,10.1]); set(gcf,'color','w'); 
legend([p1,p2],{'true frequences','learned frequencies'},IN,LT,FS,20,'Location','SouthEast')
xlabel('oscillator number',IN,LT,FS,20); ylabel('natural frequency',IN,LT,FS,20)

%% ODE for generating data
function TD = kuramoto(t,theta,omega,kappa,h)
       n = length(theta);
       TD = omega + 1/n*sum(kappa.*sin(theta-theta'),1)' + h*sin(theta);
end