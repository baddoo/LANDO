rng(2);
% Set parameters of problem
tend = 60; dt = 5e-2;
a = -1; b = -1; L = 14*pi; % Parameters of model
nx = 2^10;
xgrid = linspace(-L/2,L/2,nx+1); xgrid(end) = [];

% Set up spin operator
dom = [-L/2 L/2]; x = chebfun('x',dom); 
tspan = (0:dt:tend); nt = numel(tspan);
S = spinop(dom,tspan);
lin = @(u) a*diff(u,2) + b*diff(u,4);
nonlin = @(u) -.5*diff(u.^2);

S.lin = lin;
S.nonlin = nonlin;
nTraj = 20; % Number of trajectories
X = zeros(nx,nt*nTraj); % Initialize data matrices
Y = zeros(nx,nt*nTraj);
data = zeros(nx,nt);
datadt = zeros(nx,nt);

for j = 1:nTraj
    u0 = (randnfun(L/round(2+1.5*rand),dom,'trig'));
    u0 = u0 - sum(u0)/L; % Ensure that the data has zero integral
    S.init = u0;
    u = spin(S,nx,dt,'plot','off');
    lu = lin(u);
    nlu = nonlin(u);
    for k = 1:nt
          data(:,k) = u{k}(xgrid);
          datadt(:,k) = lu{k}(xgrid) + nlu{k}(xgrid); % Calculate the time derivative
    end
    X(:,(1:nt) + (j-1)*nt) = data;
    Y(:,(1:nt) + (j-1)*nt) = datadt;
end