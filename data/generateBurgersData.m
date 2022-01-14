rng(2);
tend = 10; dt = 1e-2; % Set the time step and length
tspan = 0:dt:tend; % Define time vector
nt = numel(tspan);
dom = [-1 1]; % Set the domain size
nu  = .01; % Set the nu parameter
S = spinop(dom,tspan); % Setup spin operator
lin = @(u) nu*diff(u,2);
nonlin = @(u) -.5*diff(u.^2);
S.lin = lin; S.nonlin = nonlin;
nTraj = 10; % Number of trajectories
nx = 2^10; % Number of grid points
xgrid = linspace(-1,1,nx+1); xgrid(end) = [];

% Initialize data vectors
data = zeros([nx,nt]);
X  = zeros([nx,(nt-1)*nTraj]);
Y  = zeros([nx,(nt-1)*nTraj]);

x = chebfun('x',[-1,1]);

for j = 1:nTraj
    S.init = (2*rand*sech(5*sin(pi*(x-2*rand)/2)).^2 + rand*sech(5*sin(pi*(x-2*rand)/2)).^2);
    u = spin(S,nx,0.1*dt,'plot','off');
    for k = 1:nt
            data(:,k) = u{k}(xgrid);
    end
    X(:,(1:nt-1) + (j-1)*(nt-1)) = data(:,1:end-1);
    Y(:,(1:nt-1) + (j-1)*(nt-1)) = data(:,2:end);
end