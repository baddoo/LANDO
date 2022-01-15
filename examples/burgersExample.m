%% LANDO on the Burgers' equation
% This example applies LANDO to the Burgers' equation. Note that you need to
% download the Burgers' equation data from the Dropbox link on the Github
% readme. Alternatively, you can generate the data manually using the code
% below. In this case, LANDO learns from discrete-time measurements:
% y_k = x_k+1.
addpath('../src/')
load('burgersData'); % Loads the data -- you need to download it first.
%generateBurgersData; % Generates the data manually

%%
rng(2); % Reset random seed for reproducability
nuV = 1e-4; % Sparsity parameter
ranp = randperm(size(X,2)); % Random permutation of samples
xScl = 1./max(abs(X),[],2); % Rescale data
Xr = X(:,ranp); Yr = Y(:,ranp);

kernel  = defineKernel('polynomial', [1,1,1e-1]); % Quadratic kernel
trainopts = {'display','psinv','xScl',xScl}; % Define training options
[model,  Xdic,  Wtilde]  = trainLANDO(Xr, Yr, nuV, kernel, trainopts{:});
xBar = zeros(nx,1);
linopts = {'xScl',xScl,'nModes',100,'xBar',xBar}; % Define linear operator options
[LANDOeVals, ~, LANDOlinop]   = linopLANDO(Xdic,   Wtilde,   kernel,   linopts{:});

%% Exact DMD eigenvalues
trunc = 100;
[Ux,Sx,Vx] = svds(X,trunc);
Atilde = Ux'*Y*Vx*pinv(Sx);
DMDlinop = Ux*Atilde*Ux';
DMDEvals = eig(Atilde);
[~,idx] = sort(abs(DMDEvals),'descend'); % Sort eigenvalues
DMDEvals = DMDEvals(idx);

%% Plot spectrum
f1 = figure(1);
LW = 'LineWidth'; IN = 'Interpreter'; FS = 'FontSize'; LT = 'Latex';
p1 = scatter(0*(-10:10),(sqrt(nu)*pi*(-10:10)),70,'o',LW,2,'MarkerEdgeColor',.8*[1 1 1]);
hold on
p2 = scatter(real(sqrt(log(DMDEvals)/dt)),imag(sqrt(log(DMDEvals)/dt)),70,'r^',LW,2);
p3 = scatter(real(sqrt(log(LANDOeVals)/dt)),   imag(sqrt(log(LANDOeVals)/dt)),70,'bx',LW,2);
hold off; grid on; box on
axis([-.5,2.5,-2,2])
set(gca,FS,15,'TickLabelInterpreter',LT)
xlabel('$\Re[\sqrt{\lambda_n}]$',IN,LT)
ylabel('$\Im[\sqrt{\lambda_n}]$',IN,LT)
title('learned eigenvalues',IN,LT,FS,20)
legend([p1,p2,p3],{'analytical','DMD','LANDO'},'Location','East',IN,LT)
f1.Position(3:4) = [350,400];

%% Compare linear operators
% This is just a qualitative comparison. Note that you can improve the
% visual performance of DMD here by reducing the rank truncation; however,
% this doesn't improve the calculation of the eigenvalues.

% Form the spectral differentiation operator
DFT = exp((0:nx-1).*(0:nx-1)'*2i*pi/nx); % Discrete Fourier transform matrix
N1 =  floor((nx-1)/2); N21 = (-nx/2)*ones(rem(nx+1,2));
wave1 = [(0:N1)  N21 (-N1:-1)]';
diffVec = (1i*wave1*pi).^2;
TRUElinop = nu*real(1/nx*conj(DFT)*(diffVec.*DFT)); % Exact linear operator

f2 = figure(2);
subplot(1,3,1)
imagesc(TRUElinop)
title('true linear part',IN,LT,FS,15)
axis equal; axis tight; xticks([]); yticks([])
caxis(10*[-1,1])

subplot(1,3,2)
imagesc(LANDOlinop)
title('LANDO linear part',IN,LT,FS,15)
axis equal; axis tight; xticks([]); yticks([])
caxis(.01*[-1,1])
colormap redblue

subplot(1,3,3)
imagesc(DMDlinop)
title('DMD linear part',IN,LT,FS,15)
axis equal; axis tight; xticks([]); yticks([])
caxis(.1*[-1,1])
colormap redblue

f2.Position(3:4) = [700,250];
