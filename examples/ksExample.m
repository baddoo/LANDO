%% LANDO on the Kuramoto-Sivashinsky (KS) equation
% This example applies LANDO to the KS equation. Note that you need to
% download the KS equation data from the Dropbox link on the Github
% readme. Alternatively, you can generate the data manually using the code
% below. In this case, LANDO learns from continuous-time measurements:
% y_k = \dot{x}_k+1.
%
% Note that LANDO takes about one minute to learn the model due to the
% large number of samples.
%
addpath('../src/')
load('ksData'); % Loads the data -- you need to download it first.
%generateKSData; % Generates the data manually

%% Plot the training data
figure(1)
LW = 'LineWidth'; IN = 'Interpreter'; FS = 'FontSize'; LT = 'Latex';
locs = 1:nt;
surf(X(:,locs))
shading interp
colormap gray; axis tight; axis off
view([-45,55]); shading interp
camlight(25,0); lighting gouraud
title({'training data', '(one trajectory out of 20)'},IN,LT,FS,20)

%% Perform LANDO -- it takes around one minute to learn the model
rng(2); % Reset random seed for reproducability
nuV = 1e-5; % Sparsity parameter
ranp = randperm(size(X,2)); % Random permutation of samples
xScl = 1./max(abs(X),[],2); % Rescale data
Xr = X(:,ranp); Yr = Y(:,ranp);

kernel  = defineKernel('polynomial', [0,1,1e-1]); % Quadratic kernel
trainopts = {'display','psinv','xScl',xScl}; % Define training options
[model,  Xdic,  Wtilde]  = trainLANDO(Xr, Yr, nuV, kernel, trainopts{:});
xBar = zeros(nx,1);
linopts = {'xScl',xScl,'nModes',100,'xBar',xBar}; % Define linear operator options
[LANDOeVals, LANDOeVecs, LANDOlinop]   = linopLANDO(Xdic,   Wtilde,   kernel,   linopts{:});
magProj = (vecnorm(LANDOeVecs'*X,2,2)).^2; % Project eigenvectors onto data to get marker size
markSize = min([magProj,1e5*ones(size(magProj))],[],2)/500;
%% Perform exact DMD
trunc = 100;
[Ux,Sx,Vx] = svds(X,trunc);
Atilde = Ux'*Y*Vx*pinv(Sx);
DMDlinop = Ux*Atilde*Ux';
DMDEvals = eig(Atilde);
[~,idx] = sort(abs(DMDEvals),'descend'); % Sort eigenvalues
DMDEvals = DMDEvals(idx);

%% Plot the eigenvalues
figure(2)
scatter(real(DMDEvals),imag(DMDEvals),10,'r^',LW,2,'MarkerEdgeAlpha',.3);
hold on
n = (-35:35);
exEvals = (a*(2i*pi*n/L).^2 + b*(2i*pi*n/L).^4) + 1i*1e-15;
scatter(real(exEvals),imag(exEvals),50,'o',LW,2,'MarkerEdgeColor',.5*[1 1 1]);
scatter(real(LANDOeVals),imag(LANDOeVals),markSize,'x','MarkerEdgeColor','b',LW,2);
hold off; grid on; axis equal
xlabel('$\Re[{\lambda_n}]$',IN,LT)
ylabel('$\Im[{\lambda_n}]$',IN,LT)
set(gca,FS,15,'TickLabelInterpreter',LT)
axis([-7,1,-1,1])
box on
title('learned eigenvalues',IN,LT,FS,20)

figure(3)
p1=scatter(real(LANDOeVals),imag(LANDOeVals),markSize,'x','MarkerEdgeColor','b',LW,2);
hold on
p2 = plot(exEvals,'o',LW,2,'Color',.5*[1 1 1]);
hold off; grid on; axis equal
xlabel('$\Re[{\lambda_n}]$',IN,LT)
ylabel('$\Im[{\lambda_n}]$',IN,LT)
set(gca,FS,15,'TickLabelInterpreter',LT)
axis([-.05,.3,-.0437,.0437])
box on
title('learned eigenvalues (close-up)',IN,LT,FS,20)

%% Plot linear operators
% Form spectral differentiation matrices
DFT = exp((0:nx-1).*(0:nx-1)'*2i*pi/nx); % Discrete Fourier transform matrix
m1 = 2; m2 = 4;
N1 =  floor((nx-1)/2);
N21 = (-nx/2)*rem(m1+1,2)*ones(rem(nx+1,2));
N22 = (-nx/2)*rem(m2+1,2)*ones(rem(nx+1,2));
wave1 = [(0:N1)  N21 (-N1:-1)]';
wave2 = [(0:N1)  N22 (-N1:-1)]';
diffVec = (1i*wave1*2*pi/L).^m1 + (1i*wave2*2*pi/L).^m2;
TRUElinop = -real(1/nx*conj(DFT)*(diffVec.*DFT));

f2 = figure(5);
subplot(1,3,1)
imagesc(TRUElinop)
title('true linear part',IN,LT,FS,15)
axis equal; axis tight; xticks([]); yticks([])
caxis(1000*[-1,1])

subplot(1,3,2)
imagesc(LANDOlinop)
title('LANDO linear part',IN,LT,FS,15)
axis equal; axis tight; xticks([]); yticks([])
caxis(.1*[-1,1])
colormap redblue

subplot(1,3,3)
imagesc(DMDlinop)
title('DMD linear part',IN,LT,FS,15)
axis equal; axis tight; xticks([]); yticks([])
caxis(100*[-1,1])
colormap redblue

f2.Position(3:4) = [700,250];

%% Plot the true forcing and the forcing calculated by LANDO
figure(4)
LANDOforcing = model(X(:,locs)) - LANDOlinop*X(:,locs);
TRUEforcing = Y(:,locs) - TRUElinop*X(:,locs);
subplot(1,2,1)
pcolor(tspan,xgrid,TRUEforcing)
shading interp; caxis(2*[-1,1])
colormap gray; axis tight;
title({'true forcing'},IN,LT,FS,20)
xlabel('$t$',IN,LT,FS,25); ylabel('$x$',IN,LT,FS,25)
set(gca,FS,15,'TickLabelInterpreter',LT)

subplot(1,2,2)
pcolor(tspan,xgrid,LANDOforcing)
shading interp; caxis(2*[-1,1])
colormap gray; axis tight;
title({'LANDO forcing'},IN,LT,FS,20)
xlabel('$t$',IN,LT); ylabel('$x$',IN,LT)
set(gca,FS,15,'TickLabelInterpreter',LT)