function test_fixed_rank
% This test is for the comparision on fixed-rank optimization methods.
randn('state',0); rand('state',0);

%% problem generation
m = 5000; n = 5000; % dimensions
k = 20; % rank
% Relative oversampling factor 
% OS=1 is minimum, 2 is difficult, 3 is OKish, 4+ is easy.
OS = 3;

% ---------------------------------------
% random factors 1 -- ill-condition
% L0 = randn(m, k); [L0,~] = qr(L0,0);
% R = randn(n, k); [R,~] = qr(R,0);
% S = diag(logspace(0,-15,k)); L = L0*S;
% dof = k*(m+n-k);
% rank_reconstruct = 20;

% L0 = randn(m, k); [L0,~] = qr(L0,0);
% R = randn(n, k); [R,~] = qr(R,0);
% S = diag([linspace(1,2,floor(k/2)) linspace(20,30,floor(k/2))]); L = L0*S;
% % S = diag([linspace(1,2,floor(k/3)) linspace(20,30,floor(k/3)) linspace(100,101,ceil(k/3))]); L = L0*S;
% dof = k*(m+n-k);
% rank_reconstruct = 20;

% random factors 2 -- well-condition
L = randn(m, k); 
R = randn(n, k); 
dof = k*(m+n-k);
rank_reconstruct = k;

% ---------------------------------------
% make random sampling, problem and initial guess
samples = floor(OS * dof);
Omega = make_rand_Omega(m,n,samples);
prob = make_prob(L,R,Omega,rank_reconstruct); % <- you can choose another rank here

%% parameters and initialization
options = default_opts(prob);
options.verbosity = 1;
options.delta = 1;
options.maxit = 3000;
options.inner_itr = 1000; 
options.rel_grad_tol = 0;

% ---------------------------------------
% initial point 1: low-rank approximation of the data matrix 
x0 = make_start_x(prob);

% initial point 2: random
% M_omega = randn(m,n);
% [U,S,V] = svds(M_omega, prob.r, 'L');
% U = U(:,1:prob.r);
% S = S(1:prob.r,1:prob.r);
% V = V(:,1:prob.r);
% x0.V = V;
% x0.sigma = diag(S);
% x0.U = U;
% x0 = prepx(prob, x0);

%% call solvers
fprintf('------------------------ LRGeomCG ------------------------\n')
t=tic;
[Xcg,hist] = LRGeomCG_timing(prob,options,x0);
out_time = toc(t);
% Xcg = Xcg.U * diag(Xcg.sigma) * Xcg.V';
fprintf('running time: %f\n', out_time)

fprintf('------------------------ Riemannian BB ------------------------\n')
tBB=tic;
[XBB,histBB] = LRGeomRRAM(prob,options,x0);
out_timeBB = toc(tBB);
% XBB = XBB.U * diag(XBB.sigma) * XBB.V';
fprintf('running time: %f\n', out_timeBB)


%% plot errors
f = figure(1);
semilogy(hist(:,5),hist(:,1),'b-.',...
    hist(:,5),hist(:,2),'b-.x',...
    histBB(:,5),histBB(:,1),'r:',...
    histBB(:,5),histBB(:,2),'r-o',...
    'MarkerSize',5,'linewidth',1.5)
set(gca,'fontsize',16); 
xlabel('time (s)')
title(['m=',num2str(m),', n=',num2str(n),', k=',num2str(rank_reconstruct),', OS=',num2str(OS)]);
legend('LRGeomCG (gradient)','LRGeomCG (residual)','RBB (gradient)','RBB (residual)')
