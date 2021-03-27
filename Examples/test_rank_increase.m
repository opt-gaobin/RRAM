function test_rank_increase
% This test is for a ill-conditioned low-rank matrix completion problem.
% It illustrates that the rank-adaptive framework significantly improves
% the performance.
randn('state',0); rand('state',0);

%% problem generation
m = 1000; n = 1000; % dimensions
k = 20; % rank
% Relative oversampling factor 
% OS=1 is minimum, 2 is difficult, 3 is OKish, 4+ is easy.
OS = 3;

% ---------------------------------------
% random factors -- ill-condition
L0 = randn(m, k); [L0,~] = qr(L0,0);
R = randn(n, k); [R,~] = qr(R,0);
S = diag(logspace(0,-15,k)); L = L0*S;
dof = k*(m+n-k);
rank_reconstruct = 20;

% ---------------------------------------
% make random sampling, problem and initial guess
samples = floor(OS * dof);
Omega = make_rand_Omega(m,n,samples);
prob = make_prob(L,R,Omega,rank_reconstruct); % <- you can choose another rank here

%% parameters and initialization
options = default_opts(prob);
options.verbosity = 1;
options.maxit = 1000;
options.rel_grad_tol = 0;
x0 = make_start_x(prob);

%% call solvers
fprintf('------------------------ LRGeomCG ------------------------\n')
t=tic;
[Xcg,hist] = LRGeomCG_timing(prob,options,x0);
out_time = toc(t);
% Xcg = Xcg.U * diag(Xcg.sigma) * Xcg.V';
fprintf('running time: %f\n', out_time)

fprintf('------------------------ RRAM ------------------------\n')
tBB=tic;
options.maxit = 200;
[XBB,histBB] = LRGeomRRAM(prob,options,x0);
out_timeBB = toc(tBB);
% XBB = XBB.U * diag(XBB.sigma) * XBB.V';
fprintf('running time: %f\n', out_timeBB)


f1 = figure(1);
semilogy(hist(:,5),hist(:,1),'b-.',...
    hist(:,5),hist(:,2),'b-.x',...
    histBB(:,5),histBB(:,1),'r:',...
    histBB(:,5),histBB(:,2),'r-o',...
    'MarkerSize',5,'linewidth',1.5)
set(gca,'fontsize',16); 
xlabel('time (s)')
title(['m=',num2str(m),', n=',num2str(n),', k=',num2str(rank_reconstruct),', OS=',num2str(OS)]);
legend('LRGeomCG (gradient)','LRGeomCG (residual)','RRAM-RBB (gradient)','RRAM-RBB (residual)','Location','southeast')