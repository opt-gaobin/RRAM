function test_rank_reduction
% This test shows that how the rank reduction strategy works for low rank matrix completion
randn('state',0); rand('state',0);

%% problem generation
m = 1000; n = 1000; % dimensions
k = 10; % rank
% Relative oversampling factor
% OS=1 is minimum, 2 is difficult, 3 is OKish, 4+ is easy.
OS = 3;

% ---------------------------------------
% random factors -- well-condition
L = randn(m, k);
R = randn(n, k);
dof = k*(m+n-k);

% different choices for rank parameter 
rank_reconstruct = [10:1:14];
s = length(rank_reconstruct);

timeCG = cell(4,s);
timeBB = cell(4,s);


for i = 1:s
    % ---------------------------------------
    % make random sampling, problem and initial guess
    samples = floor(OS * dof);
    Omega = make_rand_Omega(m,n,samples);
    prob = make_prob(L,R,Omega,rank_reconstruct(i)); % <- you can choose another rank here
    
    fprintf('------------------------ rank parameter k = %d ------------------------\n',rank_reconstruct(i))
    
    % parameters
    options = default_opts(prob);
    options.verbosity = 1;
    options.delta = 0.1;
    options.maxit = 1000;
    
    % random initial point
    M_omega = randn(m,n);
    [U,S,V] = svds(M_omega, prob.r, 'L');
    U = U(:,1:prob.r);
    S = S(1:prob.r,1:prob.r);
    V = V(:,1:prob.r);
    x0.V = V; x0.sigma = diag(S); x0.U = U;
    x0 = prepx(prob, x0);
    
    fprintf('------ LRGeomCG ------\n')
    t=tic;
    [Xcg,hist] = LRGeomCG_timing(prob,options,x0);
    out_time = toc(t);
    fprintf('running time: %f\n', out_time)
    
    timeCG{1,i} = hist(:,5);
    timeCG{2,i} = hist(:,2);
    timeCG{3,i} = hist(:,1);
    
    fprintf('------ RRAM ------\n')
    tBB=tic;
    [XBB,histBB] = LRGeomRRAM(prob,options,x0);
    out_timeBB = toc(tBB);
    fprintf('running time: %f\n', out_timeBB)
    
    timeBB{1,i} = histBB(:,5);
    timeBB{2,i} = histBB(:,2);
    timeBB{3,i} = histBB(:,1);
    timeBB{4,i} = histBB(:,6);
end


f1_residual = figure(1);
semilogy(...
    timeCG{1,1},timeCG{2,1},'g-.x',...
    timeCG{1,2},timeCG{2,2},'b-.',...
    timeBB{1,1},timeBB{2,1},'r-',...
    timeCG{1,3},timeCG{2,3},'b-.',...
    timeCG{1,4},timeCG{2,4},'b-.',...
    timeBB{1,2},timeBB{2,2},'r-',...
    timeBB{1,3},timeBB{2,3},'r-',...
    timeBB{1,4},timeBB{2,4},'r-',...
    'MarkerSize',10,'linewidth',2)
set(gca,'fontsize',16); %grid on
ylabel('relative residual','fontsize',16);
xlabel('time (s)')
title(['m=',num2str(m),', n=',num2str(n),', OS=',num2str(OS)]);
legend('LRGeomCG (k=10)', 'LRGeomCG (k>10)','RRAM-RBB')%,'Location','south')

f2_gradient = figure(2);
semilogy(...
    timeCG{1,1},timeCG{3,1},'g-.x',...
    timeCG{1,2},timeCG{3,2},'b-.',...
    timeBB{1,1},timeBB{3,1},'r-',...
    timeCG{1,3},timeCG{3,3},'b-.',...
    timeCG{1,4},timeCG{3,4},'b-.',...
    timeBB{1,2},timeBB{3,2},'r-',...
    timeBB{1,3},timeBB{3,3},'r-',...
    timeBB{1,4},timeBB{3,4},'r-',...
    'MarkerSize',10,'linewidth',2)
set(gca,'fontsize',16); %grid on
ylabel('relative gradient','fontsize',16);
xlabel('time (s)')
title(['m=',num2str(m),', n=',num2str(n),', OS=',num2str(OS)]);
legend('LRGeomCG (k=10)', 'LRGeomCG (k>10)','RRAM-RBB')%,'Location','south')