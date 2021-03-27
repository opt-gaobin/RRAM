function opts = default_opts(prob)

opts.maxit = 500;

% Tolerance on the Riemannian gradient of the objective function
opts.abs_grad_tol = 0;
opts.rel_grad_tol = 1e-12;

% Tolerance on the l_2 error on the sampling set Omega
opts.abs_f_tol = 0;
opts.rel_f_tol = 1e-12;

% Tolerance for detection of stagnation. 
opts.stagnation_detection = false;  % <--- cheap test but does take some time
opts.rel_tol_change_x = 1e-12;
opts.rel_tol_change_res = 1e-4;

% Verbosity 2 is very chatty.
opts.verbosity = 1;

% rank-related parameters 
opts.delta = 0.1; % threshold for rank reduction
opts.inner_itr = 100; % inner maximum iteration number
opts.r_c = 1; % rank increase number
opts.increase_eps = 10; % control rank increase
