function [x,histout,fail] = LRGeomRRAM(prob, opts, x0)
%%--------------------------------------------------------------------------
% LRGeomRRAM is a Riemannian rank-adaptive method for low-rank matrix completion
%
% Input: prob     = problem instance, see MAKE_PROB.
%        opts     = options, see DEFAULT_OPTS.
%        x0       = starting guess.
%
% Output: x       = solution.
%         histout = iteration history. Each row of histout is
%                   [rel_norm(grad), rel_err_on_omega, relative_change, ...
%                        number of step length reductions, timing, update rank]
%         fail    = flag for failure
%
% -------------------------------------
% Reference:
%   Bin Gao and P.-A. Absil, A Riemannian rank-adaptive method for low-rank matrix completion
% Author: Bin Gao (https://www.gaobin.cc)
%   version 0.1 ... 2020/10
%   version 1.0 ... 2021/03: published on Github (https://github.com/opt-gaobin/RRAM)
%--------------------------------------------------------------------------
%% initialization
% parameters for control the linear approximation in line search
eta = 0.1; % backtracking parameter
gamma = 0.85; % 0 for monotone line-search, 1 for average function vaule
maxiarm = 5; % maxiter for line-search
rhols  = 1e-4;

fail = true; % check convergece (fail = false when it converges)
norm_M_Omega = norm(prob.data); % ||P_{Omega}(A)||_F

reduction_signal = 0;
itc = 1;
% generate initial point
if opts.delta < 1 && prob.r > 2
    [xc,k] = rank_reduction(prob,x0,opts.delta);
    if k < prob.r
        if opts.verbosity > 0; fprintf('--- Truncated the initial point from rank %d to rank %d...\n',prob.r,k); end
        reduction_signal = 1;
    end
else
    xc = x0;
    % update-rank
    k = prob.r;
end


fc = F(prob,xc);
gc = grad(prob,xc);
ip_gc = ip(xc,gc,gc);
% first search-dir is steepest gradient
dir = scaleTxM(gc,-1);
rel_grad = sqrt(ip_gc)/max(1,norm(xc.sigma));

% line-search parameter
Q = 1; Cval = fc;

ithist=zeros(opts.maxit+opts.inner_itr,6);
ithist(1,6) = k;
%% main iteration
timing = tic;
while itc < opts.maxit
    
    % ---------------- fixed-rank optimization ----------------
    tinit = exact_search_onlyTxM(prob, xc,dir);
    t_BB = tinit;
    if opts.verbosity > 0; fprintf('--- Fixed-rank %d optimization...\n',k); end
    for itc_BB = 1:opts.inner_itr
        
        % ----- non-monotone line-search -----
        iarm = 1; % line-search number
        while 1
            xc_new = moveEIG(prob,xc,dir,t_BB);
            
            fc_new = F(prob,xc_new);
            gc_new = grad(prob,xc_new);
            ip_gc_new = ip(xc_new,gc_new,gc_new);
            
            if fc_new <= Cval - t_BB*rhols*ip_gc_new || iarm >= maxiarm
                break;
            end
            t_BB = eta*t_BB; iarm = iarm + 1;
        end
        
        
        rel_grad = sqrt(ip_gc_new)/max(1,norm(xc_new.sigma));
        rel_err_on_omega = sqrt(2*fc_new)/norm_M_Omega;
        reschg = abs(1-sqrt(2*fc_new)/sqrt(2*fc));  % LMARank's detection
        
        % ----- BB stepsize -----
        % Euclidean-BB
        % S = xc_new.U * diag(xc_new.sigma) * xc_new.V' - (xc.U * diag(xc.sigma) * xc.V');
        % Y = xc_new.U*gc_new.M*xc_new.V' + gc_new.Up*xc_new.V' + xc_new.U*gc_new.Vp' - (xc.U*gc.M*xc.V' + gc.Up*xc.V' + xc.U*gc.Vp');
        % SS = norm(S,'fro')^2;
        % YY = norm(Y,'fro')^2;
        % SY = abs(sum(sum(S.*Y)));
        
        % Riemannian-BB (IMAJNA: Iannazzo-Porcelli 2018)
        dir_OldToNew = transpVect(prob,xc,dir,xc_new,1);
        S = scaleTxM(dir_OldToNew,t_BB);
        Y = plusTxM(gc_new, dir_OldToNew, 1, 1);
        SS = ip(xc_new,S,S);
        YY = ip(xc_new,Y,Y);
        SY = abs(ip(xc_new,S,Y));
        
        if mod(itc,2)==0
            t_BB = SS/SY;
        else
            t_BB = SY/YY;
        end
        t_BB = max(min(t_BB, 10^15), 10^(-15));
        
        % ----- update _new to current -----
        gc = gc_new;
        ip_gc = ip_gc_new;
        xc = xc_new;
        fc = fc_new;
        dir = scaleTxM(gc_new,-1);
        
        ithist(itc,1) = rel_grad;
        ithist(itc,2) = rel_err_on_omega;
        ithist(itc,3) = reschg;
        ithist(itc,4) = iarm;
        ithist(itc,5) = toc(timing);
        ithist(itc+1,6) = k;
        itc = itc + 1;
        
        % ----- Test for convergence -----
        if sqrt(2*fc) < opts.abs_f_tol
            if opts.verbosity > 0
                disp('Abs f tol reached.')
            end
            fail = false;
            break;
        end
        if rel_err_on_omega < opts.rel_f_tol
            if opts.verbosity > 0; disp('Relative f tol reached.'); end
            fail = false;
            break;
        end
        
        if rel_grad < opts.rel_grad_tol
            if opts.verbosity > 0; disp('Relative gradient tol reached.'); end
            fail = false;
            break;
        end
        
        % for 'stagnation stopping criterion' after 5 iters
        %reschg = abs(sqrt(2*fc) - sqrt(2*fold)) / max(1,norm_M_Omega);
        if itc_BB > 10 && reschg < opts.rel_tol_change_res
            if opts.verbosity > 0; disp('Iteration stagnated rel_tol_change_res.'); end
            fail = true;
            break;
        end
        
        % --------------------- nonmonotone update --------------------
        Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + fc)/Q;
    end
    
    % ---------------- rank reduction step ----------------
    flag_reduction = 0;
    if ~reduction_signal && opts.delta < 1 && k > 2
        [xc_r,k_c] = rank_reduction(prob,xc,opts.delta);
        if k_c < k
            if opts.verbosity > 0; fprintf('--- Rank reduction from %d to %d...\n',k,k_c); end
            k = k_c;
            flag_reduction = 1;
            reduction_signal = 1;
            
            fc_r = F(prob,xc_r);
            gc_r = grad(prob,xc_r);
            ip_gc_r = ip(xc_r,gc_r,gc_r);
            rel_grad = sqrt(ip_gc_r)/max(1,norm(xc_r.sigma));
            rel_err_on_omega = sqrt(2*fc_r)/norm_M_Omega;
            reschg = abs(1-sqrt(2*fc_r)/sqrt(2*fc) );  % LMARank's detection
            
            % ----- update _new to current -----
            gc = gc_r;
            ip_gc = ip_gc_r;
            xc = xc_r;
            fc = fc_r;
            dir = scaleTxM(gc_r,-1);
            
            ithist(itc,1) = rel_grad;
            ithist(itc,2) = rel_err_on_omega;
            ithist(itc,3) = reschg;
            ithist(itc,4) = 0;
            ithist(itc,5) = toc(timing);
            ithist(itc+1,6) = k;
            itc = itc + 1;
        end
    end
    
    % ---------------- final stop ----------------
    if ~fail || (k == prob.r); break; end
    
    % calculate the approximation of full normal vector to check if there is
    % a need to increase the rank
    flag_increase = 0;
    if ~flag_reduction && k < prob.r
        if rel_grad < 1e-3
            dir_normal = -prob.temp_omega;
        else
            dir_normal = -prob.temp_omega + (xc.U*gc.M*xc.V' + gc.Up*xc.V' + xc.U*gc.Vp');
        end
        [U_n,D_n,V_n] = svds(dir_normal,prob.r-k);
        if norm(D_n,'fro') > opts.increase_eps*sqrt(ip_gc)
            flag_increase = 1;
        end
        % direct computation
        %         [U_n,D_n,V_n] = svds(dir_normal,prob.r-k);
        %         Nks = U_n*D_n*V_n';
        %         if norm(Nks,'fro') > 10*sqrt(ip_gc)
        %             flag_increase = 1;
        %         end
    end
    
    % ---------------- rank-increase correction step ----------------
    r_c = opts.r_c; % rank-increase number
    if ~flag_reduction && flag_increase && (k+r_c) <= prob.r
        % increase rank r_c 
        U_c = U_n(:,1:r_c);
        V_c = V_n(:,1:r_c);
        D_c = D_n(1:r_c,1:r_c);
        
        % exact line search
        tmin = exact_search_onlyNxM(prob, xc, U_c, D_c, V_c);
        
        % ----- Three different ways to increase rank -----
        % -- method-1 -- diagonal correction
        xc_c = rank_increase(prob,xc,U_c,V_c,tmin*D_c,k+r_c);
        
        % -- method-2 -- fast svd doesn't work
        % xc_c = fast_svd(prob,xc,tmin*D_c*U_c,V_c,k+r_c);
        
        % -- method-3 -- direct svd
        % xc_correct = xc.U*diag(xc.sigma)*xc.V' + tmin*(U_c*D_c*V_c');
        % [U,S,V] = svds(xc_correct,k+r_c);
        % xc_c.U = U;
        % xc_c.V = V;
        % xc_c.sigma = diag(S) + eps;
        % xc_c = prepx(prob,xc_c);
        
        if opts.verbosity > 0; fprintf('--- Rank increase from %d to %d...\n',k,length(xc_c.sigma)); end
        k = length(xc_c.sigma);  % set update-rank
        fc_c = F(prob,xc_c);
        gc_c = grad(prob,xc_c);
        ip_gc_c = ip(xc_c,gc_c,gc_c);
        rel_grad = sqrt(ip_gc_c)/max(1,norm(xc_c.sigma));
        rel_err_on_omega = sqrt(2*fc_c)/norm_M_Omega;
        reschg = abs(1-sqrt(2*fc_c)/sqrt(2*fc) );  % LMARank's detection
        
        % ----- update _new to current -----
        gc = gc_c;
        ip_gc = ip_gc_c;
        xc = xc_c;
        fc = fc_c;
        dir = scaleTxM(gc_c,-1);
        
        ithist(itc,1) = rel_grad;
        ithist(itc,2) = rel_err_on_omega;
        ithist(itc,3) = reschg;
        ithist(itc,4) = 0;
        ithist(itc,5) = toc(timing);
        ithist(itc+1,6) = k;
        itc = itc + 1;
        
        flag = false;
    end
    
end

x = xc;
histout=ithist(1:itc,:);
end

