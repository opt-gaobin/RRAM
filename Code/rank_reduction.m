function [z,r] = rank_reduction(prob,X,delta)
%
%  This function truncates the matrix X by the best
%  rank-r approximation with the threshold delta.
%

% reduction flag
flag = 0;

% relative error of singular values
rel_sv = (X.sigma(1:end-1) - X.sigma(2:end))./X.sigma(1:end-1);

% find the largest gap
[maxgap,I] = max(rel_sv);
if maxgap <= delta
    flag = 0;
else
    r = I;
    flag = 1;
end

% reduction step
if flag == 0
    r = length(X.sigma);
    z = X;
else
    z.U = X.U(:,1:r);
    z.V = X.V(:,1:r);
    z.sigma = X.sigma(1:r);
    
    z = prepx(prob,z);
end