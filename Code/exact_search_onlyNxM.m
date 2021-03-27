function tmin = exact_search_onlyNxM(prob, x, W, D, Y)

% Exact line search in the direction of dir on the normal space of x

% x, current point

% dir is search direction
%
% returns: tmin

e_omega = x.err;

dir_omega = partXY((W*D)',Y', prob.Omega_i, prob.Omega_j, prob.m);


% norm is f(t) = 0.5*||e+t*d||_F^2
% minimize analytically
% polynomial df/dt = a0+t*a1
a0 = dir_omega*e_omega;
a1 = dir_omega*dir_omega';
tmin = abs(-a0/a1);

