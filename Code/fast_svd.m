function z = fast_svd(prob,X,a,b,r)
% from the known SVD X=USV' to the SVD of low-rank correction X+ab'=[U P]K[V Q]'
% [Matthew Brand, Fast low-rank modifications of the thin singular value decomposition, Linear Algebra and its Applications, 2006]


m = X.U'*a;
p = a - X.U*m;
Ra = norm(p);
P = p/Ra;

n = X.V'*b;
q = b - X.V*n;
Rb = norm(q);
Q = q/Rb;

K = [X.sigma+m*n' Rb*m;Ra*n' Ra*Rb];
[UK,SK,VK] = svd(K);
UU = [X.U P]*UK;
VV = [X.V Q]*VK;
ssigma = diag(SK);

% fprintf('\n%3.2e\n',ssigma)

z.U = UU(:,1:r);
z.V = VV(:,1:r);
z.sigma = ssigma(1:r)+eps;
z = prepx(prob,z);
end