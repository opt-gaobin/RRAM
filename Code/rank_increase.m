function z = rank_increase(prob,X,a,b,d,r)
% from the known SVD X=USV' to the SVD of low-rank correction X+a*d*b'

%% method-1: very orthogonal
U = [X.U a]; [Qu,Ru] = qr(U,0);
V = [X.V b]; [Qv,Rv] = qr(V,0);
S = diag([X.sigma; diag(d)]);

K = Ru*S*Rv';
[UK,SK,VK] = svd(K);
UU = Qu*UK;
VV = Qv*VK;
ssigma = diag(SK);

z.U = UU(:,1:r);
z.V = VV(:,1:r);
z.sigma = ssigma(1:r)+eps;
z = prepx(prob,z);
%% method-2: sort directly
% U = [X.U a];
% V = [X.V b];
% [ssigma,I] = sort([X.sigma; diag(d)],'descend');
% 
% z.U = U(:,I);
% z.V = V(:,I);
% z.sigma = ssigma + eps;
% z = prepx(prob,z);
end