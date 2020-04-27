function [w_mat] = make_w(N,M,exact)
if(exact==true)
 w_mat = 1i*pi/4*spdiags([2*ones(N,1),3*ones(N,1),ones(N,1)],[-1,0,1],N,N);
 w_mat(N,1) = 1i*pi/4;
 w_mat(1,N) = 1i*2*pi/4;
else
    w_mat = (1i*(rand(N,M)-0.5))+(1*(rand(N,M)-0.5));
    w_mat = w_mat/norm(w_mat,2);
end
end
