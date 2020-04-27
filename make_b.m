function [b_vec] = make_b(M,exact)
if(exact==true)
   b_vec = 1i*pi/4*ones(M,1);
else
    b_vec = (rand(M,1)-0.5)+(1i*(rand(M,1)-0.5));
    b_vec = b_vec/norm(b_vec,2);
end
end
