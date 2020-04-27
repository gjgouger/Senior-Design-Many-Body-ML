function [a_vec] = make_a(N,exact)
if (exact==true)
    a_vec = zeros(N,1);
else
    a_vec = (rand(N,1)-0.5)+(1i*(rand(N,1)-0.5));
    a_vec = a_vec/norm(a_vec,2);
end
end
