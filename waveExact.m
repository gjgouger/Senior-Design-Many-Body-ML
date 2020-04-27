function [psi, otan]=waveExact(a_vec,b_vec,w_mat,s)
psi = prod (2*cosh(b_vec+w_mat*s)) * exp(a_vec.'*s);
otan=tanh(b_vec+w_mat*s);
end