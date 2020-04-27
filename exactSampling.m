function [E, GaSGD, GbSGD, GwSGD, GaLinear, GbLinear, GwLinear] = exactSampling(a_vec, b_vec, w_mat, hamiltonian, g, pbc,k)
[M,N] = size(w_mat); %Determines N,M based on the size of w_mat

%Creates zero matrices for starting points
o_w = zeros(M*N,1);
o_w_e = zeros(M*N,1);
o_a = zeros(N,1);
o_a_e = zeros(N,1);
o_b = zeros(M,1);
o_b_e = zeros(M,1);
Sa=zeros(N,1);
Sb=zeros(M,1);
Sw=zeros(N*M,1);
psi=zeros(2^N,1);

for p=1:2^N
    s=dec2spin(p,N)'; %Converts the current number to an array of 1 and -1 that represents the spin states
    psi(p)=waveExact(a_vec,b_vec,w_mat,s); %Calculates the wave function for the spin and stores it
end

psi=psi/norm(psi);
[~,H]=theoretical_energy_calc(N,g,pbc,hamiltonian); %Grabs the Hamiltonian
E=real(psi'*H*psi)
eps=(H*psi); % Shortcut step since when this is used a psi is also divided by so we just use one of the psi instead of both

for p=1:2^N
    s=dec2spin(p,N)';
    psiSquared=psi(p)*psi(p)';
    tanhw=(tanh(b_vec+w_mat*s));
    
    new_o_a=s;
    new_o_b=tanhw;
    new_o_w=tanhw*s';
    reshaped_new_o_w=reshape(new_o_w,[N*M,1]); %This gets reshapes since o_w is a matrix and some of the steps only make sense for an array
    
    o_a=o_a+psiSquared*new_o_a;
    o_b=o_b+psiSquared*(new_o_b);
    o_w=o_w+psiSquared*(reshaped_new_o_w);
    
    o_a_e=o_a_e+conj(psi(p)*new_o_a)*eps(p);
    o_b_e=o_b_e+conj(psi(p)*(new_o_b))*eps(p);
    o_w_e=o_w_e+conj(psi(p)*(reshaped_new_o_w))*eps(p);
    
    Sa = Sa + psiSquared*conj(new_o_a)*(transpose(new_o_a));
    Sb = Sb + psiSquared*conj(new_o_b)*(transpose(new_o_b));
    Sw = Sw + psiSquared*conj(reshaped_new_o_w)*(transpose(reshaped_new_o_w));
    
end

%Adds the final parts of the S matrices. With the rest being calculated in
%the for loop
o_w_r = reshape(o_w,[N*M,1]);
Sa = Sa- (conj(o_a)*(transpose(o_a)));
Sb = Sb-(conj(o_b)*(transpose(o_b)));
Sw = Sw -(conj(o_w_r)*(transpose(o_w_r)));

% This steps helps makes sure the S matrices will have an inverse
lambda=max(100*.9^k,.0001);
Sa=Sa+diag(diag(Sa))*lambda;
Sb=Sb+diag(diag(Sb))*lambda;
Sw=Sw+diag(diag(Sw))*lambda;

% Gradient calculation for linear
GaLinear = (o_a_e-(E)*conj(o_a));
GbLinear = (o_b_e-(E)*conj(o_b));
GwLinear = (o_w_e-(E)*conj(o_w));
GwLinear=reshape(GwLinear,[N,M]);

%Extra step makes this the SGD gradient
GaSGD=pinv(Sa)*GaLinear
GbSGD=pinv(Sb)*GbLinear
GwSGD=pinv(Sw)*reshape(GwLinear,[N*M,1])
GwSGD= reshape(GwSGD,[M,N]);

end
