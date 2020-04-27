
close all
clc; clear;
hamil=@tfim %Sets the hamiltonian
exact=true; %True if you want to use the exact a,b, and w
K=1; %Number of iterations
gammaSGD=0.03;
gammaLinear=.03;
N=4; M=4; g=3; pbc=1;
E=zeros(K,1);
Ecalc=zeros(K,1);
GaStoredSGD=zeros(K,1);
GbStoredSGD=zeros(K,1);
GwStoredSGD=zeros(K,1);

GaStoredLinear=zeros(K,1);
GbStoredLinear=zeros(K,1);
GwStoredLinear=zeros(K,1);



%Creates the a,b and w for SGD
w_mat_sgd = make_w(M,N,exact);
a_vec_sgd = make_a(N,exact);
b_vec_sgd = make_b(M,exact);

%Sets the linear descent a,b,w variables to the same as SGD
w_mat_linear = w_mat_sgd;
a_vec_linear = a_vec_sgd;
b_vec_linear = b_vec_sgd;





for k=1:K
    
    % For SGD
    [E(k),Ga_SGD, Gb_SGD, Gw_SGD, ~, ~, ~] = exactSampling(a_vec_sgd, b_vec_sgd, w_mat_sgd, hamil, g, pbc,k);
    a_vec_sgd = a_vec_sgd - gammaSGD*Ga_SGD;
    b_vec_sgd = b_vec_sgd - gammaSGD*Gb_SGD;
    w_mat_sgd = w_mat_sgd - gammaSGD*Gw_SGD;
    %Stores the values for graphing
    GaStoredSGD(k)=sum(Ga_SGD);
    GbStoredSGD(k)=sum(Gb_SGD);
    GwStoredSGD(k)=sum(sum(Gw_SGD));
    
    % For Linear
    [Ecalc(k),~, ~, ~, Ga_Linear, Gb_Linear, Gw_Linear] = exactSampling(a_vec_linear, b_vec_linear, w_mat_linear, hamil, g, pbc,k);
    a_vec_linear = a_vec_linear - gammaLinear*Ga_Linear;
    b_vec_linear = b_vec_linear - gammaLinear*Gb_Linear;
    w_mat_linear = w_mat_linear - gammaLinear*Gw_Linear;
    %Stores the values for graphing
    GaStoredLinear(k)=sum(Ga_Linear);
    GbStoredLinear(k)=sum(Gb_Linear);
    GwStoredLinear(k)=sum(sum(Gw_Linear));
    
    
    
end

[theoEnergy,~]=theoretical_energy_calc(N,g,pbc,hamil) %Grabs the theoretical Energy

%% Plot for SGD
figure;
%Plots the Energy
subplot(4,2,1)
plot(1:size(Ecalc),theoEnergy*ones(size(Ecalc)),'-g')
hold on
plot(E,'r');
title("Energy using SGD",'LineWidth',2.0)
hold off;

%Plots the gradient of a
subplot(4,2,3)
plot(real(GaStoredSGD),'r','LineWidth',2.0)
hold on;
plot(imag(GaStoredSGD),'b','LineWidth',2.0)
title("Gradient of a")
legend("real","imag")
hold off

%Plots the gradient of b
subplot(4,2,5)
plot(real(GbStoredSGD),'r','LineWidth',2.0)
hold on
plot(imag(GbStoredSGD),'b','LineWidth',2.0)
legend("real","imag")
title("Gradient of b")
hold off

%Plots the gradient of w
subplot(4,2,7)
plot(real(GwStoredSGD),'r','LineWidth',2.0)
hold on
plot(imag(GwStoredSGD),'b','LineWidth',2.0)
legend("real","imag")
title("Gradient of w")
hold off

%% Plot for Linear

%Plots the Energy
subplot(4,2,2)
plot(1:size(Ecalc),theoEnergy*ones(size(Ecalc)),'-g')
hold on
title("Energy using Linear",'LineWidth',2.0)
plot(Ecalc,'r');
hold off

%Plots the gradient of a
subplot(4,2,4)
plot(real(GaStoredLinear),'r','LineWidth',2.0)
hold on
plot(imag(GaStoredLinear),'b','LineWidth',2.0)
title("Gradient of a")
legend("real","imag")
hold off

%Plots the gradient of b
subplot(4,2,6)
plot(real(GbStoredLinear),'r','LineWidth',2.0)
hold on
plot(imag(GbStoredLinear),'b','LineWidth',2.0)
legend("real","imag")
title("Gradient of b")
hold off

%Plots the gradient of w
subplot(4,2,8)
plot(real(GwStoredLinear),'r','LineWidth',2.0)
hold on
plot(imag(GwStoredLinear),'b','LineWidth',2.0)
legend("real","imag")
title("Gradient of w")
hold off



