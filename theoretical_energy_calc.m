function [E,h]=theoretical_energy_calc(N,g,pbc,hamil)
sigmaZ=[1,0;0,-1]; %Pauli matrix for Z
sigmaX=[0,1;1,0];  %Puali matrix for X
if (strcmp(func2str(hamil),'tfim')) %For the tfim hamiltonian
    hE=zeros(2^N);
    hG=zeros(2^N);

    for k=1:N
        if k==N && pbc
            hE=hE+kron(kron(sigmaZ,eye(2^(N-2))),sigmaZ);%hE is the entangled state
        elseif k<N
            hE=hE+kron(kron(kron(eye(2^(k-1)),sigmaZ),sigmaZ),eye(2^(N-1-k)));%hE is the entangled state
        end
        hG=hG+kron(kron(eye(2^(k-1)),sigmaX),eye(2^(N-k)));% hG is the non entangled stae
    end
    h=g*hG+hE; %Combines the two parts of the hamiltonian

elseif(strcmp(func2str(hamil),'cluster')) %For the cluster state hamiltonian
    h=zeros(2^N);
    if pbc
        h=h+kron(kron(kron(sigmaZ,eye(2^(N-3))),sigmaZ),sigmaX); %Defines the periodic Boundary Condition
        h=h+kron(kron(kron(sigmaX,sigmaZ),eye(2^(N-3))),sigmaZ); %Defines the periodic Boundary Condition
    end
    for k=1:N-2
        h=h+kron(kron(kron(kron(eye(2^(k-1)),sigmaZ),sigmaX),sigmaZ),eye(2^(N-2-k)));
    end
    h=-h;
end

[~,w]=eig(h);
E=min(min(w));
end