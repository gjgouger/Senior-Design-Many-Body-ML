function [E, Ga, Gb, Gw] = calcDef(a_vec, b_vec, w_mat, hamiltonian, g, pbc,k)
[N,M] = size(w_mat);
Ga=zeros(N,1);
Gb=zeros(M,1);
Gw=zeros(N,M);
psi=zeros(2^N,1);
psia=zeros(2^N,N);
psib=zeros(2^N,M);
psiw=zeros(2^N,N,M);
[~,H]=theoretical_energy_calc(N,g,pbc,hamiltonian);
deltas=[.00001,.000001*j,-.000001*j,-.000001];
%% derivtave of a

for delta = deltas
    
    for l =1:N
        for p=1:2^N
            s=dec2spin(p,N)';
            psi(p)=waveExact(a_vec,b_vec,w_mat,s);
            delt=zeros(N,1);
            delt(l)=delta;
            psia(p,l)=waveExact(a_vec+delt,b_vec,w_mat,s);
        end
        psi=psi/norm(psi);
        psia(:,l)=psia(:,l)/norm(psia(:,l));
        E=real(psi'*H*psi);
        Ea=real(psia(:,l)'*H*psia(:,l));
        Ga(l)=Ga(l)+(Ea-E)/delta;
    end
    
    for l =1:M
        for p=1:2^N
            s=dec2spin(p,N)';
            psi(p)=waveExact(a_vec,b_vec,w_mat,s);
            delt=zeros(M,1);
            delt(l)=delta;
            psib(p,l)=waveExact(a_vec,b_vec+delt,w_mat,s);
        end
        psi=psi/norm(psi);
        psib(:,l)=psib(:,l)/norm(psib(:,l));
        E=real(psi'*H*psi);
        Eb=real(psib(:,l)'*H*psib(:,l));
        Gb(l)=Gb(l)+ (Eb-E)/delta;
    end
    
    for l =1:N
        for k=1:M
            for p=1:2^N
                s=dec2spin(p,N)';
                psi(p)=waveExact(a_vec,b_vec,w_mat,s);
                delt=zeros(N,M);
                delt(l,k)=delta;
                psiw(p,l,k)=waveExact(a_vec,b_vec,w_mat+delt,s);
            end
            psi=psi/norm(psi);
            psiw(:,l,k)=psiw(:,l,k)/norm(psiw(:,l,k));
            E=real(psi'*H*psi);
            Ew=real(psiw(:,l,k)'*H*psiw(:,l,k));
            Gw(l,k)=Gw(l,k)+(Ew-E)/delta;
        end
    end
    
    
end
s=size(deltas,2);
Ga=conj(Ga)/s;
Gb=conj(Gb)/s;
Gw=conj(Gw)/s;


end
