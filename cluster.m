function [energy] = cluster(v, a, b, w, g, old, pbc)
% Cluster State Hamiltonian 

[N,M] = size(w); %read in N and M
energy = 0.0;

% Find local energy at each spin site and average
for j = 1:N
    if j == 1
        if pbc == true %check for periodic boundary conditions
            factor = v(N)*v(j+1);
        else
            factor = 0;
        end
    elseif j == N
        v(j-1) = -1*v(j-1);
        if pbc == true
            factor = v(j-1)*v(1);
        else
            factor = 0;
        end
    else
        v(j-1) = -1*v(j-1); %reverse flip done by previous iteration
        factor = v(j-1)*v(j+1); %calculate sigma z factor
    end
    v(j) = -1*v(j); %flip spin at current site
    [new,~] = waveExact(a,b,w,v); %find new wavefunction
    energy = energy - factor*(conj(new/old)); %calculate sigma x, multiply by sigma z factor and add to energy
end
end
