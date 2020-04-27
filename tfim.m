function [energy] = tfim(v, a, b, w, g, old, pbc)
% Transfer Field Ising Model

[N,M] = size(w); %read in N and M
xterm = 0.0;
zterm = 0.0;

% Find local energy at each spin site and average
for j = 1:N
    if j == 1
        factor = v(1)*v(2);
    elseif j == N
        v(j-1) = -1*v(j-1);
        if pbc %check for periodic boundary conditions
            factor = v(N)*v(1);
        else
            factor = 0;
        end
    else
        v(j-1) = -1*v(j-1); %reverse flip done by previous iteration
        factor = v(j)*v(j+1); %calculate sigma z factor for the z term
    end
    v(j) = -1*v(j); %flip spin at current site
    [new,~] = waveExact(a,b,w,v); %find new wavefunction
    xterm = xterm +(conj(new/old)); %calculate sigma x and add to x term
    zterm = zterm + factor; %add factor to z term
end
energy = -(g*xterm)-zterm; %compute the energy
end
