function [E, grad_a, grad_b, grad_w] = metrohast(a_vec, b_vec, w_mat, L, burn, hamiltonian, g, walks, pbc)
[N,M] = size(w_mat); %read in N and M
E = 0;

% Initialize all arrays for gradient calculation
o_w = zeros(N*M,1);
o_w_e = zeros(N*M,1);
o_a = zeros(N,1);
o_a_e = zeros(N,1);
o_b = zeros(M,1);
o_b_e = zeros(M,1);

S_a = zeros(N);
S_b = zeros(M);
S_w = zeros(N*M,1);

% Loop through desired number of walks
for k = 1:walks

v_vec = rand(N,1); %generate random v

% Convert v and -1 and 1
for j = 1:N
    if v_vec(j) < 0.5
        v_vec(j) = -1;
    else
        v_vec(j) = 1;
    end
end

[old_wave, otan] = waveExact(a_vec,b_vec,w_mat,v_vec); %calculate starting wavefunction

% Begin Metropolis Hastings
for i = 1:L
    
    new_v_vec = v_vec; %copy v %new_v_vec should be copied onto gpu
    
    r = randi([1 N]); %generate random index to flip
    new_v_vec(r) = -new_v_vec(r); %flip index
    [new_wave, new_otan] = waveExact(a_vec,b_vec,w_mat,new_v_vec); %find new wavefunction
    ratio = abs(((new_wave)/(old_wave))^2); %calculate ratio of probabilities
    
    % Compare probabilities
    if ratio >= 1
        % New v accepted
        %                                                                                    
        old_wave = new_wave;
        v_vec = new_v_vec;
        otan = new_otan;
    else
        ran = rand; %generate random double 
        if ratio > ran
            % New v accepted
            old_wave = new_wave;
            v_vec = new_v_vec;
            otan = new_otan;
        end
    end
    
    % Do not perform energy and gradient calculation if burning
    if i > burn
        
        % Calculate energy of current v using hamiltonian
        epsilon = feval(hamiltonian,v_vec,a_vec,b_vec,w_mat,g,old_wave,pbc);
        
        % Add current loops results to gradient calculation
        votan = v_vec*otan.';
        revotan=reshape(votan,[N*M,1]);
        o_w = o_w + revotan;
        o_w_e = o_w_e + conj(epsilon*(revotan));
        o_a = o_a + (v_vec);
        o_a_e = o_a_e + conj((epsilon)*(v_vec));
        o_b = o_b + otan;
        o_b_e = o_b_e + conj(epsilon*(otan));
        
        % Stochastic reconfiguration matrices
        S_a = S_a + conj(v_vec)*v_vec.';
        S_b = S_b + conj(otan)*otan.';
        S_w = S_w + conj(revotan)*revotan.';
        
        % Add current energy to total energy
        E = E+epsilon;
    end
    
end
end

loops = walks*(L-burn); %calculate total number of loops used

% Average everything by the number of loops
E = E/loops;
S_a = S_a - conj(o_a)*o_a.';
S_b = S_b - conj(o_b)*o_b.';
S_w = S_w - conj(o_w)*o_w.';
o_w = o_w/loops;
o_w_e = o_w_e/loops;
o_a = o_a/loops;
o_a_e = o_a_e/loops;
o_b = o_b/loops;
o_b_e = o_b_e/loops;


% Finish calculation stochastic matrices


S_a = S_a/loops;
S_b = S_b/loops;
S_w = S_w/loops;

S_a = S_a + 1E-5*eye(size(S_a));
S_b = S_b + 1E-5*eye(size(S_b));
S_w = S_w + 1E-5*eye(size(S_w));

% Calculate gradients
grad_a = inv(S_a)*((o_a_e) - conj(o_a)*(E)); %make on gpu
grad_b = inv(S_b)*((o_b_e) - conj(o_b)*(E));
grad_w = inv(S_w)*((o_w_e) - conj(o_w)*(E));
grad_w=reshape(grad_w,[N,M]).';
E = real(E); %take only the real part of the energy to pass out

end