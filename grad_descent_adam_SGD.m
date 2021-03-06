function [E_min, E_conv, num_kick, time,theo] = grad_descent_adam_SGD(N, M, L, burn, walks, R, g, hamiltonian, learn_par, learn_func,beta,adambeta1, adambeta2, kick, t, trials, fileID, pbc,exact,w,a,b)

stime = tic; %start timer for trial
gd_last_num = 20; %number of iterations used to determine convergence

E_conv = 0; %lowest converged energy
conv = 0; %iteration number where previous convergence occurred
num_kick = 0; %number of kicks

% Generate random complex a, b, and w as starting point
w_mat = w;
a_vec = a;
b_vec = b;
momentum_a=0;  %momentum parameters
momentum_b=0;
momentum_w=0;

adam_m_a=0; %adam parameters
adam_m_b=0;
adam_m_w=0;
adam_v_a=0;
adam_v_b=0;
adam_v_w=0;

% Output formatting for printing w matrix to file
output = '%f%+fi  ';
output_w = repmat(output,1,M);
output_w = [output_w '\n'];
wprint = zeros(N,2*M); %make in gpu

E_array_k = zeros(R,1); %stores energy value at each iteration %make in gpu

% Begin gradient descent
for k = 1:R
    
learn = feval(learn_func,learn_par,k,R); %calculate learning rate
 
% Find energy and gradients from a, b, and w using Metropolis Hastings
[E,grad_a,grad_b,grad_w] = metrohast_SGD(a_vec, b_vec, w_mat, L, burn, hamiltonian, g, walks, pbc);

% Print results to file
% fprintf(fileID,'k = %i\n',k);
% fprintf(fileID,'a_vec =\n');
% fprintf(fileID,'%f%+fi\n',[real(a_vec) imag(a_vec)].');
% fprintf(fileID,'b_vec =\n');
% fprintf(fileID,'%f%+fi\n',[real(b_vec) imag(b_vec)].');
% fprintf(fileID,'w_mat =\n');
for j = 1:M
    wprint(:,2*j-1:2*j) = [real(w_mat(:,j)) imag(w_mat(:,j))];
end
% fprintf(fileID,output_w,wprint.');
% fprintf(fileID,'E = %f\n',E);
% fprintf(fileID,'grad_a =\n');
% fprintf(fileID,'%f%+fi\n',[real(grad_a) imag(grad_a)].');
% fprintf(fileID,'grad_b =\n');
% fprintf(fileID,'%f%+fi\n',[real(grad_b) imag(grad_b)].');
% fprintf(fileID,'grad_w =\n');
% for j = 1:M
%     wprint(:,2*j-1:2*j) = [real(grad_w(:,j)) imag(grad_w(:,j))];
% end
% fprintf(fileID,output_w,wprint.');
% fprintf(fileID,'----------------\n');

% Calculate 2nd p-norm of gradients for normalization
norm_a = norm((grad_a),2);
norm_b = norm((grad_b),2);
norm_w = norm((grad_w),2);

% Move a, b, and w in direction of gradient
% If the norm is greater than the learning rate, normalize with norm

% a_vec = ((a_vec) - (grad_a)*min([1 learn/norm_a]));
% b_vec = ((b_vec) - (grad_b)*min([1 learn/norm_b]));
% w_mat = ((w_mat) - (grad_w)*min([1 learn/norm_w]));
[update_a,adam_m_a, adam_v_a]=adam(adam_m_a, adam_v_a, grad_a, adambeta1, adambeta2, learn);
[update_b,adam_m_b, adam_v_b]=adam(adam_m_b, adam_v_b, grad_b, adambeta1, adambeta2, learn);
[update_w,adam_m_w, adam_v_w]=adam(adam_m_w, adam_v_w, grad_w, adambeta1, adambeta2, learn);

a_vec = (a_vec) - update_a;
b_vec = (b_vec) - update_b;
w_mat = (w_mat) - update_w;

E_array_k(k) = E; %add E to array

% Check for convergence
% If last few iterations have std less than the square root of the learning rate
if k > gd_last_num && k >= conv+gd_last_num && std(E_array_k(k+1-gd_last_num:k)) < sqrt(learn)
    
    E_conv = min([min(E_array_k(k-9:k)) E_conv]); %determine if converged energy is a minimum
    conv = k; %update last convergence iteration
    
    % Kick a, b, and w in a random direction
    a_vec = a_vec + kick*make_a(N,exact);
    b_vec = b_vec + kick*make_b(M,exact);
    w_mat = w_mat + kick*make_w(N,M,exact);
    num_kick = num_kick + 1;
end

end

% Print final network to file
% fprintf(fileID,'Final network:\n');
% fprintf(fileID,'a_vec =\n');
% fprintf(fileID,'%f%+fi\n',[real(a_vec) imag(a_vec)].');
% fprintf(fileID,'b_vec =\n');
% fprintf(fileID,'%f%+fi\n',[real(b_vec) imag(b_vec)].');
% fprintf(fileID,'w_mat =\n');
% for j = 1:M
%     wprint(:,2*j-1:2*j) = [real(w_mat(:,j)) imag(w_mat(:,j))];
% end
% fprintf(fileID,output_w,wprint.');

%read E_array_k out of gpu
E_min = min(E_array_k); %find overall minimum energy

% Plot the results
[theo,~]=theoretical_energy_calc(N,g,pbc,hamiltonian); %grabs the theoretical energy and plots it with the experimental
rows = min([4 trials]);
cols = floor(trials/4)+1;
subplot(rows,cols,t)
ax = 1:R;
plot(ax,E_array_k,'b')
hold on
plot(ax,theo*ones(R),'r--')
hold off
xlabel('Gradient Descent Loop')
ylabel('Energy')
legend("Experimental Energy","Theoretical Energy")
axis([1 R -2*N N])
title('Adam descent with stochastic gradient descent')

time = toc(stime); %find the total time of the trial
