clear
clc
start_time = tic;

% This is the primary script to run the RBM gradient descent algorithm.
% This is where inputs such as the number of spin bodies and the
% hamiltonian function are changed.

% Inputs:
exact=false;
N = 4; %number of spin bodies
alpha = 1; %ratio of hidden to visible nodes
L = 1000; %number of Metroplis Hastings loops per walker
burn = 100; %number of initial MH loops burned
walks = 1; %number of MH walkers
R = 600; %number of gradient descent iterations

hamiltonian = @cluster; %hamiltonian function (cluster, tfim, ...)
g = 2; %hamiltonian parameter
pbc = true; %periodic boundary condition boolean

learn = 0.05; %learning rate parameter
beta=0.5; %Momentum decay rate
adambeta1=0.5; %adam parameters
adambeta2=.5;
learn_func = @const; %learning rate function (const, square, decay, ...)
kick = 0; %kick strength

trials = 1; %number of gradient descent trials
% End of inputs

M = round(alpha*N); %calculates number of hidden nodes M
results = zeros(1,4);
fig = figure(1);
kick = kick*N/10;
w = make_w(N,M,exact);
a = make_a(N,exact);
b = make_b(M,exact);


fprintf('%21s %11s %6s %10s %21s\n', 'Experimental Minimum', 'Converged', 'Kicks', 'Time(s)','Theoretical Mininum')

% Begin looping through trials
for t = 1:trials
    %% 1
    %open trial specific file to write into
    filename = ['trial' num2str(1) '.txt'];
    fileID = fopen(filename,'w');
    
    %call gradient descent function with all inputs
    [results(1,1),results(1,2),results(1,3),results(1,4),results(1,5)] = grad_descent_direct_linear(N,M,L,burn,walks,R,g,hamiltonian,learn,learn_func,beta, adambeta1, adambeta2,kick,1,4,fileID,pbc,exact,w,a,b);
    
    %print results
    fprintf('%21.6f %11.6f %6d %10.2f %21.6f\n', results(1,1),results(1,2),results(1,3),results(1,4),results(1,5))
    fprintf(fileID,'End of trial');
    
    %% 2
  
        %open trial specific file to write into
    filename = ['trial' num2str(2) '.txt'];
    fileID = fopen(filename,'w');
    
    %call gradient descent function with all inputs
    [results(2,1),results(2,2),results(2,3),results(2,4),results(2,5)] = grad_descent_direct_SGD(N,M,L,burn,walks,R,g,hamiltonian,learn,learn_func,beta, adambeta1, adambeta2,kick,2,4,fileID,pbc,exact,w,a,b);
    
    %print results
    fprintf('%21.6f %11.6f %6d %10.2f %21.6f\n', results(2,1),results(2,2),results(2,3),results(2,4),results(2,5))
    fprintf(fileID,'End of trial');
   
    %% 3

           %open trial specific file to write into
    filename = ['trial' num2str(3) '.txt'];
    fileID = fopen(filename,'w');
    
    %call gradient descent function with all inputs
    [results(3,1),results(3,2),results(3,3),results(3,4),results(3,5)] = grad_descent_momentum_linear(N,M,L,burn,walks,R,g,hamiltonian,learn,learn_func,beta, adambeta1, adambeta2,kick,3,4,fileID,pbc,exact,w,a,b);
    
    %print results
    fprintf('%21.6f %11.6f %6d %10.2f %21.6f\n', results(3,1),results(3,2),results(3,3),results(3,4),results(3,5))
    fprintf(fileID,'End of trial');
    
    %% 4
    

           %open trial specific file to write into
    filename = ['trial' num2str(4) '.txt'];
    fileID = fopen(filename,'w');
    
    %call gradient descent function with all inputs
    [results(4,1),results(4,2),results(4,3),results(4,4),results(4,5)] = grad_descent_momentum_SGD(N,M,L,burn,walks,R,g,hamiltonian,learn,learn_func,beta, adambeta1, adambeta2,kick,4,4,fileID,pbc,exact,w,a,b);
    
    %print results
    fprintf('%21.6f %11.6f %6d %10.2f %21.6f\n', results(4,1),results(4,2),results(4,3),results(4,4),results(4,5))
    fprintf(fileID,'End of trial');
    
    
    
end

set(fig, 'Position', [0 0 2400 1500])

% Find minimum energies and total calculation time
fprintf('Lowest energy found at E = %10.6f\n', min(results(:,1)))
fprintf('Lowest converged energy found at E = %10.6f\n', min(results(:,2)))
tot_time = toc(start_time);
fprintf('Total time (seconds): %10.2f\n',tot_time)



