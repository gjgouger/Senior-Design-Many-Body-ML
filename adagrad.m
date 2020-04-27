function [update,new_g]=adagrad(previous_g, gradient_cost, learn)
new_g=previous_g+gradient_cost.^2;
update=learn./((new_g+1E-7).^.5).*gradient_cost;