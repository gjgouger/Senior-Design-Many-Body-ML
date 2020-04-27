function [update,new_m]=momentum(previous_m, gradient_cost, beta, learn)
new_m=beta*previous_m+(1-beta)*gradient_cost;
update=learn*new_m;
