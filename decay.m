function [l] = decay(p,k,R)
decayrate=0.00025;
modification=1/(1+decayrate*(k*R));
l = p*modification;
end