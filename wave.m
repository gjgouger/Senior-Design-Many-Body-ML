function [wave, otan] = wave(v, a, b, w)
wave = -(a.'*v); %calculate a.v
temp = w.'*v+b; %calculate w'*v+b
otan = tanh(temp); %compute tanh(theta) for gradient calculation
temp = log(cosh(temp)); %take the cosh and log of theta for wavefunction calculation
wave = wave+sum(temp); %add everything to wave to get the log of the wavefunction
end
