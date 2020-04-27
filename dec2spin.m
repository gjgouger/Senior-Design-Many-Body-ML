function [s]=dec2spin(p,N)
p=p-1;
s=de2bi(p,N);
s=flip(s);
s(s==1)=1;
s(s==0)=-1;
end