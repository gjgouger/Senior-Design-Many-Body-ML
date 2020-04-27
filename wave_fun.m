function wave = wave_fun(v, a, b, w)
wave = -(a.'*v)+sum(log(cosh(w.'*v+b)));
end
