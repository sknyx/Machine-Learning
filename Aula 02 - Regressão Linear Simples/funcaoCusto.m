function J = funcaoCusto (X, y, theta)
  J = 0;
  %calculo da hipótese para todos os pontos de X
  m = length(y);
  J = sum(((X*theta)-y).^2)./(2*m);
endfunction