function [theta, J_iteracao] = gradienteDescendente(X, y, thetaInicial, alpha, iteracoes)
  J_iteracoes = zeros(iteracoes, 1);
  theta = thetaInicial;
  m = length(y);
  for i=1 : iteracoes;
    temp0 = theta(1) - (alpha./m) .* sum(X*theta-y);
    temp1 = theta(2) - (alpha./m) .* sum((X*theta-y).* X(:,2));
    theta(1) = temp0;
    theta(2) = temp1;
    J_iteracao = funcaoCusto(X, y, theta);
  endfor
endfunction
