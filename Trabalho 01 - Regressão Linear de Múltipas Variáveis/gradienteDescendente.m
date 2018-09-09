function [theta, J] = gradienteDescendente(X, y, theta, alpha, num_iter)

  m = length(y);
  J = zeros(num_iter, 1); 

  for iter = 1:num_iter
    
      theta = theta - (alpha * X' *(X * theta - y) ./m); % atualiza��o dos valores theta a cada itera��o
      J(iter) = funcaoCusto(X, y, theta); % vetor de valores da fun��o custo para cada itera��o

  end
end