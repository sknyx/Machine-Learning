function [theta, J] = gradienteDescendente(X, y, theta, alpha, num_iter)

  m = length(y);
  J = zeros(num_iter, 1); 

  for iter = 1:num_iter
    
      theta = theta - (alpha * X' *(X * theta - y) ./m); % atualização dos valores theta a cada iteração
      J(iter) = funcaoCusto(X, y, theta); % vetor de valores da função custo para cada iteração

  end
end