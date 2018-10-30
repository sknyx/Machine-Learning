function [J, grad] = costFunction(theta, X, y, lambda)

  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  
  z = X*theta; 
  h = sigmoid(z);
  
  termo_reg = (lambda/(2*m))*sum(theta(2:end).^2);
  
  J = (1/m)*sum((-y.*log(h))-((1-y).*log(1-h)))+termo_reg;
  
  grad(1) = (1/m)*(X(:,1)'*(h-y));                                   
  grad(2:end) = (1/m)*(X(:,2:end)'*(h-y))+(lambda/m)*theta(2:end); 
    
  grad = grad(:);
end