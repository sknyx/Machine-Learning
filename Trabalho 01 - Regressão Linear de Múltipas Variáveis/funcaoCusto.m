function J = funcaoCusto(X, y, theta)
J = 0;
J = sum((X * theta - y).^2)/(2*length(X));
end