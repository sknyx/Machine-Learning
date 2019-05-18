function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) calcula a função-custo e suas derivadas parciais.
%   Os parâmetros (pesos) estão contidos no vetor nn_params, e devem
%   ser convertidos em usas matrizes originais (já implementado abaixo)
% 

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0; % variável para o valor da função-custo
Theta1_grad = zeros(size(Theta1)); % matriz com as derivadas para a primeira matriz de pesos
Theta2_grad = zeros(size(Theta2)); % matriz com as derivadas para a segunda matriz de pesos

% ====================== SEU CÓDIGO AQUI ======================
% Implemente seu código em duas partes. Primeiro, implemente o código
% da função-custo sem regularização. Depois implemente o cálculo do
% termo de regularização. Em seguida, implemente o código para o cálculo
% das derivadas parciais.
%
% Dica: O vetor y passado para a função é um vetor de rótulos contendo
%       os valores de 1..K. É necessário mapear esse vetor para os valores
%       binários 0 e 1 para serem usados no cálculo da função-custo.
%

X = [ones(m,1) X];

a2 = sigmoid(X*Theta1');
a2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(Theta2*a2');

temp_y = y == [1:num_labels];
temp_y = temp_y';
J = sum(sum(-temp_y.*log(a3)-(1-temp_y).*log(1-a3)))*(1/m);

termreg = (lambda/(2*m))*(sum(sum(Theta1(:,2:size(Theta1,2)).^2))+sum(sum(Theta2(:,2:size(Theta2,2)).^2)));
J = J+termreg;

dp1 = zeros(size(Theta1));
dp2 = zeros(size(Theta2));

z2 = X*Theta1'; 
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

errk = a3-temp_y';
err2 = (errk*Theta2)(:,2:end).*sigmoidGradient(z2);

dp2 = errk'*a2; 
dp1 = err2'*X; 

Theta1_grad=(1/m)*dp1;
Theta2_grad=(1/m)*dp2;

termreg1 = (lambda/m)*Theta1;
termreg1(:,1) = 0;
termreg2 = (lambda/m)*Theta2;
termreg2(:,1) = 0;

Theta1_grad = Theta1_grad+termreg1;
Theta2_grad = Theta2_grad+termreg2;

% -------------------------------------------------------------
% =========================================================================

% Transforma as matrizes de derivadas em um único vetor
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
