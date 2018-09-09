% Instituto Federal de Brasília - Campus Taguatinga
% Curso: Bacharelado em Ciência da Computação
% Disciplina: Aprendizagem de Máquina - 2/2018
% Professor: Lucas Moreira

% Estudante: Carolina Ataíde de Assis
% Matrícula: 161057600051
% Período: 6° semestre
% Data: 09/09/2018

          % Trabalho 01 - Regressão Linear de Multiplas Variáveis 
            % utilizando o algoritmo do gradiente descendente.

% Limpeza
  clear ; close all; clc

% Load dos dados
  data = load('data.txt');
  X = data(:, 1:2);
  y = data(:, 3);
  m = length(y);

% Escalonamento de variáveis: normalização pela média 
  
  num_var = size(X, 2); % número total de variáveis a serem consideradas
  var = X; % matriz que será utilizada para normalizar X

  % 1 - média do conjunto de amostras para as variáveis:

    for j = 1:num_var
      media(1, j) = sum(var(:, j))./length(var); % vetor com a média das amostras de cada variável
      X(:, j) = X(:, j) - media(1, j); % atualização da matriz X: amostras subtraidas pela média do conjunto de amostras
    end

  % 2 - valor máximo subtraído do seu valor mínimo:

    for k = 1: num_var
      S(1, k) = max(var(:, k)) - min(var(:, k)); % vetor com a faixa de valores de cada variável
      X(:, k) = X(:, k) ./ S(1, k); % atualização da matriz X: amostras divididas pela suas faixa de valores
    end
    
% ------------------------------------------------------- %

  X = [ones(m, 1) X]; % Adição de uma coluna de 1's em X
  alpha = 0.9; % Taxa de aprendizagem
  num_iter = 610; % Quantidade de iterações
  theta_ini = zeros(3, 1); % Valores iniciais de theta

% Gradiente Descendente
  [theta, J] = gradienteDescendente(X, y, theta_ini, alpha, num_iter);

% Vetor de parâmetros theta com os valores ótimos
  fprintf('Theta: \n');
  fprintf(' %f \n', theta);
  fprintf('\n');

% Estimativa do preço de um imovel de 1650 ft² e 3 quartos 
  estimar = [1650, 3];
  
  for p = 1: num_var
    estimar_normalizado(p) = (estimar(p) - media(p)) ./ S(p); % Vetor com os valores normalizados
  end
  
  Valor_Estimado = [1, estimar_normalizado] * theta;

  fprintf(' Estimativa do preco de um imovel de 1650 pes quadrados e 3 quartos, a partir do gradiente descendente, utilizando: \n');
  fprintf(' Alpha = %.2f e %d iteracoes\n', alpha, num_iter);
  fprintf(' Valor estimado: %.2f \n\n', Valor_Estimado);
  
% Grafico do custo de acordo com as iteracoes
  plot(J, 'LineWidth', 2, 'r');
  xlabel('Iteracoes'); ylabel('Custo');