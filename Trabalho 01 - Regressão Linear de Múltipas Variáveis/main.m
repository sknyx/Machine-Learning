% Instituto Federal de Bras�lia - Campus Taguatinga
% Curso: Bacharelado em Ci�ncia da Computa��o
% Disciplina: Aprendizagem de M�quina - 2/2018
% Professor: Lucas Moreira

% Estudante: Carolina Ata�de de Assis
% Matr�cula: 161057600051
% Per�odo: 6� semestre
% Data: 09/09/2018

          % Trabalho 01 - Regress�o Linear de Multiplas Vari�veis 
            % utilizando o algoritmo do gradiente descendente.

% Limpeza
  clear ; close all; clc

% Load dos dados
  data = load('data.txt');
  X = data(:, 1:2);
  y = data(:, 3);
  m = length(y);

% Escalonamento de vari�veis: normaliza��o pela m�dia 
  
  num_var = size(X, 2); % n�mero total de vari�veis a serem consideradas
  var = X; % matriz que ser� utilizada para normalizar X

  % 1 - m�dia do conjunto de amostras para as vari�veis:

    for j = 1:num_var
      media(1, j) = sum(var(:, j))./length(var); % vetor com a m�dia das amostras de cada vari�vel
      X(:, j) = X(:, j) - media(1, j); % atualiza��o da matriz X: amostras subtraidas pela m�dia do conjunto de amostras
    end

  % 2 - valor m�ximo subtra�do do seu valor m�nimo:

    for k = 1: num_var
      S(1, k) = max(var(:, k)) - min(var(:, k)); % vetor com a faixa de valores de cada vari�vel
      X(:, k) = X(:, k) ./ S(1, k); % atualiza��o da matriz X: amostras divididas pela suas faixa de valores
    end
    
% ------------------------------------------------------- %

  X = [ones(m, 1) X]; % Adi��o de uma coluna de 1's em X
  alpha = 0.9; % Taxa de aprendizagem
  num_iter = 610; % Quantidade de itera��es
  theta_ini = zeros(3, 1); % Valores iniciais de theta

% Gradiente Descendente
  [theta, J] = gradienteDescendente(X, y, theta_ini, alpha, num_iter);

% Vetor de par�metros theta com os valores �timos
  fprintf('Theta: \n');
  fprintf(' %f \n', theta);
  fprintf('\n');

% Estimativa do pre�o de um imovel de 1650 ft� e 3 quartos 
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