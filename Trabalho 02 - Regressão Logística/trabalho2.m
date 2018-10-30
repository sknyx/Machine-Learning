% Instituto Federal de Bras�lia - Campus Taguatinga
% Curso: Bacharelado em Ci�ncia da Computa��o
% Disciplina: Aprendizagem de M�quina - 2/2018
% Professor: Lucas Moreira

% Estudante: Carolina Ata�de de Assis
% Matr�cula: 161057600051
% Per�odo: 6� semestre
% Data: 21/10/2018

% Trabalho 02 - Regress�o Log�stica

clear ; close all; clc

function p = predict(theta, X)
  m = size(X, 1);
  p = zeros(size(X, 1), 1);
  X = [ones(m, 1) X];
  l = X*theta';
  [Y,p] = max(l,[],2);
end

function [theta] = oneVsAll(X, y, num_classes, lambda)
  m = size(X, 1);        
  n = size(X, 2);        
  theta = zeros(num_classes, n+1);  
  X = [ones(m, 1) X];  
  ini_theta = zeros(n+1, 1);
  op = optimset('GradObj', 'on', 'MaxIter', 50);
  
  for c=1:num_classes
    theta(c,:) = fminunc (@(t)(costFunction(t, X, (y == c), lambda)),ini_theta,op);
  end
  
end

num_classes = 10; 
%% === Load e visualiza��o dos dados
load('data.mat');
m = size(X, 1); 
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

%% == Classifica��o de m�ltiplas classes, usando a estrat�gia oneVsAll
fprintf('\nTreinamento da regressao logistica one-vs-all...\n')

lambda = 0.1;
[theta] = oneVsAll(X, y, num_classes, lambda);

fprintf('Pausa. Enter para continuar.\n');
pause;

%% == Predi��o

predicao = predict(theta, X);
fprintf('\nPrecisao: %f\n', mean(double(predicao == y)) * 100);

%% == Visualiza��o

rp = randperm(m);
number = 'Insira o numero: ';   
c = input(number);

for i = 1:m
  p = predict(theta, X(rp(i),:));
  if p == c
    displayData(X(rp(i), :));
    prompt = 'Qualquer tecla para continuar, [s] para sair:';
    stop = input(prompt, 's');
    if stop == 's'
      break
      else
        clc;
        print('%s', number);  
        c = input(number);   
      end
    end 
    end