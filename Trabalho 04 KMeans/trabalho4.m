%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Instituto Federal de Brasília - Campus Taguatinga
% Curso: Bacharelado em Ciência da Computação
% Disciplina: Aprendizagem de Máquina - 2/2018
% Professor: Lucas Moreira

% Estudante: Carolina Ataíde de Assis
% Matrícula: 161057600051
% Período: 6° semestre
% Data: 25/11/2018

% Trabalho 04 - K-Means
 
  clear ; close all; clc

  load('data.mat');
  X = [millage, carbon];
  K = 5;
  max_iterations = 50;

  %inicializa centroides aleatoriamente
  centroids = zeros(K,size(X,2)); 
  randidx = randperm(size(X,1));
  centroids = X(randidx(1:K), :);

  printf('Calculando centroides...\n');   
  for i=1:max_iterations
    printf('Iteracao %d de %d...', i, max_iterations);
    indices = findClosestCentroids(X, centroids);
    centroids = computeCentroids(X, indices, centroids);
    clc;
  end

  printf('Valor medio de consumo e emissao de carbono (centroids) de cada classe:\n\n');
  printf('   Milhagem | Carbono\n');
  disp(centroids);