% Gen Data for phase transition experiments 
% Sirisha Rambhatla, Aug 2018

function [X, R, A] = gen_dat_col (n, m, d, lr, k)

% X - Low rank part
P = nrmc(randn(n, lr)); 
Q = nrmc(randn(m-k, lr)); 
X = [P*Q' zeros(n,k)]; 

% dictionary 
R = nrmc(randn(n , d));

% A - coefficients 
A = [zeros(d, m-k) sqrt(1/d)*randn(d, k)];
end 
