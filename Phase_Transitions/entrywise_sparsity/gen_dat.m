% Gen Data for phase transition diagrams
% Sirisha Rambhatla, April 2016
function [X, R, A] = gen_dat (n, m, d, lr, k)

% X - Low rank part
P = nrmc(randn(n, lr)); 
Q = nrmc(randn(m, lr)); 

X = P*Q'; 

% dictionary 
R = nrmc(randn(n , d));

% A - coefficients 
p = d*m;

A = (reshape(gen_coeff_mat_X1(1, k, p, 1, 1), d, m)); 
end
