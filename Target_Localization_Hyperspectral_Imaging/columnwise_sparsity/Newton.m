function [D] = Newton(A, Y,rho)
% Function to update D
% Implementation of Newton procedure to solve the :
%
% D_est = argmin_D rho/2 ||D*A-Y||_F^2 
%        ||D_i|| = 1
%
% Sirisha Rambhatla
% Oct, 2015

close all

del = 1e-9;
epsilon = 1e-6;
D = zeros(size(Y, 1), size(A, 1));
Dprev = ones(size(Y, 1), size(A, 1));
eps = [];

% Number of iterations
k = 200;
its = 1; 


while((norm(D - Dprev) / norm(Dprev) >= epsilon) || (its < k))
    Dprev = D;
    D = D - rho * (D * A - Y) * A' / (rho * A * A' + del * eye(size(A, 1)));
    D=nrmc(D);
    
    
%      eps = [eps norm(D - Dprev) / norm(Dprev) ];
%      semilogy(eps, '-*')
%      drawnow
   
    its = its + 1;
    
    if (its >=k)
        break
    end
  
end