% Sirisha Rambhatla, April 2016
function [X_est, A_est, errX_e, errA_e] = run_prox_grad_col(Y, R, X_init, A_init, X, A)
% Function to search over all allowable values of regularization parameter and find the best performing lam
% using the accelerated proximal gradient algorithm
%
% Y -- the data matrix, R -- the dictionary, A -- the Sparse coefficient matrix, X -- the low-rank component



% Initialize parameters
errX = [];
errA = [];

% Find the range of regularization parameters we need to sweep
nu = norm(Y);
lam = linspace(max(cellfun(@norm, num2cell(abs(R'*Y), 1))/nu), 0, 100);
C = num2cell(A,1);
col_energy = cellfun(@norm,C);

% Set parameter
tol = 2*1e-2; 
bestErrA = 1000000;

close all
% For each lam in the range of allowable lam run the accelerated proximal gradient algorithm
% to find the best
    for i1 = 1:100
   
    [X_est, A_est] = acc_proj_grad_col(Y, R, lam(i1), nu, X_init, A_init); 
    
    C_est = num2cell(A_est,1); col_energy_est = cellfun(@norm,C_est);
    errX = [errX norm(X - X_est, 'fro')/norm(X, 'fro')];
    errA = [errA norm(col_energy - col_energy_est)/norm(col_energy)];
    
    
    if(errA(end) < bestErrA )
        bestErrA = errA(end);
        bestErrX = errX(end);
        bestX_est = X_est; 
        bestA_est = A_est;
        bestId = i1; 
    end
    
    if((errA(end) <= tol) && (errX(end) <= tol) )
         break; 
    end
    
    
    X_init = X_est;
    A_init = A_est;
     
     % plot(errA)
     % hold on
     % plot(errX, 'r')
     % drawnow
    end

    if (i1 ==100)
         errX_e = bestErrX;
         errA_e = bestErrA;
         X_est = bestX_est; 
         A_est = bestA_est;
    else
        errX_e = errX(end);
        errA_e = errA(end);
    end
        
    
end
