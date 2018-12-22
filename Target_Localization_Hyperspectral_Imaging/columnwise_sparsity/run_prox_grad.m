% Run DRPCA and Pseudo-inversed version for all values of the parameters 
% to find the best parameters 
%
% Sirisha Rambhatla, August 2016
function [flag] = run_prox_grad (Y, R, X_init, A_init, n, w, clss, lab, Y_p, R_p, X_init_p, A_init_p, lab_p, id_par, out_folder)

flag = 0;

% Set nu and lam
nu = norm(Y);
lam1= max(cellfun(@norm, num2cell(abs(R'*Y), 1))/nu);
lam = linspace(w*lam1,lam1, n);

% Set nu and lam
nu_p = norm(Y_p);
lam1_p= max(cellfun(@norm, num2cell(abs(R_p'*Y_p), 1))/nu_p);
lam_p = linspace(w*lam1_p,lam1_p, n);

% Check if all the regularization parameters need to be sweeped or just some 
% specific ones 

partial = isempty(id_par);

if (~partial)
  
  parfor i1 = 1:numel(id_par)
    [X_est, A_est] = acc_proj_grad_hs(Y, R,lam(id_par(i1)), nu, X_init, A_init, w, id_par(i1), clss,lab, out_folder);
    [X_est_p, A_est_p] = acc_proj_grad_hs(Y_p, R_p,lam_p(id_par(i1)), nu_p, X_init_p, A_init_p, w, id_par(i1), clss,lab_p, out_folder);
  end

else

  parfor i1 = 1:n
    [X_est, A_est] = acc_proj_grad_hs(Y, R,lam(i1), nu, X_init, A_init, w, i1, clss,lab, out_folder);
    [X_est_p, A_est_p] = acc_proj_grad_hs(Y_p, R_p,lam_p(i1), nu_p, X_init_p, A_init_p, w, i1, clss,lab_p, out_folder);
  end
end
    
flag = 1
end
