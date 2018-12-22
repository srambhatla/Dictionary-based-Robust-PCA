% Function to run the actual accelerated proximal gradient algorithm for the particular 
% choice of lam and nu; see the applications paper for details. 
% Warm start enabled via initializing X_init and A_init.
% 
% Y -- the data matrix, R -- the dictionary, A -- the Sparse coefficient matrix, X -- the low-rank component.
% Sirisha Rambhatla, 2016

function [X, A] = acc_proj_grad_hs(Y, R, lam, nu, X_init, A_init, w, i, clss, lab, out_folder)

% Set the nu and nu_bar parameters
v = 0.95; v_avg = 1e-4; 

% Evaluate the Lipschitz constant
L_f = lam_max([eye(size(R,1)) R]);

% Initialize parameters
tmin1 = 1; 
t = 1; 
tol = 1e-4;

X = X_init; 
Xmin1 = X ; 

A = A_init; 
Amin1 = A;

k = 1; 
obj = []; 
sp =[];
flag =1;

display('Running...')

while (flag)
    
    Tx = X + ( (tmin1 - 1)/t )* (X - Xmin1); 
    Ta = A + ( (tmin1 - 1)/t )* (A - Amin1); 
    
    Gx = Tx + (1/L_f) * (Y - Tx - R*Ta);
    Ga = Ta + (1/L_f) * R'*(Y - Tx - R*Ta);
   
    [U, S, V] = svd(Gx); 
    
    Xmin1 = X; 

    X = U*softThr(S, nu/L_f)*V';
    
    Amin1 = A; 
    A = softThrCol(Ga, lam*nu/L_f);

    
    tmin1 = t; 
    t = (1 + sqrt(4*t^2 + 1))/2; 
     
    nu = max(nu*v, v_avg); 
 
    k = k + 1; 

    % Decide to exit the loop
    if (k > 5000)
        break;
    end
    
    Z = [L_f*(Tx - X) + X + R*A - Tx - R*Ta ; L_f*(Ta - A) + R'*(X + R*A - Tx - R*Ta)];
    if(norm(Z, 'fro') <= tol*max(1, L_f*norm(Xmin1,'fro')))
        flag =0;
    end
display(strcat('Iter outer = ', num2str(i),', iter in = ', num2str(k), 'tol = ', num2str(tol*max(1, L_f*norm(Xmin1,'fro'))), ', ||Z||_F =', num2str(norm(Z,'fro'))))
end
    
display('Done!')

 name_fil = strcat('./',out_folder,'/res_X_A_cl_', num2str(clss),'_', num2str(i),'_',lab, '.mat');
 save(name_fil, 'X', 'A', 'w', 'i', 'lam','lab');

    function [L_f] = lam_max(Z)
         L_f = norm(Z'*Z);    
    end

end
