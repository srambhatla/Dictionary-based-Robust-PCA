% Function to run the actual accelerated proximal gradient algorithm for the particular 
% choice of lam and nu; see the applications paper for details. 
% Warm start enabled via initializing X_init and A_init.
% 
% Y -- the data matrix, R -- the dictionary, A -- the Sparse coefficient matrix, X -- the low-rank component.
% Sirisha Rambhatla, 2016

function [X, A] = acc_proj_grad(Y_s, R, lam,nu, X_init, A_init)

% Set the nu and nu_bar parameters
v = 0.9; v_avg = 0.01; 

% Evaluate the Lipschitz constant
L_f = lam_max([eye(size(R,1)) R]);

% Initialize parameters
tmin1 = 1; 
t = 1; 

X = X_init; 
Xmin1 = X ; 

A = A_init; 
Amin1 = A;

tol = 1e-4;

k = 0; 
flag = 1;

 while(flag) 

    Tx = X + ( (tmin1 - 1)/t )* (X - Xmin1); 
    Ta = A + ( (tmin1 - 1)/t )* (A - Amin1); 
    
    Gx = Tx + (1/L_f) * (Y_s - Tx - R*Ta);
    Ga = Ta + (1/L_f) * R'*(Y_s - Tx - R*Ta);
    
    [U, S, V] = svd(Gx); 
    
    Xmin1 = X; 
    X = U*softThr(S, nu/L_f)*V';
    
    Amin1 = A; 
    A = softThr(Ga, lam*nu/L_f); 
    
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
        flag = 0;
    end
    
end
    

% Calculate the Lipschitz Constant
    function [L_f] = lam_max(Z)
        L_f = norm(Z'*Z);       
    end

end
