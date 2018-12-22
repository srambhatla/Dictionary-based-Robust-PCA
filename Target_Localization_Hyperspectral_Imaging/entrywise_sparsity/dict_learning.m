%% Dictionary Learning to learn a dictionary for a class 
% D_cl the dict learned
% A_cl the coefficient matrix
% Y_cl data corresponding to the class 

% Solves the following optimization problem 
%   min.  || Y - DA ||_F^2 + ||A||_1  
%  D, A
%
% Sirisha Rambhatla, 2016 

function [R,A_cl] = dict_learning(Y_b, gt, d, lam, class)
% R stores the final dictionary D_cl

% Extract the data corresponding to the class of interest
[sam_x,sam_y] = find(gt==class); 
Y_cl = zeros(size(Y_b,3),numel(sam_x));
for i = 1:numel(sam_x)
    Y_cl(:,i) = squeeze(Y_b(sam_x(i),sam_y(i),:));
end
Y_cl = Y_cl./max(max(abs(Y_cl)));

% Initialize convergence metrics 

tol = 1e-3;
i = 1; 
flag = 1; 
err = [];

% Set-up parameters 
rho = 2; 
A_cl = zeros(d, size(Y_cl,2));
D_cl = randn(size(Y_cl,1), d);

display(strcat('Learning dictionary for class ',num2str(class), ' with ', num2str(d), ' elements.'))
% Begin alternating minimization

while (flag)
    
    % Update coefficients
    A_cl =  FISTA(D_cl,Y_cl,lam);
    
    % Update dictionary
    D_cl = Newton(A_cl, Y_cl,rho); 
    
    % Check the data fit
    Y_est = D_cl*A_cl;
    
    err = [err norm(Y_cl - Y_est,'fro')/norm(Y_cl)];
    
    % Check if we need to exit
    if (i >2)
        if ((abs(err(end)) <= tol) || (norm(err(end) - err(end-1)) <= 1e-4))
            flag = 0;
        end
    end
    display(strcat('Iter =',num2str(i), 'Error =', num2str(err(end))))

    % plot(err)
    % drawnow
    i = i + 1;
end
% Set the dictionary 
R = D_cl;
