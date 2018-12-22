% Low rank plus sparse in a dictionary -- The main function
% Sirisha Rambhatla, April 2016

function[done] = run_lr_dict_ent_sp(lr, k, d, out_folder)

% Set the size of the data matrix
n = 100; 
m = 100; 

% Number of Monte-Carlo simulations
monte = 10; 

% Initialize parallel workers
parpool('local',10);

count = 1; 
for i4 = 1:numel(d)
    for i3 = 1:numel(k)
        for i2 = 1: numel(lr)
        
        % Generate n x m x Monte-Carlo tensors for placeholder
        X_est_m = zeros(n, m, monte);
        A_est_m = zeros(d(i4), m, monte);
        
        X = zeros(n, m, monte);
        A = zeros(d(i4), m, monte);
        
        R = zeros(n, d(i4), monte);
        
        % Set the parameters for current run
        r_p = lr(i2);
        d_p = d(i4);
        k_p = k(i3);
        
        % Generate a random seed for this run for reproducibility
        rng(str2num(strcat(num2str(r_p),num2str(d_p),num2str(k_p))));
        fprintf('Iter = %2d; Running r = %2d, k = %2d, d = %2d  \n',count, r_p, k_p, d_p)
        
        % Run the Monte-Carlo simulations 
        parfor i = 1:monte
            
            % Generate the data 
            [X(:,:,i), R(:,:,i), A(:,:,i)] = gen_dat (n, m, d_p, r_p, k_p);
            
            % Form the data matrix Y  
            Y = X(:,:, i) + R(:,:,i)*A(:,:,i);
            
            % Initialize components
            X_init = zeros(size(X(:,:,i))); 
            A_init = zeros(size(A(:,:,i))); 
            
            % Run the accelerated proximal gradient algorithm across the space of regulalization parameter
            [X_est_m(:,:,i), A_est_m(:,:,i), ~, ~] = run_prox_grad (Y, R(:,:,i), X_init, A_init, X(:,:,i), A(:,:,i)); 
        end

        % Create the name (for saving the results)
        nam = strcat(out_folder,'dat_r_',num2str(r_p),'_k_', num2str(k_p),'_d_', num2str(d_p));
    
        % Save the matrices
        save(nam, 'X', 'R', 'A', 'd_p', 'r_p', 'k_p', 'X_est_m', 'A_est_m')
        
        count = count + 1;
       
        end

    end
end
done = 1;
end
