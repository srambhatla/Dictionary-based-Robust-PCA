function [flag] = hyperSpec_func(clss, dict_lam, min_dict_size, out_folder, gt, S,id_par)
flag=0;

parpool('local',10)

%% Set parameters
x = size(S, 1);
y = size(S, 2);
z = size(S, 3);

Y = reshape(S, x*y, z);
Y = Y';
Y_b = reshape(Y', x, y, z);

%% Form Mask the Ground Truth Class

for i = 1:z
    M(:,:,i) = Y_b(:,:,i).*(gt==clss);
end

Y_w = reshape(M, x*y, z);
Y_w = Y_w';

%% Form Dictionary Elements

dict_siz = min_dict_size;
if (dict_siz > 0)
	[R,A_cl] = dict_learning(Y_b, gt,dict_siz , dict_lam, clss);
	R = nrmc(R);
else
	load('R.mat')

% From static pixels in case of Indian Pines dataset
% Form Dictionary Elements
%R = [squeeze(Y_b(16, 49, :))...
%    squeeze(Y_b(17, 48, :))...
%    squeeze(Y_b(17, 47, :))...
%    squeeze(Y_b(21, 47, :))...
%    squeeze(Y_b(20, 50, :))...
%    squeeze(Y_b(24, 47, :))...
%    squeeze(Y_b(27, 50, :))...
%    squeeze(Y_b(20, 48, :))...
%    squeeze(Y_b(21, 51, :))...
%    squeeze(Y_b(20, 48, :))...
%    squeeze(Y_b(20, 46, :))...
%    squeeze(Y_b(20, 49, :))...
%    squeeze(Y_b(19, 46, :))...
%    squeeze(Y_b(19, 47, :))...
%    squeeze(Y_b(23, 45, :))];
%
	R = (R)./max(max(Y));
	R = nrmc(R);

end
display('Size of dictionary')
size(R)

%% Run simulations on the data for different values. 
n = 100; % number of points
w = 0.01; % interval
Y1 = Y./max(max(abs(Y)));

% Initialize matrices
X_init = zeros(size(Y1)); 
A_init = zeros(size(R,2), size(Y1,2));

% Pseudo-inverse of R 
R_pseud = pinv(R);
Y1_p = R_pseud*Y1;

display('Singular Values of R_pinv')
svd(R_pseud)

% Initialize Matrices
X_init_p = zeros(size(Y1_p));
A_init_p = zeros(size(Y1_p));

%% Run Stuff
name = strcat('./',out_folder,'/param_roc_cl_',num2str(clss),'.mat');
if ~isempty(id_par)
   load(name)
else
   save(name, 'Y1', 'Y1_p','R', 'M', 'Y_w', 'n', 'w','clss','dict_lam','min_dict_size')
end

% Run L + RA and also the Pseudo-inversed version
flag = run_prox_grad (Y1, R, X_init, A_init, n, w, clss,'LpRA',Y1_p, eye(size(Y1_p,1)), X_init_p, A_init_p,'RPCA',id_par, out_folder);

