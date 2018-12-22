% Soft-Thresholding Operator
% Sirisha Rambhatla, 2018
function [X]=softThrCol(M,lambda)

% Convert to cells
C = num2cell(M, 1);

% Col norms 
col_norm = cellfun(@norm, C);
nnz_col = find(col_norm>0);

% Initialize X
X = zeros(size(M));

% soft threshold columns corresponding to the norm != 0
X(:,nnz_col) = max (M(:,nnz_col) - lambda*M(:,nnz_col)./kron(col_norm(nnz_col), ones(size(M,1),1)), 0);
