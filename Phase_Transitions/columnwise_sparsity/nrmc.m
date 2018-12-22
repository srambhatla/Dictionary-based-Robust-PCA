function [A_n] = nrmc(A)
% Sirisha Rambhatla, 2014
% Alternative to normc for Matlab 2014b and above. 

A_n = zeros(size(A));

    for i = 1:size(A, 2)
       if (sum(A(:,i)) ~= 0)
         A_n(:,i) = A(:,i)/norm(A(:,i));
       end
    end
end