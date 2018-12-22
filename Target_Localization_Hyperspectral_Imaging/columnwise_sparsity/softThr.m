% Soft-Thresholding Operator
% Sirisha Rambhatla, 2014
function [x]=softThr(y,lambda)
	x=sign(y).*max(abs(y)-lambda, 0);
end