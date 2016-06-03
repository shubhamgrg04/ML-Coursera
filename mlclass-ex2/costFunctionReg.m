function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

h = sigmoid(X*theta);
n = length(theta);
J = ((-y'*log(h) - (1-y)'*log(1-h))' + lambda*sum(theta.^2)/2)/m;
grad = zeros(n,1);
grad(1) = (h(1) - y(1))/m;
grad(2:n) = ((h-y)'*X(:,2:n))'  +   lambda*theta(2:n);
grad = grad/m;


end
