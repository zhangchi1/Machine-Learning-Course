function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples = 12

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%calculate the cost function J
% x = 12 x2 t = 2 x 1 y= 12 x 1
J = sum((X * theta - y) .^ 2) / (2 * m) + lambda * sum(theta(2:end,:) .^ 2) / (2 * m);

%calculate the gradient
grad(1,:) = sum((X * theta - y) .* X(:,1)) / m;
for i = 2:size(theta)
    grad(i,:) = (sum((X * theta - y) .* X(:,i)) / m) + (lambda * theta(i) / m);
end









% =========================================================================

grad = grad(:);

end
