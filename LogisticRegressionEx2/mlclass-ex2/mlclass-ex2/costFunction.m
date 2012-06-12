function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%This snippet calculates cost function
z = X * theta;
sig = sigmoid(z);
onesY = ones(size(y));
onesSig = ones(size(sig));
J = (1/m) * ((log (sig)' * (-y)) - (log(onesSig - sig)' * (onesY-y)));

%This snippet calculates the gradient
grad = (1/m) * (X' * (sig - y));
%numberOfFeatures = size(X,2);
%for i = 1:numberOfFeatures,
%        grad(i,:) = (1/m) * ((sig - y)' * X(:,i));
%end;



% =============================================================

end
