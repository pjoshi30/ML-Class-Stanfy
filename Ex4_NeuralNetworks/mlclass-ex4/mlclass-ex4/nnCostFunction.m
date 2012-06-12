function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

accDeltaGrad1 = zeros(size(Theta1));
accDeltaGrad2 = zeros(size(Theta2));

%Add the column of ones to X
X = [ones(X, 1) X];

for i =1:m,
    a1 = X(i,:);
    a1 = [1 a1];
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);
    a2 = [1 a2'];
    z3 = Theta2 * a2';
    a3 = sigmoid(z3);
    sig = a3';
    actualY = zeros(num_labels,1);

    %Set the appropriate index of actualY to 1
    if y(i) <= size(actualY,1),
        actualY(y(i)) = 1;     
    end;

    onesY = ones(size(actualY));
    onesSig = ones(size(sig));

    J = J + ((log (sig) * ...
             (-actualY)) - (log(onesSig - sig) * (onesY-actualY)));

    %BackPropogation algorithm starts here
    delta3 = a3 - actualY;
    delta2_temp = Theta2'*delta3;
    delta2 = delta2_temp(2:end).*sigmoidGradient(z2); 


    %Accumulate the gradient
    accDeltaGrad2 = accDeltaGrad2 +  (delta3*a2);
    accDeltaGrad1 = accDeltaGrad1 + (delta2*a1);

end;

%Add regularization
regTheta1Temp = (lambda/m)*(Theta1(:,2:end));
regTheta1Temp = [zeros(size(Theta1,1),1) regTheta1Temp];
regTheta2Temp = (lambda/m)*(Theta2(:,2:end));
regTheta2Temp = [zeros(size(Theta2,1),1) regTheta2Temp];
Theta1_grad = (accDeltaGrad1/m) + regTheta1Temp;
Theta2_grad = (accDeltaGrad2/m) + regTheta2Temp;

J = J/m;

%Add regularization to the above computed cost
Theta1ExcludingOne = Theta1(:,2:end);
Theta2ExcludingOne = Theta2(:,2:end);
sumTheta = sum(sum(Theta1ExcludingOne.^2)) + sum(sum(Theta2ExcludingOne.^2));
reg = (lambda/(2*m)) * sumTheta;

%Add the regularization to the cost
J += reg;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
