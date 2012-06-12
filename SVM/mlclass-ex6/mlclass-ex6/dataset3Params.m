function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% =========================================================================
C = 0.98;
sigma = 0.16;
C = 1;
sigma = 0.2;
C = 0.01
sigma = 0.01;
prevError = 999;
error = 999;
val =[999 C sigma]
%while (true)
while (C < 30)
	sigma = 0.01;
	while (sigma < 30)
%for sigma = [0.01 0.03 0.1 0.3 1 3 10 30]
	x1 = [1 2 1]; x2 = [0 4 -1];

	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

	predictions = svmPredict(model, Xval);

	error = mean(double(predictions ~= yval));
	if prevError < error,
		val = [error C sigma; val];
	else
		prevError = error;
	endif

	sigma = sigma * 1.1;
	endwhile
	C = C * 1.1;
endwhile
best = val(val == min(val(:,1)), :)
C = best(1,2);
sigma = best(1, 3);
%find the minimum error among t
%C
%sigma
%pause;
%C = C * 1.5;
%sigma = sigma - 0.01;
%endwhile

end
