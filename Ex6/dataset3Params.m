function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

values = [0.01 0.03 0.1 0.3 1 3 10 30];
%initialize the the min_error to a positive infinite number
min_error = Inf;
%try all possible pairs of values for C and sigma
for c_i = values
    for sigma_i = values
        %get the train model
        train_model= svmTrain(X, y, c_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i));
        predictions = svmPredict(train_model, Xval);
        error = mean(double(predictions ~= yval));
        if(error < min_error)
            C = c_i;
            sigma = sigma_i;
            min_error = error;
        end
    end
end

      
        

    
    







% =========================================================================

end