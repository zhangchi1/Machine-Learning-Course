function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1= X'; % 400 x m
a1 = [ones(1, size(a1, 2)); a1]; % 401 x m
z2 = Theta1 * a1; %25 x 401 x 401 x m = 25 x m
a2 = sigmoid(z2); % 25 x m
a2 = [ones(1, size(a1, 2)); a2]; % 26 x m
z3 = Theta2 * a2; % 10 x 26 x 26 x m = 10 x m
a3 = sigmoid(z3); % 10 x m
[x, p] = max(a3, [], 1);

% =========================================================================


end
