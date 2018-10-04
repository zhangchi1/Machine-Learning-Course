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


% part1
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
a1= X'; % 400 x m
a1 = [ones(1, size(a1, 2)); a1]; % 401 x m
z2 = Theta1 * a1; %25 x 401 x 401 x m = 25 x m
a2 = sigmoid(z2); % 25 x m
a2 = [ones(1, size(a1, 2)); a2]; % 26 x m
z3 = Theta2 * a2; % 10 x 26 x 26 x m = 10 x m
a3 = sigmoid(z3); % 10 x m

%Create a matrixY, which convert a number format output to a matrix format output
% Example: 1 => [1, 0, ...], 2 => [0, 1, 0, ...]
unitMatrix = eye(num_labels);
matrixY = unitMatrix(y,:); % m x k(num_labels = 10)

% iterate through each training example
tempSum = 0;
for i = 1:m
    for k = 1: num_labels
    hTheta = a3(k,i); % 1 x 1
    tempSum = tempSum + (- matrixY(i,k)) * log(hTheta) - (1 - matrixY(i,k)) * log(1 - hTheta);
    end
end

% set the unregularized cost J
J = tempSum / m;

% Regularized part of the cost function
t1 = Theta1(:,2:end); % select only unbias neurons 25 x 400
t2 = Theta2(:,2:end); % select only unbias neurons 10 x 25
% sum up the regularized part
tempSumTheta = (sum(sum(t1 .* t1)) + sum(sum(t2 .* t2))) * lambda / (2 * m)

% set the regularized cost J
J = J + tempSumTheta;

% part 2 backpropagation
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
% iterate through each training example 
for i = 1:m
    a1_i = X(i,:); % 1 x 400
    a1_i = [1, a1_i]; % 1 x 401
    z2_i = Theta1 * a1_i'; % 25 x 401 x 401 x 1 = 25 x 1
    a2_i = sigmoid(z2_i); % 25 x 1
    a2_i = [1; a2_i]; % 26 x 1
    z3_i = Theta2 * a2_i; % 10 x 26 x 26 x 1 = 10 x 1
    a3_i = sigmoid(z3_i); % last output layer
    
    % implement backpropagation
    d3_i = a3_i - matrixY(i,:)'; % 10 x 1 - 10 x 1 = 10 x 1
    d2_i = (Theta2' * d3_i) .* sigmoidGradient([1; z2_i]); % 26 x 1 .*  26 x 1 = 26 x 1
    Theta2_grad = (Theta2_grad + d3_i * a2_i'); % 10 x 1 * 1 x 26 = 10 x 26
    Theta1_grad = (Theta1_grad + d2_i(2:end,:) * a1_i); % 25 x 1 * 1 x 401 =  25 x 401
end


% comment out this part when running part 3 for regularization
% Theta2_grad = Theta2_grad ./ m;
% Theta1_grad = Theta1_grad ./ m;


% part 3
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% for layer j equals to zero
Theta1_grad(:,1) = Theta1_grad(:,1)./m; % 25 x 1
Theta2_grad(:,1) = Theta2_grad(:,1)./m; % 10 x 1

% for layer j greater than zero
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)./m + (lambda / m) * Theta1(:,2:end); % 25 x 400 + 25 x 400
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)./m + (lambda / m) * Theta2(:,2:end); % 10 x 25 + 10 x 25

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
