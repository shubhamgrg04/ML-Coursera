function p = predict(theta1, theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(size(X, 1),1) X];

hidden = sigmoid(X*theta1');
hidden = [ones(size(hidden, 1),1) hidden];
h = sigmoid(hidden*theta2');
[~, p] = max(h');
p = p';

% =========================================================================


end
