%An octave function that computes the cost function for a linear regression problem


function J= CostFunction(x,y,theta)

m=size(x,1); % number of training examples

prediction=x*theta; %hypothesis

J=1/(2*m) * sum((prediction-y).^2);

