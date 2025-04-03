import torch
import matplotlib.pyplot as plt

class LinearModel:
    def __init__(self):
        # Initialize weights to None; they will be set on first use.
        self.w = None 

    def score(self, X):
        """
        Calculates the score for each data point in X using the dot product w adn x.
        If weights are uninitialized then we randomly initialize them.
        Args:
             X (torch.Tensor): Feature matrix of shape (n, p)

        Returns:
            s (torch.Tensor): Score vector of shape (n,)
        """
        if self.w is None:
            self.w = torch.rand(X.size()[1])
        s = X @ self.w  #the dot product for all data points.
        return s

    def predict(self, X):
        """
        Compute predictions for each data point in the feature matrix X.
        The prediction is 1 if the score is >= 0 and 0 otherwise.
        
        argument: 
            X, torch.Tensor: Feature matrix with shape (n, p).
                             Assumes the final column is a constant column of 1s.
        Returns: 
            y_hat, torch.Tensor: Vector of predictions in {0.0, 1.0}, shape (n,)
        """
        s = self.score(X)
        return (s >= 0).float()


class Perceptron(LinearModel):
    def loss(self, X, y):
        """
        Compute the misclassification rate. A data point is classified correctly if s[i] * y[i]_ > 0,
        where y[i]_ is the modified value in {-1, 1}. If y is not already in {-1, 1},
        we conver it using: y_ = 2*y - 1.
        
        argument: 
            X, torch.Tensor: Feature matrix with shape (n, p).
                             Assumes the final column is a constant column of 1s.
            y, torch.Tensor: Target vector with shape (n,). The possible labels are {0, 1} or {-1, 1}.
        returns:
            misclassification_rate, torch.Tensor: Scalar tensor with the misclassification rate.
        """
        s = self.score(X)
        # Convert y to {-1, 1} if we need to
        if ((y == 1) | (y == -1)).all():
            y_mod = y
        else:
            y_mod = 2 * y - 1
        return ((s * y_mod) <= 0).float().mean()

    def grad(self, X, y):
        """
        calculate the perceptron gradient for a single example.
        If the data point is misclassified (s * y < 0), then the gradient is -y * x orelse the gradient would just be a zero vector.
        
        argument:
            X, torch.Tensor: Single feature vector of shape (p,) or (1, p).
            y, torch.Tensor or scalar: Single label in {0, 1} or {-1, 1}.
        returns:
            grad, torch.Tensor: Gradient vector for weight update.
        """
        # we make sure that X is a 1D tensor
        if len(X.shape) == 2:
            X = X.squeeze(0)
        # Convert y to {-1, 1} if we have to
        y_mod = y if y in (-1, 1) else 2 * y - 1

        #calculate the score for the single example
        s = torch.dot(self.w, X)
        if s * y_mod < 0:
            return -y_mod * X
        else:
            return torch.zeros_like(self.w) # we dont need to update if it is classified correctly


class PerceptronOptimizer:
    def __init__(self, model):
        """
        argument:
            model (Perceptron): The perceptron model to be trained.
        """
        self.model = model 

    def step(self, X, y):
        """
        Perform a step of the Perceptron learning rule:
        calculate the current loss, then we caluclate the gradient, and then we update the weights.
        
        argument:
            X, torch.Tensor: Feature matrix (or single example) with shape (n, p) or (1, p).
            y, torch.Tensor: Target vector (or single label) with shape (n,) or scalar.
        
        Returns:
            loss, torch.Tensor: The misclassification rate after the update.
        """
        loss = self.model.loss(X, y)
        grad = self.model.grad(X, y)
        self.model.w = self.model.w - grad
        return loss


def perceptron_data(n_points=300, noise=0.2):
    """
    This function is just to create a dataset for training the perceptron.( used inclass notes for help for this)
    
    arguments:
        n_points (int): Total number of data points.
        noise (float): Standard deviation of the noise added to the features.
    
    returns:
        X, torch.Tensor: Feature matrix of shape (n_points, 3). The final column is a constant 1.
        y, torch.Tensor: Target vector with values in {-1, 1}, shape (n_points,).
    """
    torch.manual_seed(1234)
    # Create binary labels: first half 0, second half 1, then convert to boolean.
    y = torch.arange(n_points) >= int(n_points / 2)
    # Generate features with noise; here we add y[:, None] to shift one of the classes.
    X = y[:, None].float() + torch.normal(0.0, noise, size=(n_points, 2))
    # Append a constant column (for the bias term).
    X = torch.cat((X, torch.ones((X.shape[0], 1))), 1)
    # Convert y from {0, 1} to {-1, 1}
    y = 2 * y.float() - 1
    return X, y


def plot_perceptron_data(X, y, ax): #used inclass notes help for creating this function too
    """
    Plot 2D perceptron data.
    
    argument:
        X, torch.Tensor: Feature matrix of shape (n, 3) (the final column is a constant 1).
        y, torch.Tensor: Target vector with values in {-1, 1}, shape (n,).
        ax, matplotlib.axes.Axes: Axes object on which to plot.
    """
    assert X.shape[1] == 3, "This function only works for data with 2 features and a constant column."
    for label, color in zip([-1, 1], ['blue', 'red']):
        indices = (y == label).nonzero(as_tuple=True)[0]
        ax.scatter(X[indices, 0], X[indices, 1], color=color, label=f'Class {label}')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()



