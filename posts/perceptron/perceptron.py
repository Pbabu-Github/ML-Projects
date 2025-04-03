import torch

class LinearModel:
    def __init__(self):
        #intializing weights to none; will set a value to them on first use
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
        s = X @ self.w # Matrix multiplication ( vectorized dot product for all points)
        return s

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 
        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        Returns:
            y_hat (torch.Tensor): Predictions in {0.0, 1.0}, shape (n,)
        """
        s = self.score(X)
        return (s >= 0).float()

class Perceptron(LinearModel):
    def loss(self, X, y):
        """
        Compute misclassification rate of predictions vs. true labels.
         Argument:
            X (torch.Tensor): Feature matrix (n, p)
            y (torch.Tensor): True labels in {0, 1} or {-1, 1}, shape (n,)

        Returns:
            misclassification_rate (torch.Tensor): Scalar tensor
        """
        s = self.score(X)
        y_ = y if ((y == 1) | (y == -1)).all() else (2 * y - 1)
        return ((s * y_) <= 0).float().mean()

    def grad(self, X, y):
        if len(X.shape) == 2:
            X = X.squeeze(0)  # shape: (p,)
        if isinstance(y, torch.Tensor):
            y = y.item()
        y_ = y if y in (-1, 1) else 2 * y - 1  # convert to ±1 if in {0, 1}

        s = torch.dot(self.w, X)  # ⟨w, x⟩
        
        if s * y_ < 0:
            return -y_ * X
        else:
            return torch.zeros_like(self.w)
    # def grad(self, X, y):
    #     """
    #    Computes the perceptron gradient for each single example.
    #     Arguments:
    #     X (torch.Tensor): Single feature vector of shape (p,) or (1, p)
    #     y (torch.Tensor): Single label in {0, 1} or {-1, 1}
    #     Returns:
    #     update (torch.Tensor): Gradient vector for weight update
    #     """
    #     if len(X.shape) == 2:
    #         X = X.squeeze(0)  # make sure the shape is (p,)
    #     y_ = y if y in (-1, 1) else 2 * y - 1

    #     # Compute score: s = <w, x>
    #     s = torch.dot(self.w, X)

    #     # If misclassified, return update vector
    #     if s * y_ < 0:
    #         return -y_ * X
    #     else:
    #         return torch.zeros_like(self.w)  # No update needed if classified correctly

class PerceptronOptimizer:
    def __init__(self, model):
        """
        Arguments:
            model (Perceptron): our perceptron model to be trained
        """
        self.model = model

    def step(self, X, y):
        """
        Does one step of the Perceptron learning rule:
        Sample one point (xi, yi), compute loss and gradient, update w.
        """
        # Choose a single random index
        idx = torch.randint(0, X.size(0), (1,)).item()
        xi = X[idx]
        yi = y[idx]

        # Compute loss (optional here, can be tracked externally)
        loss = self.model.loss(X, y)

        # Compute and apply gradient
        grad = self.model.grad(xi, yi)
        self.model.w = self.model.w - grad

        return loss