import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class LinearModel:
    def __init__(self):
        self.w = None

    def score(self, X):
        """
        Calculates the scores for each data point in X using s = X @ w.
        If self.w is None, initialize it randomly.

        Arguments:
            X (torch.Tensor): Feature matrix of shape (n, p)
                              Assumes last column is a bias term of 1s.

        Returns:
            s (torch.Tensor): Score vector of shape (n,)
        """
        if self.w is None:
            self.w = torch.rand(X.size(1)) # shape (p,)
        s = X @ self.w # shape (n,)
        return s


class LogisticRegression(LinearModel):
    def loss(self, X, y):
        """
        We calculte the average logistic loss.

        ars:
            X (torch.Tensor):feature matrix (n, p)
            y (torch.Tensor):target vector (n,) with labels in {0, 1}

        returns:
            loss (torch.Tensor):which is a scalar that is the average logistic loss
        """
        s = self.score(X)  # scores
        sigma = torch.sigmoid(s)  # sigma(s)
        loss = -torch.mean(y * torch.log(sigma + 1e-8) + (1 - y) * torch.log(1 - sigma + 1e-8))
        return loss

    def grad(self, X, y):
        """
        Calculates the gradient of the logistic loss.

        args:
            X (torch.Tensor): Feature matrix (n, p)
            y (torch.Tensor): Target vector (n,) with labels in {0, 1}

        returns:
            grad (torch.Tensor): Gradient vector with shape (p,)
        """
        s = self.score(X) # (n,)
        sigma = torch.sigmoid(s) # (n,)
        error = sigma - y # (n,)
        grad = X.T @ error / X.size(0) # (p,)
        return grad
class GradientDescentOptimizer:
    def __init__(self, model):
        """
        Initialize the optimizer with a model
        storing the previous weight vector w_{k-1} for momentum
        """
        self.model = model
        self.prev_w = None  # to store w_{k-1}

    def step(self, X, y, alpha=0.1, beta=0.9):
        """
        Perform one gradient descent step with momentum.

        Args:
            X : Feature matrix (n, p)
            y : Labels (n,)
            alpha: Learning rate for the gradient step
            beta: learning rate paramenter

        updates we do:
            self.model.w (torch.Tensor): Updated weights in-place
            self.prev_w (torch.Tensor): Stores previous weights for momentum
        """
        grad = self.model.grad(X, y)

        # If this is the first step, initialize prev_w to current w
        if self.prev_w is None:
            self.prev_w = self.model.w.clone()

        # Save current weights before updating
        w_current = self.model.w.clone()

        # Gradient descent with momentum update
        self.model.w = self.model.w - alpha * grad + beta * (self.model.w - self.prev_w)

        # Update previous weights for the next step
        self.prev_w = w_current
    