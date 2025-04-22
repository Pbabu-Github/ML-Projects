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

class MyLinearRegression(LinearModel):
    def __init__(self):
        super().__init__()
    def predict(self, X):
        return self.score(X)

    def loss(self, X, y):
        y_hat = self.predict(X)
        return torch.mean((y - y_hat) ** 2)

class OverParameterizedLinearRegressionOptimizer:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # calculate the pseudoinverse and optimal weights
        X_pinv = torch.linalg.pinv(X)
        self.model.w = X_pinv @ y
