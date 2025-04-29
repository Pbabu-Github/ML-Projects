import torch

class KernelLogisticRegression:
    def __init__(self, kernel_function, lam=0.1, **kernel_params):
        # Initialize the kernel logistic regression model.
        self.kernel_function = kernel_function
        self.kernel_params = kernel_params
        self.lam = lam
        self.X_train = None  # to save the training data
        self.a = None        # dual weights

    def score(self, X, recompute_kernel=False):
        """
        Compute s = K(X, X_train)^T @ a
        """
        if recompute_kernel or self.K_train is None:
            K = self.kernel_function(X, self.X_train, **self.kernel_params)
        else:
            K = self.K_train
        return K @ self.a

    def loss(self, X, y):
        """
        Logistic loss + L1 regularization.
        """
        s = self.score(X)
        sigma = torch.sigmoid(s)
        log_loss = -torch.mean(y * torch.log(sigma + 1e-8) + (1 - y) * torch.log(1 - sigma + 1e-8))
        reg = self.lam * torch.norm(self.a, p=1)
        return log_loss + reg

    def grad(self, X, y):
        """
        Gradient of the logistic loss + L1 regularization.
        """
        s = self.score(X)
        sigma = torch.sigmoid(s)
        error = sigma - y  #has shape (n,)
        grad_loss = self.K_train.T @ error / X.size(0)  # (n,)
        grad_reg = self.lam * torch.sign(self.a)  # L1 gradient
        return grad_loss + grad_reg

    def fit(self, X, y, m_epochs=500000, lr=1e-4):
        """
        Fit using gradient descent.
        """
        self.X_train = X
        self.K_train = self.kernel_function(X, X, **self.kernel_params)
        self.a = torch.zeros(X.shape[0])  # dual weights

        for epoch in range(m_epochs):
            grad = self.grad(X, y)
            self.a = self.a - lr * grad  # Simple gradient descent
