import numpy as np

class LinearRegressionScratch:
    """
    Linear Regression with L2 regularization (Ridge), using closed-form solution.

    This matches the formulation used in the MMLR paper:
        min_w  ||y - X w||^2 + λ ||w||^2
    solved by:
        w = (X^T X + λ I)^(-1) X^T y

    Implementation details:
    - We always add a bias term (intercept) as the first column of ones.
    - L2 regularization is applied to weights *except* the bias term.
    - We keep the same __init__ signature (lr, epochs, lambda_reg, tol)
      so existing code (MMLR / WMMLR) still works, but lr/epochs/tol
      are not used in the closed-form solver.
    """

    def __init__(self, lr=0.01, epochs=1000, lambda_reg=0.01, tol=1e-6):
        # lr, epochs, tol kept for API compatibility but not used in closed-form
        self.lr = lr
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.tol = tol

        self.weights = None          # shape (d+1,)
        self.loss_history = []       # store final training loss for analysis

    def fit(self, X, y):
        """
        Fit ridge regression model using closed-form solution.

            X: array-like, shape (n_samples, n_features)
            y: array-like, shape (n_samples,) or (n_samples, 1)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Add bias column
        Xb = self._add_bias(X)   # shape (n, d+1)
        n_samples, n_features = Xb.shape

        # Construct (X^T X + λ I) with NO regularization on bias weight
        XtX = Xb.T @ Xb
        Xty = Xb.T @ y

        I = np.eye(n_features)
        I[0, 0] = 0.0  # do not regularize bias term

        lambda_I = self.lambda_reg * I
        A = XtX + lambda_I

        # Solve for weights: (X^T X + λ I) w = X^T y
        try:
            self.weights = np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular / ill-conditioned
            self.weights = np.linalg.pinv(A) @ Xty

        # Optional: compute training loss for logging
        preds = Xb @ self.weights
        errors = preds - y
        mse = np.mean(errors ** 2)
        reg_term = 0.5 * self.lambda_reg * np.sum(self.weights[1:] ** 2)
        loss = mse + reg_term
        self.loss_history = [loss]

    def predict(self, X):
        """
        Predict using the learned ridge regression model.
        """
        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        return Xb @ self.weights

    def _add_bias(self, X):
        """
        Add a column of ones as the first column (bias / intercept).
        """
        X = np.asarray(X, dtype=float)
        return np.hstack([np.ones((X.shape[0], 1)), X])
