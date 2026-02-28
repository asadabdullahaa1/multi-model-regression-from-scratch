import numpy as np
from models.kmeans import KMeansScratch
from models.linear_regression import LinearRegressionScratch

class MMLR:
    """
    Enhanced Multi-Model Linear Regression with better handling
    """

    def __init__(self, k=3, lr=0.01, epochs=1000, lambda_reg=0.01, random_state=42):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.lambda_reg = lambda_reg
        self.random_state = random_state

        self.cluster_model = None
        self.local_models = []
        self.cluster_info = {}  # Store metadata about clusters

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Cluster the dataset
        self.cluster_model = KMeansScratch(k=self.k, random_state=self.random_state)
        self.cluster_model.fit(X)
        labels = self.cluster_model.labels

        # Train local models
        self.local_models = []
        self.cluster_info = {}

        for c in range(self.k):
            X_c = X[labels == c]
            y_c = y[labels == c]

            # Store cluster information
            self.cluster_info[c] = {
                'size': len(X_c),
                'mean': np.mean(y_c) if len(y_c) > 0 else y.mean(),
                'std': np.std(y_c) if len(y_c) > 0 else y.std()
            }

            if len(X_c) < 2:  # Need at least 2 points
                # Create dummy model that predicts cluster mean
                model = LinearRegressionScratch(lr=self.lr, epochs=self.epochs,
                                               lambda_reg=self.lambda_reg)
                dummy_X = np.zeros((2, X.shape[1]))
                dummy_y = np.array([y.mean(), y.mean()])
                model.fit(dummy_X, dummy_y)
                self.local_models.append(model)
                continue

            # Train normal model
            model = LinearRegressionScratch(lr=self.lr, epochs=self.epochs,
                                           lambda_reg=self.lambda_reg)
            model.fit(X_c, y_c)
            self.local_models.append(model)

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        # Assign to clusters
        labels = self.cluster_model.predict(X)
        predictions = np.zeros(len(X))

        # Predict using respective local model
        for i in range(self.k):
            mask = (labels == i)
            if np.any(mask):
                predictions[mask] = self.local_models[i].predict(X[mask])

        return predictions