from models.kmeans import KMeansScratch
from models.linear_regression import LinearRegressionScratch

import numpy as np

class WMMLR:
    """
    Weighted Multiple-Model Linear Regression (WMMLR)
    """

    def __init__(
        self,
        k=3,
        lambda_reg=0.01,
        min_reliability=1e-3,
        random_state=42,
        top_m=2,           # use at most top-m experts per sample
        reliability_pow=1  # exponent on reliability (>=1 to emphasize best clusters)
    ):
        self.k = k
        self.lambda_reg = lambda_reg
        self.min_reliability = min_reliability
        self.random_state = random_state
        self.top_m = top_m
        self.reliability_pow = reliability_pow

        self.cluster_model = None
        self.local_models = []
        self.val_mse = None
        self.global_reliability = None
        self.cluster_info = {}

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = np.asarray(X_train, float)
        y_train = np.asarray(y_train, float).ravel()
        X_val = np.asarray(X_val, float)
        y_val = np.asarray(y_val, float).ravel()

        # 1. Cluster training data
        self.cluster_model = KMeansScratch(
            k=self.k,
            random_state=self.random_state
        )
        self.cluster_model.fit(X_train)
        train_labels = self.cluster_model.labels
        val_labels = self.cluster_model.predict(X_val)

        self.local_models = []
        val_mse_list = []

        # 2. Train local experts + compute per-cluster val MSE
        for c in range(self.k):
            X_c = X_train[train_labels == c]
            y_c = y_train[train_labels == c]

            # basic stats
            if len(X_c) > 0:
                mean_y = float(np.mean(y_c))
                std_y = float(np.std(y_c))
            else:
                mean_y = float(np.mean(y_train))
                std_y = float(np.std(y_train))

            self.cluster_info[c] = {
                "train_size": int(len(X_c)),
                "train_mean": mean_y,
                "train_std": std_y,
            }

            # handle tiny cluster
            if len(X_c) < 2:
                dummy = LinearRegressionScratch(lambda_reg=self.lambda_reg)
                dummy.fit(
                    np.zeros((2, X_train.shape[1])),
                    np.array([mean_y, mean_y])
                )
                self.local_models.append(dummy)
                val_mse_list.append(1e6)
                continue

            # train local ridge regression expert
            model = LinearRegressionScratch(lambda_reg=self.lambda_reg)
            model.fit(X_c, y_c)
            self.local_models.append(model)

            # val MSE for this cluster (only points whose nearest centroid is c)
            X_val_c = X_val[val_labels == c]
            y_val_c = y_val[val_labels == c]

            if len(X_val_c) == 0:
                mse = 1e6
            else:
                y_pred_c = model.predict(X_val_c)
                mse = float(np.mean((y_pred_c - y_val_c) ** 2))

            val_mse_list.append(mse)

        self.val_mse = np.array(val_mse_list)

        # 3. Global reliability from validation MSE
        eps = 1e-8
        inv = 1.0 / (self.val_mse + eps)          # larger = better
        inv = inv ** self.reliability_pow         # emphasize best clusters if pow>1

        inv_sum = inv.sum()
        if inv_sum == 0:
            rel = np.ones(self.k) / self.k
        else:
            rel = inv / inv_sum

        # enforce small floor
        rel = np.maximum(rel, self.min_reliability / self.k)
        rel /= rel.sum()

        self.global_reliability = rel

        # store in cluster_info
        for c in range(self.k):
            self.cluster_info[c]["val_mse"] = float(self.val_mse[c])
            self.cluster_info[c]["global_reliability"] = float(self.global_reliability[c])

    def _compute_memberships(self, X):
        """
        Compute soft membership using inverse squared distances
        """
        X = np.asarray(X, float)
        distances = self.cluster_model._compute_distances(X)  # (n, k)

        eps = 1e-8
        # inverse squared distance
        inv = 1.0 / (distances**2 + eps)

        row_sum = inv.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        memberships = inv / row_sum  # (n, k)

        return memberships

    def predict(self, X):
        X = np.asarray(X, float)
        n_samples = X.shape[0]

        # FIX: Initialize preds with correct shape
        preds = np.zeros((n_samples, self.k))

        # Get predictions from each expert
        for j, model in enumerate(self.local_models):
            preds[:, j] = model.predict(X)

        # local responsibilities
        memberships = self._compute_memberships(X)   # (n, k)

        # blend local membership with global reliability
        weights = memberships * self.global_reliability.reshape(1, -1)

        # --- keep only top-m experts for each sample ---
        if self.top_m is not None and self.top_m < self.k:
            # argsort descending along axis=1
            top_idx = np.argsort(-weights, axis=1)[:, :self.top_m]
            mask = np.zeros_like(weights, dtype=bool)
            rows = np.arange(n_samples)[:, None]
            mask[rows, top_idx] = True
            weights[~mask] = 0.0

        # normalize per sample
        w_sum = weights.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        weights /= w_sum

        # weighted sum of predictions
        y_pred = np.sum(weights * preds, axis=1)
        return y_pred

    def get_cluster_analysis(self):
        return [
            {
                "cluster": c,
                **self.cluster_info[c]
            }
            for c in range(self.k)
        ]
