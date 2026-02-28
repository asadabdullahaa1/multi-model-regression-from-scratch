import numpy as np

class KMeansScratch:


    def __init__(self, k=3, max_iter=300, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None  # For tracking convergence

    def fit(self, X):
        np.random.seed(self.random_state)

        # K-means++ initialization for better starting centroids
        self.centroids = self._kmeans_plusplus_init(X)

        for iteration in range(self.max_iter):
            # Assign points to nearest centroid
            distances = self._compute_distances(X)
            new_labels = np.argmin(distances, axis=1)

            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()

            # Update centroids
            new_centroids = []
            for i in range(self.k):
                cluster_points = X[new_labels == i]
                if len(cluster_points) == 0:
                    # Reinitialize empty cluster to furthest point
                    distances_to_centroids = self._compute_distances(X)
                    furthest_point_idx = np.argmax(distances_to_centroids.min(axis=1))
                    new_centroids.append(X[furthest_point_idx])
                else:
                    new_centroids.append(cluster_points.mean(axis=0))

            self.centroids = np.array(new_centroids)
            self.labels = new_labels

            # Calculate inertia (within-cluster sum of squares)
            self.inertia_ = self._calculate_inertia(X, new_labels)

            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids - old_centroids)
            if centroid_shift < self.tol:
                break

    def _kmeans_plusplus_init(self, X):
        """K-means++ initialization for better starting centroids"""
        n_samples = X.shape[0]
        centroids = []

        # Choose first centroid randomly
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx])

        # Choose remaining centroids
        for _ in range(1, self.k):
            # Calculate distances to nearest existing centroid
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids])
                                 for x in X])

            # Choose next centroid with probability proportional to distance^2
            # Guard against degenerate cases where all distances are zero.
            distances_sum = distances.sum()
            if not np.isfinite(distances_sum) or distances_sum <= 0:
                next_idx = np.random.randint(n_samples)
            else:
                probabilities = distances / distances_sum
                next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _compute_distances(self, X):
        """Compute distances from all points to all centroids"""
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - self.centroids[i], axis=1)
        return distances

    def _calculate_inertia(self, X, labels):
        """Calculate within-cluster sum of squares"""
        inertia = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i])**2)
        return inertia

    def predict(self, X):
        """Assign new points to nearest centroid"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

