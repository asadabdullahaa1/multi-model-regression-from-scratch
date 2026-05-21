import numpy as np

from models.kmeans import KMeansScratch


def test_kmeans_finds_two_clear_clusters():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
    )

    km = KMeansScratch(k=2, random_state=42)
    km.fit(X)

    assert km.centroids.shape == (2, 2)
    assert km.labels.shape == (6,)
    assert set(km.labels.tolist()) == {0, 1}
    assert km.inertia_ >= 0

    labels = km.predict(np.array([[0.0, 0.0], [10.0, 10.0]]))
    assert labels.shape == (2,)
    assert labels[0] != labels[1]
