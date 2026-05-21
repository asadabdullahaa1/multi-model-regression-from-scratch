import numpy as np

from models.wmmlr_model import WMMLR


def test_wmmlr_learns_reliability_weights_and_predicts_finite_values():
    rng = np.random.default_rng(42)
    X_left = rng.normal(loc=-2.0, scale=0.2, size=(40, 2))
    X_right = rng.normal(loc=2.0, scale=0.2, size=(40, 2))
    X = np.vstack([X_left, X_right])
    y = np.concatenate(
        [
            X_left[:, 0] + 0.5 * X_left[:, 1],
            -1.5 * X_right[:, 0] + X_right[:, 1],
        ]
    )

    X_train, y_train = X[:60], y[:60]
    X_val, y_val = X[60:], y[60:]

    model = WMMLR(k=2, lambda_reg=0.01, random_state=42)
    model.fit(X_train, y_train, X_val, y_val)
    pred = model.predict(X_val)
    analysis = model.get_cluster_analysis()

    assert len(model.local_models) == 2
    assert model.val_mse.shape == (2,)
    assert model.global_reliability.shape == (2,)
    assert np.isclose(model.global_reliability.sum(), 1.0)
    assert pred.shape == y_val.shape
    assert np.isfinite(pred).all()
    assert len(analysis) == 2
    assert {"cluster", "train_size", "train_mean", "train_std", "val_mse", "global_reliability"} <= set(analysis[0])
