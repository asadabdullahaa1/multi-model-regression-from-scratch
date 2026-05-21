import numpy as np

from models.mmlr_model import MMLR
from utils.metrics import rmse


def test_mmlr_trains_local_models_and_predicts_finite_values():
    rng = np.random.default_rng(42)
    X_left = rng.normal(loc=-2.0, scale=0.2, size=(30, 2))
    X_right = rng.normal(loc=2.0, scale=0.2, size=(30, 2))
    X = np.vstack([X_left, X_right])
    y = np.concatenate(
        [
            1.5 * X_left[:, 0] - 0.5 * X_left[:, 1],
            -2.0 * X_right[:, 0] + 0.25 * X_right[:, 1],
        ]
    )

    model = MMLR(k=2, lambda_reg=0.01, random_state=42)
    model.fit(X, y)
    pred = model.predict(X)

    assert len(model.local_models) == 2
    assert set(model.cluster_info.keys()) == {0, 1}
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    assert rmse(y, pred) < 0.2
