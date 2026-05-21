import numpy as np
import pytest

from models.linear_regression import LinearRegressionScratch
from utils.metrics import mae, r2_score, rmse


def test_linear_regression_fits_simple_linear_relationship():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2.0 * X.ravel() + 3.0

    model = LinearRegressionScratch(lambda_reg=0.0)
    model.fit(X, y)
    pred = model.predict(X)

    assert np.allclose(pred, y, atol=1e-8)
    assert model.weights == pytest.approx([3.0, 2.0], abs=1e-8)
    assert rmse(y, pred) < 1e-8
    assert mae(y, pred) < 1e-8
    assert r2_score(y, pred) == pytest.approx(1.0)


def test_linear_regression_predict_requires_fit():
    model = LinearRegressionScratch()

    with pytest.raises(ValueError, match="not been fitted"):
        model.predict(np.zeros((2, 1)))
