import numpy as np

from utils.data_loader import load_airquality, load_bike, load_energy


def test_load_energy_returns_clean_scaled_arrays():
    X, y = load_energy()

    assert X.shape == (768, 8)
    assert y.shape == (768,)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-7)


def test_load_bike_removes_leakage_columns_and_scales_features():
    X, y = load_bike()

    assert X.shape == (17379, 12)
    assert y.shape == (17379,)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-7)


def test_load_airquality_imputes_missing_values_and_scales_features():
    X, y = load_airquality()

    assert X.shape == (9357, 12)
    assert y.shape == (9357,)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-7)
