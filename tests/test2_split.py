import numpy as np

from utils.preprocessing import train_val_test_split


def test_train_val_test_split_preserves_all_rows_and_expected_sizes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, train_ratio=0.6, val_ratio=0.2, seed=42
    )

    assert X_train.shape == (30, 2)
    assert X_val.shape == (10, 2)
    assert X_test.shape == (10, 2)
    assert y_train.shape == (30,)
    assert y_val.shape == (10,)
    assert y_test.shape == (10,)

    combined_y = np.concatenate([y_train, y_val, y_test])
    assert sorted(combined_y.tolist()) == list(range(50))


def test_train_val_test_split_is_reproducible_with_seed():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)

    first = train_val_test_split(X, y, seed=7)
    second = train_val_test_split(X, y, seed=7)

    for first_array, second_array in zip(first, second):
        assert np.array_equal(first_array, second_array)
