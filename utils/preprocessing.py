import numpy as np

def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    return (
        X[idx[:train_end]], y[idx[:train_end]],
        X[idx[train_end:val_end]], y[idx[train_end:val_end]],
        X[idx[val_end:]], y[idx[val_end:]]
    )
