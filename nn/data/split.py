import numpy as np


def train_test_split(X, y=None, seed: int = 0, train_size: float | int = 0.8):
    n_samples = len(X)
    if y is not None and len(y) != n_samples:
        raise ValueError("Both X, and Y should have the same lenght")
    
    if isinstance(train_size, float):
        if not (0 <= train_size <= 1):
            raise ValueError("Train size should be between 0 and 1")
        absolute_train_size = int(n_samples * train_size)
    elif isinstance(train_size, int):
        if train_size < 0:
            raise ValueError("Unexpected negative value")
        absolute_train_size = int(train_size)
    else:
        raise TypeError("Invalid type")
      

    indices = np.arange(n_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_idx = indices[:absolute_train_size]
    test_idx = indices[absolute_train_size:]

    if y is None:
        return X[train_idx], X[test_idx]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
