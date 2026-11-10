import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

animals = ["dolphin", "camel", "kangaroo", "dog", "crab", "sheep", "rabbit", "rhinoceros", "elephant", "hedgehog"]

def load_data(names, per_class=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for label, name in enumerate(names):
        data = np.load(DATA_DIR / f"{name}.npy", mmap_mode="r")
        idx = rng.choice(len(data), per_class, replace=False)
        X.append(data[idx].astype(np.float32)/255.0)
        y.append(np.full(per_class, label))
    return np.concatenate(X), np.concatenate(y)

def train_test_split(X, y, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]