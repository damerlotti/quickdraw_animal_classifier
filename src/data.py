import numpy as np

animals = ["dolphin", "camel", "kangaroo", "dog", "crab", "sheep", "rabbit", "rhinoceros", "elephant", "hedgehog"]
def load_data(names, per_class=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    y = []
    for label, name in enumerate(names):
        data = np.load(f"data/{name}.npy", mmap_mode = "r")
        idx = rng.choice(len(data), per_class, replace=False)
        X.append(data[idx].astype(np.float32)/255.0)
        y.append(np.full(per_class, label))
    return np.concatenate(X), np.concatenate(y)