import numpy as np

class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean
        _, S, Vt = np.linalg.svd(Xc, full_matrices = False)
        self.components = Vt[:self.k]
        var = S**2
        self.explained = var / var.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean
        