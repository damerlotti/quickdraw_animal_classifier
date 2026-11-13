import numpy as np

class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean
        cov = Xc.T @ Xc
        eigvals, eigvecs = np.linalg.eigh(cov)          # sorted from smallest to largest
        order = np.argsort(eigvals)[::-1]               # flip to largest first
        self.components = eigvecs[:, order[:self.k]].T
        self.explained = eigvals[order] / eigvals.sum()
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, Z):
        return Z @ self.components + self.mean