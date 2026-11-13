import numpy as np

def sq_dists(A, B):
    return (A**2).sum(1)[:, None] - 2 * A @ B.T + (B**2).sum(1)[None, :]

class NMC:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = []
        for c in self.classes:
            c_drawings = X[y == c]        
            c_mean = np.mean(c_drawings, axis=0)
            self.means.append(c_mean)
        self.means = np.array(self.means)
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            best_class = None
            best_dist = np.inf
            for c, m in zip(self.classes, self.means):
                dist = np.sum((x - m) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_class = c
            predictions.append(best_class)
        return np.array(predictions)

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X, batch=1000):
        predictions = []
        for start in range(0, len(X), batch):
            d = sq_dists(X[start:start + batch], self.X)
            nearest = np.argpartition(d, self.k, axis=1)[:, :self.k]
            for neighbours in self.y[nearest]:
                votes = np.bincount(neighbours, minlength=10)
                predictions.append(np.argmax(votes))
        return np.array(predictions)

class LDA:
    def __init__(self, lam=1e-3):
        self.lam = lam

    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])

        cov = np.zeros((d, d))
        for c, m in zip(self.classes, self.means):
            diff = X[y == c] - m
            cov += diff.T @ diff
        cov = cov / (n - len(self.classes)) + self.lam * np.eye(d)

        self.W = np.linalg.solve(cov, self.means.T)
        priors = np.array([np.mean(y == c) for c in self.classes])
        self.b = -0.5 * np.sum(self.means * self.W.T, axis=1) + np.log(priors)
        return self

    def predict(self, X):
        scores = X @ self.W + self.b
        return self.classes[np.argmax(scores, axis=1)]