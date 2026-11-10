import numpy as np

class NMC:
    def fit(self, X, y):
        self.classes = np.unique(y) #labels
        self.means = []
        for c in self.classes:
            c_drawings = X[y == c]        
            c_mean = np.mean(c_drawings, axis=0)
            self.means.append(c_mean)
        self.means = np.array(self.means)    # shape (10, number of features)
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