import numpy as np
from src.pca import PCA


def stratified_kfold(y, k=5, seed=0):
    """Split row indices into k folds, each with the same mix of animals."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].extend(part)
    return [np.array(f) for f in folds]


def cross_validate(make_model, n_components, X, y, k=5):
    """Average accuracy over k folds, with PCA refit inside each fold."""
    scores = []
    for val_idx in stratified_kfold(y, k):
        train = np.ones(len(y), dtype=bool)
        train[val_idx] = False

        pca = PCA(n_components).fit(X[train])
        model = make_model().fit(pca.transform(X[train]), y[train])
        pred = model.predict(pca.transform(X[val_idx]))
        scores.append(np.mean(pred == y[val_idx]))
    return np.mean(scores), np.std(scores)


def cv_pca_curve(models, sizes, X, y, k=5):
    """CV accuracy for every model and PCA size, fitting PCA only once per fold."""
    max_n = max(sizes)
    scores = {name: np.zeros((len(sizes), k)) for name in models}

    for f, val_idx in enumerate(stratified_kfold(y, k)):
        train = np.ones(len(y), dtype=bool)
        train[val_idx] = False

        pca = PCA(max_n).fit(X[train])
        Z_tr = pca.transform(X[train])
        Z_val = pca.transform(X[val_idx])

        for name, make_model in models.items():
            for i, n in enumerate(sizes):
                # the first n columns = PCA with n components
                model = make_model().fit(Z_tr[:, :n], y[train])
                pred = model.predict(Z_val[:, :n])
                scores[name][i, f] = np.mean(pred == y[val_idx])
        print(f"fold {f + 1}/{k} done")

    return {name: (s.mean(axis=1), s.std(axis=1)) for name, s in scores.items()}

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)   # count each (true, predicted) pair
    return cm