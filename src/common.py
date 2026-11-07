"""Shared helpers: vectorisation, classifiers, Parzen bandwidth selection."""
import json
import os

import numpy as np
from sklearn import ensemble, linear_model, neighbors, svm, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

# Parzen bandwidths published in the authors' code (evaluate_explanations.py
# and data_trusting.py). For datasets they did not use, sigma is chosen by the
# same criterion they used: minimum disagreement with the classifier's
# cross-validated predictions (find_sigma in parzen_windows.py).
PAPER_SIGMAS_FIDELITY = {'books': {'tree': 2.0, 'l1logreg': 2.0},
                         'dvd': {'tree': 8.0, 'l1logreg': 1.0}}
PAPER_SIGMAS_FIDELITY['books_acl'] = PAPER_SIGMAS_FIDELITY['books']
PAPER_SIGMAS_FIDELITY['dvd_acl'] = PAPER_SIGMAS_FIDELITY['dvd']
PAPER_SIGMAS_TRUST = {'books': {'neighbors': 0.5, 'svm': 7.0, 'logreg': 1.0, 'random_forest': 1.0},
                      'dvd': {'neighbors': 0.5, 'svm': 0.75, 'logreg': 0.75, 'random_forest': 0.5}}
PAPER_SIGMAS_TRUST['books_acl'] = PAPER_SIGMAS_TRUST['books']
PAPER_SIGMAS_TRUST['dvd_acl'] = PAPER_SIGMAS_TRUST['dvd']
SIGMA_GRID = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


def vectorize(train_docs, test_docs, lowercase=True):
    # original: CountVectorizer(lowercase=False, binary=True) on the pre-tokenised
    # processed_acl text (used for the *_acl datasets); raw text needs lowercase=True.
    v = CountVectorizer(lowercase=lowercase, binary=True)
    Xtr = v.fit_transform(train_docs).astype(float)
    Xte = v.transform(test_docs).astype(float)
    return v, Xtr, Xte


def get_classifier(name):
    """Exactly the settings in data_trusting.py:get_classifier."""
    if name == 'logreg':
        return linear_model.LogisticRegression(max_iter=1000)
    if name == 'random_forest':
        return ensemble.RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1, n_jobs=-1)
    if name == 'svm':
        return svm.SVC(probability=True, kernel='rbf', C=10, gamma=0.001, random_state=1)
    if name == 'tree':
        return tree.DecisionTreeClassifier(random_state=1)
    if name == 'neighbors':
        return neighbors.KNeighborsClassifier()
    raise ValueError(name)


def choose_sigma(X, cv_preds, grid=SIGMA_GRID):
    """Leave-one-out Parzen disagreement with the classifier's CV predictions."""
    X = X.tocsr() if hasattr(X, 'tocsr') else X
    sq = np.asarray(X.multiply(X).sum(1)).ravel() if hasattr(X, 'multiply') else (X ** 2).sum(1)
    G = X @ X.T
    G = G.toarray() if hasattr(G, 'toarray') else G
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    best, best_err = grid[0], np.inf
    for s in grid:
        A = -.5 * D2 / s ** 2
        np.fill_diagonal(A, -np.inf)
        A = A - A.max(1, keepdims=True)
        K = np.exp(A)
        p1 = K[:, cv_preds == 1].sum(1) / K.sum(1)
        err = np.mean((p1 > .5).astype(int) != cv_preds)
        if err < best_err:
            best, best_err = s, err
    return best, float(best_err)


def cv_predictions(clf, X, y):
    return cross_val_predict(clf, X, y, cv=5)


def save(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), 'w') as f:
        json.dump(obj, f, indent=2)
