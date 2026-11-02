"""Models, baselines and explainers for the tabular (Heart Disease) experiments.

"Removing" a tabular feature has no natural analogue of deleting a word, so a
feature is removed by replacing it with its training-set baseline (median for
continuous features, mode for categorical ones). This is used consistently by
the oracle, the Greedy explainer and the Parzen explainer.
"""
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn import ensemble, linear_model, neighbors, svm, tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import HEART_CATEGORICAL, HEART_COLS, HEART_LABELS

FEATURES = HEART_COLS[:-1]
CAT_IDX = [FEATURES.index(c) for c in HEART_CATEGORICAL]
NUM_IDX = [i for i in range(len(FEATURES)) if i not in CAT_IDX]


def preprocessor():
    return ColumnTransformer([
        ('num', StandardScaler(), NUM_IDX),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_IDX)])


def get_model(name, seed=1, l1_C=None):
    """Same model families as the text experiments, wrapped in a
    scale/one-hot pipeline so they accept the raw 13-column records."""
    est = {
        'logreg': lambda: linear_model.LogisticRegression(max_iter=1000),
        'neighbors': lambda: neighbors.KNeighborsClassifier(),
        'random_forest': lambda: ensemble.RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=seed, n_jobs=-1),
        'svm': lambda: svm.SVC(probability=True, kernel='rbf', C=10, gamma=0.001, random_state=seed),
        # the classifier used in the student's earlier Heart Disease script
        'rf_annot': lambda: ensemble.RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed),
        'tree': lambda: tree.DecisionTreeClassifier(max_depth=5, random_state=seed),
        'l1logreg': lambda: linear_model.LogisticRegression(penalty='l1', C=l1_C, solver='liblinear'),
    }[name]()
    if name in ('tree',):
        return est                       # raw features keep the tree's splits interpretable
    if name == 'l1logreg':
        return make_pipeline(StandardScaler(), est)   # one coefficient per original feature
    return make_pipeline(preprocessor(), est)


def baseline_values(Xtr):
    base = np.median(Xtr, axis=0)
    for j in CAT_IDX:
        vals, counts = np.unique(Xtr[:, j], return_counts=True)
        base[j] = vals[np.argmax(counts)]
    return base


def lime_explainer(Xtr, seed=1):
    cat_names = {}
    for name, labels in HEART_LABELS.items():
        j = FEATURES.index(name)
        mx = int(max(labels))
        cat_names[j] = [labels.get(v, str(v)) for v in range(mx + 1)]
    return LimeTabularExplainer(Xtr, feature_names=FEATURES, class_names=['no disease', 'disease'],
                                categorical_features=CAT_IDX, categorical_names=cat_names,
                                discretize_continuous=True, random_state=seed)


def explain_lime(explainer, x, predict_proba, K, num_samples):
    e = explainer.explain_instance(x, predict_proba, num_features=K, num_samples=num_samples, labels=(1,))
    pairs = e.as_map()[1]                                # [(feature_index, weight)]
    return pairs, float(e.intercept[1]), float(e.score), e.as_list(label=1)


def explain_random(rng, K):
    return [(int(f), 1) for f in rng.choice(len(FEATURES), K, replace=False)]


def explain_greedy(x, label, predict_proba, base, K):
    z = x.copy()
    cur = predict_proba(z[None])[0, label]
    exp, remaining = [], list(range(len(FEATURES)))
    while len(exp) < K and remaining:
        cand = np.repeat(z[None], len(remaining), 0)
        for r, f in enumerate(remaining):
            cand[r, f] = base[f]
        p = predict_proba(cand)[:, label]
        j = int(np.argmax(cur - p))
        if cur - p[j] < 0:
            break
        f = remaining.pop(j)
        exp.append((f, 1))
        cur = p[j]
        z[f] = base[f]
        if cur < .5:
            break
    return exp


class TabularParzen:
    """Parzen-window approximation of the classifier in the (scaled, one-hot)
    model input space. The contribution of original feature f is the
    first-order change in the Parzen estimate when f is reset to its baseline:
    grad . (x_baseline - x) restricted to f's columns."""

    def __init__(self, Xtr, cv_preds, base, sigma):
        self.pre = preprocessor().fit(Xtr)
        self.T = self.pre.transform(Xtr)
        self.T = self.T.toarray() if hasattr(self.T, 'toarray') else self.T
        self.y = np.asarray(cv_preds)
        self.sigma = sigma
        self.base = base
        # map transformed columns back to original features
        cols = []
        for j in NUM_IDX:
            cols.append([j])
        enc = self.pre.named_transformers_['cat']
        for j, cats in zip(CAT_IDX, enc.categories_):
            cols.extend([[j]] * len(cats))
        self.col_feature = np.array([c[0] for c in cols])

    def _t(self, x):
        t = self.pre.transform(x[None])
        return (t.toarray() if hasattr(t, 'toarray') else t)[0]

    def explain(self, x, K):
        t = self._t(x)
        d2 = ((self.T - t) ** 2).sum(1)
        a = -.5 * d2 / self.sigma ** 2
        k = np.exp(a - a.max())
        ones, zeros = self.y == 1, self.y == 0
        sk0, sk1 = k[zeros].sum(), k[ones].sum()
        minus = (self.T - t) * k[:, None]
        grad = (sk0 * minus[ones].sum(0) - sk1 * minus[zeros].sum(0)) / (self.sigma ** 2 * (sk0 + sk1) ** 2)
        delta = np.zeros_like(t)
        for f in range(len(FEATURES)):
            xb2 = x.copy(); xb2[f] = self.base[f]
            m = self.col_feature == f
            delta[m] = (self._t(xb2) - t)[m]
        contrib = np.zeros(len(FEATURES))
        for f in range(len(FEATURES)):
            m = self.col_feature == f
            contrib[f] = grad[m] @ delta[m]
        p1 = sk1 / (sk0 + sk1)
        order = np.argsort(-np.abs(contrib))[:K]
        return [(int(f), float(contrib[f])) for f in order], float(p1)


def choose_sigma_tab(T, cv_preds, grid=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)):
    D2 = ((T[:, None, :] - T[None, :, :]) ** 2).sum(-1)
    best, err_best = grid[0], np.inf
    for s in grid:
        A = -.5 * D2 / s ** 2
        np.fill_diagonal(A, -np.inf)
        A -= A.max(1, keepdims=True)
        K = np.exp(A)
        p1 = K[:, cv_preds == 1].sum(1) / K.sum(1)
        err = np.mean((p1 > .5) != cv_preds)
        if err < err_best:
            best, err_best = s, err
    return best
