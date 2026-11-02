"""
Python 3 port of the explainers used in the original LIME experiments
(Ribeiro, Singh & Guestrin, KDD 2016; https://github.com/marcotcr/lime-experiments,
files explainers.py and parzen_windows.py).

The authors' code is frozen at Python 2.7 / numpy 1.14 / scikit-learn 0.18 and
does not install on a modern system. This module re-implements the same four
explainers with the same algorithms and hyperparameters, changing only what
was required for Python 3 / scikit-learn >= 1.0 and for speed:

  * LIME   -- GeneralizedLocalExplainer + data_labels_distances_mapping_text:
              perturb a bag-of-words vector by keeping a uniformly random
              number of its words, weight samples with the kernel
              sqrt(exp(-d^2 / rho^2)), d = 100 * cosine distance, rho = 25,
              select <= K features along the LARS-lasso path ("K-LASSO"),
              then refit the selected features by unregularised least squares.
              Perturbation is vectorised (same distribution as the original
              Python loop).
  * Parzen -- global Parzen-window approximation of the classifier, fitted on
              cross-validated predictions; explanation = K features with the
              largest absolute gradient of the Parzen estimate at x.
              Re-implemented with sparse algebra (the original densified the
              whole training matrix).
  * Greedy -- remove, one at a time, the word whose removal most decreases
              P(predicted class), until the prediction flips or K words.
              (explain_greedy_martens in the original.) Batched prediction.
  * Random -- K random words present in the instance.

Every explainer returns a list of (feature_index, weight) pairs; LIME and
Parzen additionally return a scalar "mean"/intercept so that the local linear
prediction can be reconstructed, exactly as in data_trusting.py.
"""
import numpy as np
import scipy.sparse as sp
from sklearn import linear_model
from sklearn.metrics.pairwise import cosine_distances


# ---------------------------------------------------------------- LIME (text)
class LimeText:
    def __init__(self, num_samples=15000, rho=25.0, seed=1):
        self.num_samples = num_samples
        self.rho = rho
        self.rng = np.random.RandomState(seed)

    def kernel(self, d):
        return np.sqrt(np.exp(-(d ** 2) / self.rho ** 2))

    def _perturb(self, x):
        """Vectorised version of data_labels_distances_mapping_text."""
        features = x.indices if sp.issparse(x) else np.nonzero(x)[0]
        features = np.sort(features)
        d = len(features)
        n = self.num_samples
        data = np.zeros((n, d))
        data[0] = 1
        if d > 1:
            # number of words to keep, s ~ U{1, ..., d-1} (as np.random.randint(1, d))
            s = self.rng.randint(1, d, n - 1)
            # keep a uniformly random subset of size s: rank random keys per row
            keys = self.rng.rand(n - 1, d)
            ranks = keys.argsort(axis=1).argsort(axis=1)
            data[1:] = (ranks < s[:, None]).astype(float)
        else:
            data[1:] = 1
        rows, cols = np.nonzero(data)
        inv = sp.csr_matrix((np.ones(len(rows)), (rows, features[cols])),
                            shape=(n, x.shape[1]))
        return data, inv, features

    def explain(self, x, label, predict_proba, num_features=10):
        data, inv, mapping = self._perturb(x)
        labels = predict_proba(inv)[:, label]
        distances = cosine_distances(inv[0], inv)[0] * 100
        weights = self.kernel(distances)
        mean = labels.mean()
        wdata = data * weights[:, None]
        wlab = (labels - mean) * weights
        used = np.arange(data.shape[1])
        if data.shape[1] > num_features:
            _, _, coefs = linear_model.lars_path(wdata, wlab, method='lasso')
            for i in range(coefs.shape[1] - 1, 0, -1):
                nz = coefs[:, i].nonzero()[0]
                if len(nz) <= num_features:
                    used = nz
                    break
        if len(used) == 0:
            return [], mean
        model = linear_model.Ridge(alpha=0, fit_intercept=False)
        model.fit(wdata[:, used], wlab)
        exp = sorted(zip(mapping[used], model.coef_), key=lambda t: -abs(t[1]))
        return exp, mean


# ---------------------------------------------------------------- Random
class RandomExplainer:
    def __init__(self, seed=1):
        self.rng = np.random.RandomState(seed)

    def explain(self, x, num_features=10):
        nz = x.indices
        if len(nz) == 0:
            return []
        # original samples WITH replacement (np.random.choice default)
        return [(int(f), 1) for f in self.rng.choice(nz, num_features)]


# ---------------------------------------------------------------- Greedy
def explain_greedy(x, label, predict_proba, num_features=10):
    """explain_greedy_martens: remove the word that most reduces
    P(label) until P(label) < 0.5 or K words chosen."""
    z = sp.csr_matrix(x, copy=True).astype(float)
    cur = predict_proba(z)[0, label]
    exp = []
    while len(exp) < num_features:
        nz = z.indices.copy()
        if len(nz) == 0:
            break
        # batch: one row per candidate removal
        reps = sp.vstack([z] * len(nz)).tolil()
        reps[np.arange(len(nz)), nz] = 0
        p = predict_proba(reps.tocsr())[:, label]
        change = cur - p
        j = int(np.argmax(change))
        if change[j] < 0:
            break
        exp.append((int(nz[j]), 1))
        cur -= change[j]
        if cur < .5:
            break
        z = z.tolil(); z[0, nz[j]] = 0; z = z.tocsr(); z.eliminate_zeros()
    return exp


# ---------------------------------------------------------------- Parzen
class Parzen:
    """Parzen-window classifier on binary bag-of-words (or dense) vectors."""

    def fit(self, X, y, sigma):
        self.X = sp.csr_matrix(X).astype(float)
        self.sq = np.asarray(self.X.multiply(self.X).sum(1)).ravel()
        self.y = np.asarray(y)
        self.sigma = sigma
        return self

    def _ker(self, x):
        x = sp.csr_matrix(x).astype(float)
        d2 = self.sq + x.multiply(x).sum() - 2 * np.asarray((self.X @ x.T).todense()).ravel()
        d2 = np.maximum(d2, 0)
        # log-sum-exp for numerical safety; constant factors cancel in ratios
        a = -.5 * d2 / self.sigma ** 2
        return np.exp(a - a.max())

    def predict_proba1(self, x):
        k = self._ker(x)
        return k[self.y == 1].sum() / k.sum()

    def explain(self, x, num_features=10):
        x = sp.csr_matrix(x).astype(float)
        k = self._ker(x)
        ones, zeros = self.y == 1, self.y == 0
        sk0, sk1 = k[zeros].sum(), k[ones].sum()
        feats = x.indices
        xv = x.data
        Xc = self.X[:, feats].toarray()            # n x |feats|
        minus = Xc - xv[None, :]                   # (X_i - x) restricted to feats
        t = minus * k[:, None]
        st0, st1 = t[zeros].sum(0), t[ones].sum(0)
        tot = sk0 + sk1
        grad = (sk0 * st1 - sk1 * st0) / (self.sigma ** 2 * tot ** 2)
        exp = sorted(zip(feats, grad), key=lambda t: -abs(t[1]))[:num_features]
        return [(int(f), float(w)) for f, w in exp], sk1 / tot


# ---------------------------------------------------------------- fast exact SVM
class FastRBFSVC:
    """Exact, faster predict_proba for a fitted binary sklearn SVC(kernel='rbf')
    on samples whose non-zeros lie inside a small feature subset (LIME's
    perturbations of one document). ||z - s||^2 = |z|^2 + |s|^2 - 2 z.s, and
    z.s only involves the document's own words. Verified against
    SVC.decision_function / predict_proba in tests/test_fast_svm.py."""

    def __init__(self, svc):
        self.svc = svc
        self.SV = sp.csr_matrix(svc.support_vectors_)
        self.sv_sq = np.asarray(self.SV.multiply(self.SV).sum(1)).ravel()
        dc = svc.dual_coef_
        self.dual = np.asarray(dc.todense() if sp.issparse(dc) else dc).ravel()
        self.b = svc.intercept_[0]
        self.gamma = svc._gamma
        self.A, self.B = svc.probA_[0], svc.probB_[0]

    def predict_proba(self, Z):
        Z = sp.csr_matrix(Z)
        cols = np.unique(Z.indices)
        Zs = Z[:, cols]
        dot = (Zs @ self.SV[:, cols].T).toarray()
        z_sq = np.asarray(Zs.multiply(Zs).sum(1)).ravel()
        d2 = z_sq[:, None] + self.sv_sq[None, :] - 2 * dot
        dec = np.exp(-self.gamma * d2) @ self.dual + self.b      # sklearn decision_function
        # Platt scaling as stored by libsvm: P(y = classes_[1]) = 1 / (1 + exp(A*dec + B)).
        # sklearn's own predict_proba passes this through libsvm's iterative
        # pairwise coupling (stopping tolerance 0.005/k), so the two agree to
        # within ~0.007; decision values agree to ~1e-11.
        p1 = 1. / (1. + np.exp(self.A * dec + self.B))
        return np.column_stack([1 - p1, p1])
