"""
Ablation for the one replication gap we could not explain directly: LIME's
recall for the Books sparse logistic regression (85.1 vs. 92.1 in the paper).
Hypothesis: which few words the L1 model keeps depends on the solver and on
the exact C, and LIME's recall is sensitive to that choice. We re-run only the
sparse-LR part of the faithfulness experiment on the authors' processed_acl
input with (i) the paper's C-selection rule under three solvers, and (ii) a
range of fixed C values, and report how LIME's recall moves.

Usage: python src/ablation_l1_solver.py
"""
import warnings

import numpy as np
from sklearn import linear_model

from common import save, vectorize
from data import load_text
from explainers import LimeText

warnings.filterwarnings('ignore')
K = 10


def lime_recall(clf, Xte, n_samples):
    nz = set(np.nonzero(clf.coef_.ravel())[0])
    lime = LimeText(num_samples=n_samples, seed=1)
    rec = []
    for i in range(Xte.shape[0]):
        x = Xte[i]
        gold = set(x.indices) & nz
        if gold:
            exp, _ = lime.explain(x, int(clf.predict(x)[0]), clf.predict_proba, K)
            rec.append(len(gold & {f for f, _ in exp}) / len(gold))
    return float(np.mean(rec)), len(rec), len(nz)


def paper_rule(Xtr, ytr, make):
    """lower C from 0.1 in 0.01 steps until no training doc uses > 10 features"""
    for c in np.arange(.1, 0, -.01):
        clf = make(c).fit(Xtr, ytr)
        used = clf.coef_.ravel() != 0
        if (Xtr[:, used] != 0).sum(1).max() <= 10:
            return clf, float(c)
    return clf, float(c)


def main(n_samples=5000):
    out = {}
    for d in ['books_acl', 'dvd_acl']:
        tr, ytr, te, yte = load_text(d)
        _, Xtr, Xte = vectorize(tr, te, lowercase=False)
        res = {}
        solvers = {'liblinear': lambda c: linear_model.LogisticRegression(penalty='l1', C=c, solver='liblinear'),
                   'saga': lambda c: linear_model.LogisticRegression(penalty='l1', C=c, solver='saga', max_iter=5000, random_state=1),
                   'liblinear_noint': lambda c: linear_model.LogisticRegression(penalty='l1', C=c, solver='liblinear', fit_intercept=False)}
        for name, make in solvers.items():
            clf, c = paper_rule(Xtr, ytr, make)
            r, n, nz = lime_recall(clf, Xte, n_samples)
            res[f'paper_rule_{name}'] = dict(C=round(c, 3), nonzero=nz, n_scored=n, lime_recall=r,
                                             acc=float((clf.predict(Xte) == yte).mean()))
            print(d, name, res[f'paper_rule_{name}'], flush=True)
        for c in [0.04, 0.05, 0.07, 0.1]:
            clf = solvers['liblinear'](c).fit(Xtr, ytr)
            r, n, nz = lime_recall(clf, Xte, n_samples)
            res[f'fixed_C_{c}'] = dict(C=c, nonzero=nz, n_scored=n, lime_recall=r,
                                       acc=float((clf.predict(Xte) == yte).mean()))
            print(d, 'C', c, res[f'fixed_C_{c}'], flush=True)
        out[d] = res
    out['note'] = 'LIME with N=5000 samples here (15000 in the main run); recall explained for the predicted label'
    save(out, 'ablation_l1_solver.json')


if __name__ == '__main__':
    main()
