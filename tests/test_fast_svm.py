"""Check that FastRBFSVC reproduces sklearn's SVC on Books test documents."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from common import get_classifier, vectorize
from data import load_text
from explainers import FastRBFSVC, LimeText


def test_fast_svm_matches_sklearn():
    tr, ytr, te, yte = load_text('books')
    _, Xtr, Xte = vectorize(tr, te)
    clf = get_classifier('svm').fit(Xtr, ytr)
    fast = FastRBFSVC(clf)
    # sklearn's predict_proba goes through libsvm's iterative coupling
    # (tolerance 0.005/k), so exact equality is not expected
    assert np.abs(fast.predict_proba(Xte)[:, 1] - clf.predict_proba(Xte)[:, 1]).max() < 0.02
    # also on LIME-style perturbations of one document
    _, Z, _ = LimeText(num_samples=500)._perturb(Xte[0])
    assert np.abs(fast.predict_proba(Z)[:, 1] - clf.predict_proba(Z)[:, 1]).max() < 0.02
    assert (fast.predict_proba(Xte).argmax(1) == clf.predict(Xte)).mean() >= 0.99


if __name__ == '__main__':
    test_fast_svm_matches_sklearn(); print('ok')
