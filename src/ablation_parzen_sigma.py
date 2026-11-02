"""
Ablation: is the Parzen gap in the faithfulness replication caused by the
authors' bandwidths (tuned on their pre-tokenised features) being wrong for
our raw-text vocabulary? Re-runs only the Parzen explainer of §5.2 on Books and
DVDs with sigma chosen on OUR features by the authors' own criterion
(least disagreement with the classifier's cross-validated predictions), and
also reports a sweep over sigma.

Usage: python src/ablation_parzen_sigma.py
"""
import warnings

import numpy as np
from sklearn import tree as sktree

from common import PAPER_SIGMAS_FIDELITY, SIGMA_GRID, choose_sigma, cv_predictions, save, vectorize
from data import load_text
from explainers import Parzen
from run_faithfulness import K, train_l1, tree_gold

warnings.filterwarnings('ignore')


def parzen_recall(Xtr, cvp, Xte, gold_fn, sigma):
    p = Parzen().fit(Xtr, cvp, sigma)
    rec = []
    for i in range(Xte.shape[0]):
        g = gold_fn(Xte[i])
        if g:
            rec.append(len(g & {f for f, _ in p.explain(Xte[i], K)[0]}) / len(g))
    return float(np.mean(rec))


def main():
    out = {}
    for d in ['books', 'dvd']:
        tr, ytr, te, yte = load_text(d)
        _, Xtr, Xte = vectorize(tr, te)
        for algo in ['l1logreg', 'tree']:
            if algo == 'l1logreg':
                clf, _ = train_l1(Xtr, ytr)
                nz = set(np.nonzero(clf.coef_.ravel())[0])
                gold_fn = lambda x, nz=nz: set(x.indices) & nz
            else:
                clf = sktree.DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)
                gold_fn = lambda x, clf=clf: tree_gold(clf, x)
            cvp = cv_predictions(clf, Xtr, ytr)
            s_cv, err = choose_sigma(Xtr, cvp)
            sweep = {str(s): parzen_recall(Xtr, cvp, Xte, gold_fn, s) for s in SIGMA_GRID}
            out[f'{d}_{algo}'] = dict(sigma_paper=PAPER_SIGMAS_FIDELITY[d][algo], sigma_cv=s_cv, cv_disagreement=err,
                                      recall_paper_sigma=sweep[str(PAPER_SIGMAS_FIDELITY[d][algo])],
                                      recall_cv_sigma=sweep[str(s_cv)], recall_best_sigma=max(sweep.values()),
                                      best_sigma=max(sweep, key=sweep.get), sweep=sweep)
            print(d, algo, {k: v for k, v in out[f'{d}_{algo}'].items() if k != 'sweep'}, flush=True)
    save(out, 'ablation_parzen_sigma.json')


if __name__ == '__main__':
    main()
