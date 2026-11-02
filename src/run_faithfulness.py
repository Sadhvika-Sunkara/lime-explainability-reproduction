"""
Section 5.2 of Ribeiro et al. (2016): "Are explanations faithful to the model?"

Two interpretable classifiers whose truly-used ("gold") features are known:
  * l1logreg -- L1 logistic regression; C is lowered from 0.1 in steps of
                0.01 until no training document uses more than 10 features.
                Gold = instance's words with non-zero coefficient.
  * tree     -- DecisionTreeClassifier(random_state=1). Gold = words present in
                the instance that are tested on its root-to-leaf path.
Each explainer returns K=10 features; score = recall of gold features,
averaged over test instances with a non-empty gold set.

Usage:  python src/run_faithfulness.py --dataset books   (books | dvd | imdb)
"""
import argparse
import warnings
import time

import numpy as np
from sklearn import linear_model, tree as sktree

from common import (PAPER_SIGMAS_FIDELITY, choose_sigma, cv_predictions, save,
                    vectorize)
from data import load_text
from explainers import LimeText, Parzen, RandomExplainer, explain_greedy

K = 10
warnings.filterwarnings("ignore")  # sklearn deprecation noise for penalty="l1"


def train_l1(Xtr, ytr):
    for c in np.arange(.1, 0, -.01):
        clf = linear_model.LogisticRegression(penalty='l1', C=c, solver='liblinear')
        clf.fit(Xtr, ytr)
        used = clf.coef_.ravel() != 0
        if (Xtr[:, used] != 0).sum(1).max() <= 10:
            return clf, float(c)
    return clf, float(c)


def tree_gold(clf, x):
    t = clf.tree_
    present = set(x.indices)
    node, gold = 0, set()
    while t.children_left[node] != sktree._tree.TREE_LEAF:
        f = t.feature[node]
        if f in present:
            gold.add(f)
        node = t.children_left[node] if x[0, f] <= t.threshold[node] else t.children_right[node]
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--num-samples', type=int, default=15000)
    ap.add_argument('--max-test', type=int, default=None)
    a = ap.parse_args()

    tr, ytr, te, yte = load_text(a.dataset)
    _, Xtr, Xte = vectorize(tr, te)
    if a.max_test:
        Xte, yte = Xte[:a.max_test], yte[:a.max_test]
    out = {'dataset': a.dataset, 'num_samples': a.num_samples, 'n_test': Xte.shape[0]}

    for algo in ['l1logreg', 'tree']:
        if algo == 'l1logreg':
            clf, c = train_l1(Xtr, ytr)
            coef_nz = set(np.nonzero(clf.coef_.ravel())[0])
            gold_fn = lambda x: set(x.indices) & coef_nz
            info = {'C': c, 'n_nonzero_coef': len(coef_nz)}
        else:
            clf = sktree.DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)
            gold_fn = lambda x: tree_gold(clf, x)
            info = {'depth': int(clf.get_depth())}
        info['test_accuracy'] = float((clf.predict(Xte) == yte).mean())

        if a.dataset in PAPER_SIGMAS_FIDELITY:
            sigma, src = PAPER_SIGMAS_FIDELITY[a.dataset][algo], 'paper'
        else:
            sigma, _ = choose_sigma(Xtr, cv_predictions(clf, Xtr, ytr)); src = 'cv'
        parzen = Parzen().fit(Xtr, cv_predictions(clf, Xtr, ytr), sigma)
        info['parzen_sigma'], info['parzen_sigma_source'] = sigma, src

        lime = LimeText(num_samples=a.num_samples, seed=1)
        rnd = RandomExplainer(seed=1)
        rec = {k: [] for k in ['random', 'parzen', 'greedy', 'LIME']}
        gold_sizes = []
        t0 = time.time()
        for i in range(Xte.shape[0]):
            x = Xte[i]
            gold = gold_fn(x)
            if not gold:
                continue
            gold_sizes.append(len(gold))
            lab = int(yte[i])  # original code explains the true label
            sets = {
                'LIME': {f for f, _ in lime.explain(x, lab, clf.predict_proba, K)[0]},
                'parzen': {f for f, _ in parzen.explain(x, K)[0]},
                'greedy': {f for f, _ in explain_greedy(x, lab, clf.predict_proba, K)},
                'random': {f for f, _ in rnd.explain(x, K)},
            }
            for k, s in sets.items():
                rec[k].append(len(gold & s) / len(gold))
        info['n_scored'] = len(gold_sizes)
        info['mean_gold_size'] = float(np.mean(gold_sizes))
        info['seconds'] = round(time.time() - t0, 1)
        info['recall'] = {k: float(np.mean(v)) for k, v in rec.items()}
        info['recall_std'] = {k: float(np.std(v)) for k, v in rec.items()}
        out[algo] = info
        print(a.dataset, algo, {k: round(100 * v, 1) for k, v in info['recall'].items()},
              {k: v for k, v in info.items() if k not in ('recall', 'recall_std')}, flush=True)
    save(out, f'faithfulness_{a.dataset}.json')


if __name__ == '__main__':
    main()
