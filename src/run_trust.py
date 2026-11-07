"""
Section 5.3 of Ribeiro et al. (2016): "Should I trust this prediction?"
(port of data_trusting.py, text datasets)

For each of 100 rounds, 25% of the vocabulary is marked "untrustworthy".
  Oracle: a test prediction is untrustworthy if zeroing those words in the
          document flips the classifier's prediction.
  LIME / Parzen user: distrusts a prediction if the explanation's local linear
          prediction crosses 0.5 once the untrustworthy words' weights are
          subtracted (same trust_fn as the original code).
  Greedy / Random user: distrusts a prediction if any untrustworthy word
          appears in the explanation.
Metric (as in the paper's Table 1): F1 for the *trustworthy* class, averaged
over rounds. Precision and recall are also reported. Explanations are
computed once per test document and reused across rounds, as in the original.

Usage: python src/run_trust.py --dataset books --algorithm logreg
       (dataset: books | dvd | imdb; algorithm: logreg | neighbors | random_forest | svm)
"""
import argparse
import time

import numpy as np
from scipy import stats

from common import (PAPER_SIGMAS_TRUST, choose_sigma, cv_predictions,
                    get_classifier, save, vectorize)
from data import load_text
from explainers import FastRBFSVC, LimeText, Parzen, RandomExplainer, explain_greedy

EXPLAINERS = ['random', 'parzen', 'greedy', 'LIME']


def prf(trust_pred, trustworthy):
    """precision/recall/F1 for the 'trustworthy' class (original's switched definition)."""
    tp = np.sum(trust_pred & trustworthy)
    fp = np.sum(trust_pred & ~trustworthy)
    fn = np.sum(~trust_pred & trustworthy)
    p = tp / (tp + fp) if tp + fp else 0.
    r = tp / (tp + fn) if tp + fn else 0.
    f = 2 * p * r / (p + r) if p and r else 0.
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--algorithm', required=True)
    ap.add_argument('--num-samples', type=int, default=15000)
    ap.add_argument('--num-features', '-k', type=int, default=10)
    ap.add_argument('--percent-untrustworthy', '-u', type=float, default=.25)
    ap.add_argument('--num-rounds', '-r', type=int, default=100)
    ap.add_argument('--max-test', type=int, default=None)
    a = ap.parse_args()
    K = a.num_features

    tr, ytr, te, yte = load_text(a.dataset)
    _, Xtr, Xte = vectorize(tr, te, lowercase=not a.dataset.endswith('_acl'))
    if a.max_test:
        Xte, yte = Xte[:a.max_test], yte[:a.max_test]
    np.random.seed(1)
    clf = get_classifier(a.algorithm).fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())

    cvp = cv_predictions(get_classifier(a.algorithm), Xtr, ytr)
    if a.dataset in PAPER_SIGMAS_TRUST:
        sigma, src = PAPER_SIGMAS_TRUST[a.dataset][a.algorithm], 'paper'
    else:
        sigma, _ = choose_sigma(Xtr, cvp); src = 'cv'
    parzen = Parzen().fit(Xtr, cvp, sigma)
    lime = LimeText(num_samples=a.num_samples, seed=1)
    rnd = RandomExplainer(seed=1)

    # exact fast path for the SVM's probability function (see FastRBFSVC)
    proba_fn = FastRBFSVC(clf).predict_proba if a.algorithm == 'svm' else clf.predict_proba
    preds = clf.predict(Xte)
    probas = clf.predict_proba(Xte)[:, 1]
    exps = {k: [] for k in EXPLAINERS}
    t0 = time.time()
    for i in range(Xte.shape[0]):
        x = Xte[i]
        exps['LIME'].append(lime.explain(x, 1, proba_fn, K))
        exps['parzen'].append(parzen.explain(x, K))
        exps['random'].append(rnd.explain(x, K))
        exps['greedy'].append(explain_greedy(x, int(preds[i]), proba_fn, K))
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{Xte.shape[0]} explained ({time.time() - t0:.0f}s)', flush=True)
    secs = time.time() - t0

    rng = np.random.RandomState(1)
    V = Xtr.shape[1]
    n_unt = int(V * a.percent_untrustworthy)
    res = {k: {'precision': [], 'recall': [], 'f1': []} for k in EXPLAINERS}
    flipped = []
    n = Xte.shape[0]
    for _ in range(a.num_rounds):
        unt = rng.choice(V, n_unt, replace=False)
        unt_set = set(unt.tolist())
        mask = np.ones(V); mask[unt] = 0
        t = Xte.multiply(mask[None, :]).tocsr()               # zero the untrustworthy words
        trustworthy = clf.predict(t) == preds                 # oracle
        flipped.append(int((~trustworthy).sum()))
        same_side = lambda a_, b_: (a_ > .5) == (b_ > .5)
        tp = {}
        tp['LIME'] = np.array([same_side(sum(w for _, w in e) + m - sum(w for f, w in e if f in unt_set), probas[i])
                               for i, (e, m) in enumerate(exps['LIME'])])
        tp['parzen'] = np.array([same_side(m - sum(w for f, w in e if f in unt_set), m)
                                 for e, m in exps['parzen']])
        tp['random'] = np.array([not any(f in unt_set for f, _ in e) for e in exps['random']])
        tp['greedy'] = np.array([not any(f in unt_set for f, _ in e) for e in exps['greedy']])
        for k in EXPLAINERS:
            p, r, f = prf(tp[k], trustworthy)
            res[k]['precision'].append(p); res[k]['recall'].append(r); res[k]['f1'].append(f)

    summary = {}
    for k in EXPLAINERS:
        summary[k] = {m: float(np.mean(v)) for m, v in res[k].items()}
        summary[k].update({m + '_std': float(np.std(v)) for m, v in res[k].items()})
        summary[k]['f1_pvalue_vs_LIME'] = (float(stats.ttest_ind(res[k]['f1'], res['LIME']['f1'])[1])
                                           if k != 'LIME' else None)
    out = dict(dataset=a.dataset, algorithm=a.algorithm, num_samples=a.num_samples,
               n_test=n, test_accuracy=acc, parzen_sigma=sigma, parzen_sigma_source=src,
               num_rounds=a.num_rounds, percent_untrustworthy=a.percent_untrustworthy,
               flipped_mean=float(np.mean(flipped)), flipped_std=float(np.std(flipped)),
               explain_seconds=round(secs, 1), results=summary)
    print(a.dataset, a.algorithm, f'acc={acc:.3f}', f'flipped={np.mean(flipped):.1f}',
          {k: round(100 * summary[k]['f1'], 1) for k in EXPLAINERS}, flush=True)
    save(out, f'trust_{a.dataset}_{a.algorithm}.json')


if __name__ == '__main__':
    main()
