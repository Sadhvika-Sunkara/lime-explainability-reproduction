"""
The paper's two simulated-user experiments re-run on NEW data: the UCI Heart
Disease (Cleveland) table, 297 patients x 13 clinical attributes.

  faithfulness  (Sec. 5.2) -- gold features known for an L1 logistic regression
                limited to <= K non-zero coefficients and a depth-5 decision
                tree; recall of gold features by each explainer's top K.
  trust         (Sec. 5.3) -- 25% of features (3 of 13) marked untrustworthy
                per round, 100 rounds; F1 on the trustworthy class.

Because a single 60-patient test split is too small to score, every patient is
explained by a model fitted on the other four folds of a stratified 5-fold
split (cross-fitting), so all 297 patients are test instances exactly once.
K = 5 explanation features (13 features in total; K = 10 would cover almost
every feature and make recall trivial).

Usage: python src/run_heart.py --experiment faithfulness
       python src/run_heart.py --experiment trust --algorithm random_forest
"""
import argparse
import time
import warnings

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import save
from data import load_heart
from heart_common import (FEATURES, TabularParzen, baseline_values,
                          choose_sigma_tab, explain_greedy, explain_lime,
                          explain_random, get_model, lime_explainer,
                          preprocessor)

warnings.filterwarnings('ignore')
EXPLAINERS = ['random', 'parzen', 'greedy', 'LIME']


def folds(y, seed=1):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def make_parzen(Xtr, ytr, cv_model, base):
    cvp = cross_val_predict(cv_model, Xtr, ytr, cv=5)
    T = preprocessor().fit(Xtr).transform(Xtr)
    T = T.toarray() if hasattr(T, 'toarray') else T
    sigma = choose_sigma_tab(T, cvp)
    return TabularParzen(Xtr, cvp, base, sigma), sigma


def tree_gold(clf, x):
    t, node, gold = clf.tree_, 0, set()
    while t.children_left[node] != -1:
        f = t.feature[node]; gold.add(int(f))
        node = t.children_left[node] if x[f] <= t.threshold[node] else t.children_right[node]
    return gold


def faithfulness(X, y, K, N):
    out = {}
    for algo in ['l1logreg', 'tree']:
        rec = {k: [] for k in EXPLAINERS}
        info = {'fold_C': [], 'fold_accuracy': [], 'gold_sizes': [], 'parzen_sigma': []}
        rng = np.random.RandomState(1)
        for tr, te in folds(y):
            Xtr, ytr = X[tr], y[tr]
            if algo == 'l1logreg':
                for c in np.arange(1.0, 0, -.005):
                    m = get_model('l1logreg', l1_C=c).fit(Xtr, ytr)
                    nz = set(np.nonzero(m[-1].coef_.ravel())[0].tolist())
                    if 0 < len(nz) <= K:
                        break
                info['fold_C'].append(float(c))
                gold_fn = lambda x, nz=nz: nz
            else:
                m = get_model('tree').fit(Xtr, ytr)
                gold_fn = lambda x, m=m: tree_gold(m, x)
            info['fold_accuracy'].append(float((m.predict(X[te]) == y[te]).mean()))
            base = baseline_values(Xtr)
            lime = lime_explainer(Xtr)
            parzen, s = make_parzen(Xtr, ytr, m, base); info['parzen_sigma'].append(s)
            for i in te:
                x = X[i]; gold = gold_fn(x)
                info['gold_sizes'].append(len(gold))
                lab = int(y[i])
                sets = {'LIME': {f for f, _ in explain_lime(lime, x, m.predict_proba, K, N)[0]},
                        'parzen': {f for f, _ in parzen.explain(x, K)[0]},
                        'greedy': {f for f, _ in explain_greedy(x, lab, m.predict_proba, base, K)},
                        'random': {f for f, _ in explain_random(rng, K)}}
                for k, s_ in sets.items():
                    rec[k].append(len(gold & s_) / len(gold))
        info['recall'] = {k: float(np.mean(v)) for k, v in rec.items()}
        info['recall_std'] = {k: float(np.std(v)) for k, v in rec.items()}
        info['mean_gold_size'] = float(np.mean(info.pop('gold_sizes')))
        out[algo] = info
        print('heart', algo, {k: round(100 * v, 1) for k, v in info['recall'].items()},
              'acc', np.round(info['fold_accuracy'], 3), 'gold', info['mean_gold_size'], flush=True)
    save(dict(dataset='heart', K=K, num_samples=N, n=len(y), **out), 'faithfulness_heart.json')


def trust(X, y, algo, K, N, rounds, pct):
    n, F = X.shape
    preds = np.zeros(n, int); probas = np.zeros(n)
    exps = {k: [None] * n for k in EXPLAINERS}
    models, bases = [None] * n, [None] * n
    rng = np.random.RandomState(1)
    accs, sigmas, fid = [], [], []
    t0 = time.time()
    fold_list = folds(y)
    for tr, te in fold_list:
        Xtr, ytr = X[tr], y[tr]
        m = get_model(algo).fit(Xtr, ytr)
        base = baseline_values(Xtr)
        lime = lime_explainer(Xtr)
        parzen, s = make_parzen(Xtr, ytr, get_model(algo), base); sigmas.append(s)
        accs.append(float((m.predict(X[te]) == y[te]).mean()))
        for i in te:
            x = X[i]
            preds[i] = m.predict(x[None])[0]
            probas[i] = m.predict_proba(x[None])[0, 1]
            pairs, icpt, score, _ = explain_lime(lime, x, m.predict_proba, K, N)
            fid.append(score)
            exps['LIME'][i] = (pairs, icpt)
            exps['parzen'][i] = parzen.explain(x, K)
            exps['random'][i] = explain_random(rng, K)
            exps['greedy'][i] = explain_greedy(x, int(preds[i]), m.predict_proba, base, K)
            models[i], bases[i] = m, base
    secs = time.time() - t0

    n_unt = max(1, int(F * pct))
    rr = np.random.RandomState(1)
    res = {k: {'precision': [], 'recall': [], 'f1': []} for k in EXPLAINERS}
    flipped = []
    same = lambda a, b: (a > .5) == (b > .5)
    for _ in range(rounds):
        unt = set(rr.choice(F, n_unt, replace=False).tolist())
        trustworthy = np.zeros(n, bool)
        for fi, (_, te) in enumerate(fold_list):               # one batched call per fold model
            Z = X[te].copy()
            for f in unt:
                Z[:, f] = bases[te[0]][f]
            trustworthy[te] = models[te[0]].predict(Z) == preds[te]
        flipped.append(int((~trustworthy).sum()))
        tp = {
            'LIME': np.array([same(ic + sum(w for _, w in e) - sum(w for f, w in e if f in unt), probas[i])
                              for i, (e, ic) in enumerate(exps['LIME'])]),
            'parzen': np.array([same(p + sum(w for f, w in e if f in unt), p) for e, p in exps['parzen']]),
            'random': np.array([not any(f in unt for f, _ in e) for e in exps['random']]),
            'greedy': np.array([not any(f in unt for f, _ in e) for e in exps['greedy']]),
        }
        for k in EXPLAINERS:
            t_, o_ = tp[k], trustworthy
            TP = np.sum(t_ & o_); FP = np.sum(t_ & ~o_); FN = np.sum(~t_ & o_)
            p = TP / (TP + FP) if TP + FP else 0.; r = TP / (TP + FN) if TP + FN else 0.
            res[k]['precision'].append(p); res[k]['recall'].append(r)
            res[k]['f1'].append(2 * p * r / (p + r) if p and r else 0.)
    summary = {}
    for k in EXPLAINERS:
        summary[k] = {m_: float(np.mean(v)) for m_, v in res[k].items()}
        summary[k].update({m_ + '_std': float(np.std(v)) for m_, v in res[k].items()})
        summary[k]['f1_pvalue_vs_LIME'] = (float(stats.ttest_ind(res[k]['f1'], res['LIME']['f1'])[1])
                                           if k != 'LIME' else None)
    out = dict(dataset='heart', algorithm=algo, K=K, num_samples=N, n_test=n,
               test_accuracy=float(np.mean(accs)), fold_accuracy=accs, parzen_sigma=sigmas,
               lime_local_r2_mean=float(np.mean(fid)), num_rounds=rounds, percent_untrustworthy=pct,
               n_untrustworthy=n_unt, flipped_mean=float(np.mean(flipped)),
               flipped_std=float(np.std(flipped)), explain_seconds=round(secs, 1), results=summary)
    print('heart', algo, f'acc={np.mean(accs):.3f}', f'flipped={np.mean(flipped):.1f}',
          {k: round(100 * summary[k]['f1'], 1) for k in EXPLAINERS}, flush=True)
    save(out, f'trust_heart_{algo}.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', choices=['faithfulness', 'trust'], required=True)
    ap.add_argument('--algorithm', default='random_forest')
    ap.add_argument('-k', '--num-features', type=int, default=5)
    ap.add_argument('--num-samples', type=int, default=15000)
    ap.add_argument('--num-rounds', type=int, default=100)
    ap.add_argument('-u', '--percent-untrustworthy', type=float, default=.25)
    a = ap.parse_args()
    X, y = load_heart()
    X = X.values
    if a.experiment == 'faithfulness':
        faithfulness(X, y, a.num_features, a.num_samples)
    else:
        trust(X, y, a.algorithm, a.num_features, a.num_samples, a.num_rounds, a.percent_untrustworthy)


if __name__ == '__main__':
    main()
