"""
Agreement and validity analysis for the explanation-trust annotation set.

Inputs:  annotation/responses/annotator_{A,B,C}.xlsx  (completed sheets)
         annotation/llm_judgements.csv                (from llm_judge.py)
         annotation/answer_key.csv
Outputs: results/annotation_analysis.json, results/annotations_merged.csv

Reported:
  * inter-annotator agreement: Krippendorff's alpha (ordinal for Q1 trust,
    nominal for Q2), and pairwise quadratic-weighted Cohen's kappa for Q1
  * LLM judge vs humans: quadratic-weighted kappa against each annotator and
    against the human median; Cohen's kappa on Q2; LLM self-consistency
  * validity: do trust ratings separate correct from incorrect predictions?
    (ROC AUC of trust for 'prediction is correct'); relation to LIME's local
    fidelity (Spearman correlation with local R^2)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.join(HERE, '..', 'annotation')
RES = os.path.join(HERE, '..', 'results')


def krippendorff_alpha(data, level='ordinal', categories=None):
    """data: raters x items array with np.nan for missing. Coincidence-matrix
    formulation (Krippendorff 2011)."""
    data = np.asarray(data, float)
    vals = np.unique(data[~np.isnan(data)]) if categories is None else np.asarray(categories, float)
    idx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    o = np.zeros((k, k))
    for col in data.T:
        c = col[~np.isnan(col)]
        m = len(c)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[idx[c[i]], idx[c[j]]] += 1. / (m - 1)
    n_c = o.sum(1); n = n_c.sum()
    if level == 'nominal':
        d = 1 - np.eye(k)
    else:  # ordinal metric
        d = np.zeros((k, k))
        for a in range(k):
            for b in range(k):
                lo, hi = min(a, b), max(a, b)
                d[a, b] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    Do = (o * d).sum()
    De = (np.outer(n_c, n_c) * d).sum() / (n - 1)
    return float(1 - Do / De) if De > 0 else float('nan')


def read_sheet(path):
    df = pd.read_excel(path, sheet_name='Items', header=0)
    df = df[df.iloc[:, 0].astype(str).str.match(r'H\d\d')]
    return pd.DataFrame({'item_id': df.iloc[:, 0].values,
                         'trust': pd.to_numeric(df.iloc[:, 5], errors='coerce').values,
                         'sense': df.iloc[:, 6].map(lambda v: {'yes': 1, 'no': 0}.get(str(v).strip().lower(), np.nan)).values})


def qwk(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(cohen_kappa_score(a[m].astype(int), b[m].astype(int), weights='quadratic', labels=[1, 2, 3, 4, 5]))


def kappa(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(cohen_kappa_score(a[m].astype(int), b[m].astype(int)))


def main(resp_dir=os.path.join(ANN, 'responses'), out_prefix=''):
    key = pd.read_csv(os.path.join(ANN, 'answer_key.csv'))
    files = sorted(glob.glob(os.path.join(resp_dir, 'annotator_*.xlsx')))
    if not files:
        sys.exit(f'no completed sheets in {resp_dir}')
    humans = {os.path.basename(f)[10:-5]: read_sheet(f).set_index('item_id').loc[key.item_id] for f in files}
    names = sorted(humans)
    T = np.vstack([humans[h].trust.values for h in names])
    S = np.vstack([humans[h].sense.values for h in names])
    out = {'annotators': names, 'n_items': int(len(key)),
           'completion': {h: float(np.mean(~np.isnan(humans[h].trust.values))) for h in names}}
    out['krippendorff_alpha_trust_ordinal'] = krippendorff_alpha(T, 'ordinal', [1, 2, 3, 4, 5])
    out['krippendorff_alpha_trust_interval_check'] = None
    out['krippendorff_alpha_sense_nominal'] = krippendorff_alpha(S, 'nominal', [0, 1])
    out['pairwise_qwk_trust'] = {f'{a}-{b}': qwk(T[i], T[j]) for i, a in enumerate(names)
                                 for j, b in enumerate(names) if i < j}
    out['pairwise_kappa_sense'] = {f'{a}-{b}': kappa(S[i], S[j]) for i, a in enumerate(names)
                                   for j, b in enumerate(names) if i < j}
    h_med = np.round(np.nanmedian(T, 0))
    h_mean = np.nanmean(T, 0)
    s_major = (np.nanmean(S, 0) >= .5).astype(float)
    merged = key.copy()
    for h, t_, s_ in zip(names, T, S):
        merged[f'trust_{h}'] = t_; merged[f'sense_{h}'] = s_
    merged['trust_human_mean'] = h_mean; merged['trust_human_median'] = h_med
    merged['sense_human_majority'] = s_major

    llm_path = os.path.join(ANN, 'llm_judgements.csv')
    if os.path.exists(llm_path):
        L = pd.read_csv(llm_path)
        out['llm_model'] = str(L.model.iloc[0]); out['llm_repeats'] = int(L.repeat.max())
        reps = [L[L.repeat == r].set_index('item_id').loc[key.item_id] for r in sorted(L.repeat.unique())]
        RT = np.vstack([r.trust.values for r in reps]).astype(float)
        RS = np.vstack([(r.makes_sense == 'Yes').astype(int).values for r in reps]).astype(float)
        lt = np.round(np.median(RT, 0)); ls = (RS.mean(0) >= .5).astype(float)
        merged['trust_llm'] = lt; merged['sense_llm'] = ls
        out['llm_self_consistency_alpha_trust'] = krippendorff_alpha(RT, 'ordinal', [1, 2, 3, 4, 5]) if len(reps) > 1 else None
        out['llm_vs_human_median_qwk'] = qwk(lt, h_med)
        out['llm_vs_each_human_qwk'] = {h: qwk(lt, T[i]) for i, h in enumerate(names)}
        out['llm_vs_human_majority_kappa_sense'] = kappa(ls, s_major)
        out['llm_vs_human_mean_spearman'] = float(spearmanr(lt, h_mean, nan_policy='omit')[0])
        # alpha when the LLM is added as a 4th rater: does it behave like another human?
        out['krippendorff_alpha_trust_humans_plus_llm'] = krippendorff_alpha(np.vstack([T, lt]), 'ordinal', [1, 2, 3, 4, 5])
        out['mean_trust_llm'] = float(np.mean(lt))
    out['mean_trust_human'] = float(np.nanmean(T))

    correct = key.correct.values
    def auc(score):
        m = ~np.isnan(score)
        return float(roc_auc_score(correct[m], score[m]))
    out['auc_trust_detects_correct'] = {**{h: auc(T[i]) for i, h in enumerate(names)}, 'human_mean': auc(h_mean)}
    out['mean_trust_by_correctness'] = {'correct': float(np.nanmean(h_mean[correct == 1])),
                                        'incorrect': float(np.nanmean(h_mean[correct == 0]))}
    out['sense_rate_by_correctness_human'] = {'correct': float(np.nanmean(S[:, correct == 1])),
                                              'incorrect': float(np.nanmean(S[:, correct == 0]))}
    conf = np.maximum(key.p_disease.values, 1 - key.p_disease.values)
    out['auc_model_confidence_detects_correct'] = auc(conf)
    out['mannwhitney_p_trust_correct_vs_incorrect'] = float(mannwhitneyu(h_mean[correct == 1], h_mean[correct == 0]).pvalue)
    rng = np.random.RandomState(1); boots = []
    for _ in range(2000):
        i = rng.randint(0, len(correct), len(correct))
        if len(set(correct[i])) == 2:
            boots.append(roc_auc_score(correct[i], h_mean[i]))
    out['auc_human_mean_ci95'] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    # does trust carry information beyond the confidence shown on screen?
    resid = h_mean - np.polyval(np.polyfit(conf, h_mean, 1), conf)
    out['auc_human_trust_residual_after_confidence'] = auc(resid)
    out['p_human_trust_residual'] = float(mannwhitneyu(resid[correct == 1], resid[correct == 0]).pvalue)
    # robustness: annotator A is the author (who designed the study); repeat without her
    if 'A' in names and len(names) > 2:
        keep = [i for i, h in enumerate(names) if h != 'A']
        T2 = T[keep]; m2 = np.nanmean(T2, 0)
        r2 = m2 - np.polyval(np.polyfit(conf, m2, 1), conf)
        out['without_author'] = {'alpha_trust': krippendorff_alpha(T2, 'ordinal', [1, 2, 3, 4, 5]),
                                 'auc_trust': auc(m2), 'auc_residual_after_confidence': auc(r2),
                                 'spearman_trust_vs_confidence': float(spearmanr(m2, conf)[0])}
    out['spearman_human_trust_vs_lime_r2'] = float(spearmanr(h_mean, key.lime_local_r2, nan_policy='omit')[0])
    out['spearman_human_trust_vs_confidence'] = float(spearmanr(h_mean, np.maximum(key.p_disease, 1 - key.p_disease), nan_policy='omit')[0])
    if 'trust_llm' in merged:
        lt = merged.trust_llm.values
        out['auc_trust_detects_correct']['llm'] = auc(lt)
        out['mean_trust_by_correctness_llm'] = {'correct': float(lt[correct == 1].mean()),
                                                'incorrect': float(lt[correct == 0].mean())}
        out['spearman_llm_trust_vs_lime_r2'] = float(spearmanr(lt, key.lime_local_r2)[0])
        out['spearman_llm_trust_vs_confidence'] = float(spearmanr(lt, conf)[0])
        lres = lt - np.polyval(np.polyfit(conf, lt, 1), conf)
        out['auc_llm_trust_residual_after_confidence'] = auc(lres)
    out.pop('krippendorff_alpha_trust_interval_check')
    os.makedirs(RES, exist_ok=True)
    merged.to_csv(os.path.join(RES, out_prefix + 'annotations_merged.csv'), index=False)
    with open(os.path.join(RES, out_prefix + 'annotation_analysis.json'), 'w') as f:
        json.dump(out, f, indent=2)
    write_macros(out)
    print(json.dumps(out, indent=2))


MACROS = {  # macro name -> (key path in analysis json, format)
    'AlphaTrust': (('krippendorff_alpha_trust_ordinal',), '{:.2f}'),
    'AlphaSense': (('krippendorff_alpha_sense_nominal',), '{:.2f}'),
    'AlphaTrustWithLLM': (('krippendorff_alpha_trust_humans_plus_llm',), '{:.2f}'),
    'LLMvsHumanQWK': (('llm_vs_human_median_qwk',), '{:.2f}'),
    'LLMvsHumanSenseKappa': (('llm_vs_human_majority_kappa_sense',), '{:.2f}'),
    'LLMSelfAlpha': (('llm_self_consistency_alpha_trust',), '{:.2f}'),
    'LLMvsHumanRho': (('llm_vs_human_mean_spearman',), '{:.2f}'),
    'AUCHuman': (('auc_trust_detects_correct', 'human_mean'), '{:.2f}'),
    'AUCLLM': (('auc_trust_detects_correct', 'llm'), '{:.2f}'),
    'TrustCorrect': (('mean_trust_by_correctness', 'correct'), '{:.2f}'),
    'TrustIncorrect': (('mean_trust_by_correctness', 'incorrect'), '{:.2f}'),
    'LLMTrustCorrect': (('mean_trust_by_correctness_llm', 'correct'), '{:.2f}'),
    'LLMTrustIncorrect': (('mean_trust_by_correctness_llm', 'incorrect'), '{:.2f}'),
    'SenseCorrect': (('sense_rate_by_correctness_human', 'correct'), '{:.0%}'),
    'SenseIncorrect': (('sense_rate_by_correctness_human', 'incorrect'), '{:.0%}'),
    'RhoHumanRtwo': (('spearman_human_trust_vs_lime_r2',), '{:.2f}'),
    'RhoHumanConf': (('spearman_human_trust_vs_confidence',), '{:.2f}'),
    'MeanTrustHuman': (('mean_trust_human',), '{:.2f}'),
    'MeanTrustLLM': (('mean_trust_llm',), '{:.2f}'),
    'LLMModel': (('llm_model',), '{}'),
    'AlphaNoAuthor': (('without_author', 'alpha_trust'), '{:.2f}'),
    'AUCNoAuthor': (('without_author', 'auc_trust'), '{:.2f}'),
    'AUCResidNoAuthor': (('without_author', 'auc_residual_after_confidence'), '{:.2f}'),
    'AUCAuthor': (('auc_trust_detects_correct', 'A'), '{:.2f}'),
    'AUCConf': (('auc_model_confidence_detects_correct',), '{:.2f}'),
    'AUCResid': (('auc_human_trust_residual_after_confidence',), '{:.2f}'),
    'PResid': (('p_human_trust_residual',), '{:.2f}'),
    'PTrustMW': (('mannwhitney_p_trust_correct_vs_incorrect',), '{:.3f}'),
    'AUCLo': (('auc_human_mean_ci95', 0), '{:.2f}'),
    'AUCHi': (('auc_human_mean_ci95', 1), '{:.2f}'),
    'RhoLLMConf': (('spearman_llm_trust_vs_confidence',), '{:.2f}'),
    'AUCLLMResid': (('auc_llm_trust_residual_after_confidence',), '{:.2f}'),
}


def write_macros(out=None):
    """report/generated/annotation_macros.tex. Without data every value is a
    red TBD so that no placeholder can silently end up in the submitted PDF."""
    lines = ['% generated by src/analyze_annotations.py -- do not edit']
    for name, (path, f) in MACROS.items():
        v = out
        try:
            for p in path:
                v = v[p]
            txt = f.format(v).replace('%', '\\%')
        except (TypeError, KeyError):
            txt = '\\textcolor{red}{TBD}'
        lines.append(f'\\newcommand{{\\{name}}}{{{txt}}}')
    pw = (out or {}).get('pairwise_qwk_trust', {})
    rng = f"{min(pw.values()):.2f}--{max(pw.values()):.2f}" if pw else '\\textcolor{red}{TBD}'
    lines.append(f'\\newcommand{{\\PairQWKRange}}{{{rng}}}')
    gen = os.path.join(HERE, '..', 'report', 'generated')
    os.makedirs(gen, exist_ok=True)
    open(os.path.join(gen, 'annotation_macros.tex'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--placeholders':
        write_macros(None)
    else:
        main()
