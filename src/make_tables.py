"""Turns results/*.json into the LaTeX tables and figures used in the report
(report/generated/). Every number in the report's results tables comes from here."""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
GEN = os.path.join(HERE, '..', 'report', 'generated')
EXPL = ['random', 'parzen', 'greedy', 'LIME']
NAMES = {'random': 'Random', 'parzen': 'Parzen', 'greedy': 'Greedy', 'LIME': 'LIME'}

# Ribeiro et al. (2016). Table 1 (F1 of trustworthiness) is printed in the paper.
PAPER_TRUST = {
    'books': {'logreg': [14.6, 84.0, 53.7, 96.6], 'neighbors': [14.8, 87.6, 47.4, 94.5],
              'random_forest': [14.7, 94.3, 45.0, 96.2], 'svm': [14.7, 92.3, 53.3, 96.7]},
    'dvd': {'logreg': [14.2, 87.0, 52.4, 96.6], 'neighbors': [14.3, 81.7, 58.1, 91.8],
            'random_forest': [14.5, 94.2, 46.6, 96.1], 'svm': [14.4, 87.3, 55.1, 95.6]}}
# Figures 6-7 (recall of gold features), values printed above the bars.
PAPER_RECALL = {
    'books': {'l1logreg': [17.4, 72.8, 64.3, 92.1], 'tree': [20.6, 78.9, 37.0, 97.0]},
    'dvd': {'l1logreg': [19.2, 60.8, 63.4, 90.2], 'tree': [17.4, 80.6, 47.6, 97.8]}}
ALGS = ['logreg', 'neighbors', 'random_forest', 'svm']
ALG_SHORT = {'logreg': 'LR', 'neighbors': 'NN', 'random_forest': 'RF', 'svm': 'SVM',
             'l1logreg': 'Sparse LR', 'tree': 'DT'}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v):
    return '--' if v is None else f'{100 * v:.1f}'


def bold_max(cells):
    vals = [float(c) if c not in ('--',) else -1 for c in cells]
    m = max(vals)
    return [f'\\textbf{{{c}}}' if float(c if c != '--' else -1) == m else c for c in cells]


REPL = {'books': 'books_acl', 'dvd': 'dvd_acl'}   # replication = authors' processed_acl input


def recall_table():
    cols = [('books', 'l1logreg'), ('books', 'tree'), ('dvd', 'l1logreg'), ('dvd', 'tree')]
    F = {d: load(f'faithfulness_{REPL.get(d, d)}.json') for d in ['books', 'dvd', 'imdb', 'heart']}
    lines = [r'\begin{tabular}{l' + 'cc' * 4 + '}', r'\toprule',
             r' & \multicolumn{4}{c}{Books} & \multicolumn{4}{c}{DVDs} \\',
             r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
             r' & \multicolumn{2}{c}{Sparse LR} & \multicolumn{2}{c}{DT} & \multicolumn{2}{c}{Sparse LR} & \multicolumn{2}{c}{DT} \\',
             r' & Ours & Paper & Ours & Paper & Ours & Paper & Ours & Paper \\', r'\midrule']
    for k, e in enumerate(EXPL):
        row = [NAMES[e]]
        for d, a in cols:
            ours = F[d][a]['recall'][e] if F[d] else None
            row += [fmt(ours), f'{PAPER_RECALL[d][a][k]:.1f}']
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\midrule', r'Test acc.']
    acc = []
    for d, a in cols:
        acc += [fmt(F[d][a]['test_accuracy']) if F[d] else '--', '--']
    lines[-1] += ' & ' + ' & '.join(acc) + r' \\'
    lines += [r'\bottomrule', r'\end{tabular}']
    write('table_recall_replication.tex', lines)

    # new data
    lines = [r'\begin{tabular}{lcccc}', r'\toprule',
             r' & \multicolumn{2}{c}{IMDB (text, $K{=}10$)} & \multicolumn{2}{c}{Heart (tabular, $K{=}5$)} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r' & Sparse LR & DT & Sparse LR & DT \\', r'\midrule']
    for e in EXPL:
        cells = []
        for d in ['imdb', 'heart']:
            for a in ['l1logreg', 'tree']:
                cells.append(fmt(F[d][a]['recall'][e]) if F[d] else '--')
        lines.append(NAMES[e] + ' & ' + ' & '.join(cells) + r' \\')
    g = []
    for d in ['imdb', 'heart']:
        for a in ['l1logreg', 'tree']:
            g.append(f"{F[d][a]['mean_gold_size']:.1f}" if F[d] else '--')
    lines += [r'\midrule', 'Mean $|$gold$|$ & ' + ' & '.join(g) + r' \\', r'\bottomrule', r'\end{tabular}']
    write('table_recall_new.tex', lines)


def trust_table(datasets, fname, with_paper):
    T = {(d, a): load(f'trust_{REPL.get(d, d)}_{a}.json') for d in datasets for a in ALGS}
    head = ' & '.join(f'\\multicolumn{{4}}{{c}}{{{ {"books": "Books", "dvd": "DVDs", "imdb": "IMDB", "heart": "Heart"}[d] }}}' for d in datasets)
    lines = [r'\begin{tabular}{l' + 'cccc' * len(datasets) + '}', r'\toprule', ' & ' + head + r' \\',
             ''.join(f'\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}' for i in range(len(datasets))),
             ' & ' + ' & '.join(ALG_SHORT[a] for _ in datasets for a in ALGS) + r' \\', r'\midrule']
    for k, e in enumerate(EXPL):
        cells = []
        for d in datasets:
            for a in ALGS:
                r = T[(d, a)]
                c = fmt(r['results'][e]['f1']) if r else '--'
                if with_paper:
                    c = f'{c} {{\\scriptsize({PAPER_TRUST[d][a][k]:.1f})}}'
                cells.append(c)
        lines.append(NAMES[e] + ' & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
    for lab, key in [('Test acc.', 'test_accuracy'), ('Flipped', 'flipped_mean')]:
        cells = []
        for d in datasets:
            for a in ALGS:
                r = T[(d, a)]
                if not r:
                    cells.append('--')
                elif key == 'flipped_mean':
                    cells.append(f"{100 * r[key] / r['n_test']:.1f}\\%")
                else:
                    cells.append(fmt(r[key]))
        lines.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write(fname, lines)


def prf_table():
    """LIME precision / recall / F1 on the trustworthy class for every setting."""
    rows = []
    for d in ['books', 'dvd', 'imdb', 'heart']:
        for a in ALGS:
            r = load(f'trust_{REPL.get(d, d)}_{a}.json')
            if r:
                L = r['results']['LIME']
                rows.append((d, a, L['precision'], L['recall'], L['f1'], L['f1_std']))
    return rows


def prf_latex():
    rows = prf_table()
    D = {'books': 'Books', 'dvd': 'DVDs', 'imdb': 'IMDB', 'heart': 'Heart'}
    lines = [r'\begin{tabular}{llccc}', r'\toprule', r'Data & Clf. & P & R & F1 ($\pm$sd) \\', r'\midrule']
    prev = None
    for d, a, p, r, f, sd in rows:
        if prev and d != prev:
            lines.append(r'\midrule')
        lines.append(f"{D[d] if d != prev else ''} & {ALG_SHORT[a]} & {100*p:.1f} & {100*r:.1f} & {100*f:.1f} $\\pm$ {100*sd:.1f} \\\\")
        prev = d
    lines += [r'\bottomrule', r'\end{tabular}']
    write('table_lime_prf.tex', lines)


def figure_gap():
    """LIME's margin over the best baseline, per setting (trust F1)."""
    labels, ours, paper = [], [], []
    for d in ['books', 'dvd', 'imdb', 'heart']:
        for a in ALGS:
            r = load(f'trust_{REPL.get(d, d)}_{a}.json')
            if not r:
                continue
            labels.append(f"{ {'books': 'Books', 'dvd': 'DVDs', 'imdb': 'IMDB', 'heart': 'Heart'}[d] }\n{ALG_SHORT[a]}")
            R = r['results']
            ours.append(100 * R['LIME']['f1'] - 100 * max(R[e]['f1'] for e in EXPL if e != 'LIME'))
            paper.append(PAPER_TRUST[d][a][3] - max(PAPER_TRUST[d][a][:3]) if d in PAPER_TRUST else np.nan)
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    ax.bar(x - .2, paper, .4, label='Paper (Table 1)', color='#9aa5b1')
    ax.bar(x + .2, ours, .4, label='Ours', color='#1f4e79')
    ax.axhline(0, color='black', lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('LIME F1 $-$ best\nbaseline F1 (pts)', fontsize=8)
    ax.tick_params(axis='y', labelsize=7)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    n_rep = sum(1 for p in paper if not np.isnan(p))
    ax.axvspan(n_rep - .5, len(labels) - .5, color='#f2f2f2', zorder=-1)
    ax.text(n_rep + (len(labels) - n_rep) / 2 - .5, ax.get_ylim()[1] * .92, 'new data', ha='center', fontsize=7, style='italic')
    fig.tight_layout()
    fig.savefig(os.path.join(GEN, 'fig_trust_margin.pdf'))
    fig.savefig(os.path.join(GEN, 'fig_trust_margin.png'), dpi=200)


def write(name, lines):
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, name), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def results_macros():
    """Summary numbers quoted in the report's prose."""
    M = {}
    F = {d: load(f'faithfulness_{REPL.get(d, d)}.json') for d in ['books', 'dvd', 'imdb', 'heart']}
    rec = [100 * F[d][a]['recall']['LIME'] for d in ['books', 'dvd'] for a in ['l1logreg', 'tree']]
    prec = [PAPER_RECALL[d][a][3] for d in ['books', 'dvd'] for a in ['l1logreg', 'tree']]
    M['ReplRecallRange'] = f'{min(rec):.1f}--{max(rec):.1f}'
    M['ReplRecallGap'] = f'{np.mean(np.array(prec) - np.array(rec)):.1f}'
    trust, ptrust, best_all, gaps = [], [], True, []
    for d in ['books', 'dvd']:
        for a in ALGS:
            r = load(f'trust_{REPL[d]}_{a}.json')['results']
            trust.append(100 * r['LIME']['f1']); ptrust.append(PAPER_TRUST[d][a][3])
            for k, e in enumerate(EXPL):
                gaps.append(abs(100 * r[e]['f1'] - PAPER_TRUST[d][a][k]))
    M['ReplTrustRange'] = f'{min(trust):.1f}--{max(trust):.1f}'
    M['ReplTrustMAE'] = f'{np.mean(np.abs(np.array(trust) - np.array(ptrust))):.1f}'
    M['ReplTrustAllMAE'] = f'{np.mean(gaps):.1f}'
    M['ReplTrustWithinFive'] = str(sum(abs(t - p) <= 5 for t, p in zip(trust, ptrust)))
    for d, name in [('imdb', 'IMDB'), ('heart', 'Heart')]:
        for a, an in [('l1logreg', 'LR'), ('tree', 'Tree')]:
            M[f'{name}{an}Recall'] = f"{100 * F[d][a]['recall']['LIME']:.1f}"
        vals = {e: [] for e in EXPL}
        for a in ALGS:
            r = load(f'trust_{d}_{a}.json')
            if r:
                for e in EXPL:
                    vals[e].append(100 * r['results'][e]['f1'])
        for e in ['LIME', 'parzen']:
            if vals[e]:
                M[f'{name}{e.capitalize()}TrustRange'] = f'{min(vals[e]):.1f}--{max(vals[e]):.1f}'
    # input sensitivity: mean |ours - paper| on raw text vs. the authors' processed_acl input
    for tag, src in [('Raw', {'books': 'books', 'dvd': 'dvd'}), ('Acl', REPL)]:
        for e, en in [('LIME', 'Lime'), ('parzen', 'Parzen'), ('greedy', 'Greedy'), ('random', 'Random')]:
            k = EXPL.index(e)
            tr = [abs(100 * load(f'trust_{src[d]}_{a}.json')['results'][e]['f1'] - PAPER_TRUST[d][a][k])
                  for d in ['books', 'dvd'] for a in ALGS]
            fa = [abs(100 * load(f'faithfulness_{src[d]}.json')[a]['recall'][e] - PAPER_RECALL[d][a][k])
                  for d in ['books', 'dvd'] for a in ['l1logreg', 'tree']]
            M[f'MAE{tag}Trust{en}'] = f'{np.mean(tr):.1f}'
            M[f'MAE{tag}Faith{en}'] = f'{np.mean(fa):.1f}'
    for d in ['books', 'dvd']:
        r = load(f'faithfulness_{REPL[d]}_depth10.json')
        if r:
            M[f'{d.capitalize()}ShallowTreeLime'] = f"{100 * r['tree']['recall']['LIME']:.1f}"
            M[f'{d.capitalize()}ShallowTreeDepth'] = str(r['tree'].get('depth'))
            M[f'{d.capitalize()}ShallowTreeAcc'] = f"{100 * r['tree']['test_accuracy']:.1f}"
        M[f'{d.capitalize()}DeepTreeLime'] = f"{100 * F[d]['tree']['recall']['LIME']:.1f}"
    lines = ['% generated by src/make_tables.py -- do not edit']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in M.items()]
    write('results_macros.tex', lines)
    return M


def main():
    os.makedirs(GEN, exist_ok=True)
    recall_table()
    trust_table(['books', 'dvd'], 'table_trust_replication.tex', True)
    trust_table(['imdb', 'heart'], 'table_trust_new.tex', False)
    figure_gap()
    prf_latex()
    print(results_macros())
    for r in prf_table():
        print('%-6s %-14s P=%.3f R=%.3f F1=%.3f (sd %.3f)' % r)
    print('wrote', sorted(os.listdir(GEN)))


if __name__ == '__main__':
    main()
