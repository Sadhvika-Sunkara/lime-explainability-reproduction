"""
Builds the NEW constructed dataset: an "explanation trust" annotation set of
60 Heart Disease predictions, each with its LIME explanation, to be rated by
human annotators and by an LLM judge.

Model: the Random Forest from the project's earlier Heart Disease script
(300 trees, max depth 6). Every patient is predicted by a model trained on the
other four folds (5-fold cross-fitting), so no rated prediction was seen in
training. 20 incorrect and 40 correct predictions are sampled (seed 1) and
shuffled; incorrect predictions are over-sampled (base rate ~17%) so we can
test whether raters can use LIME to spot predictions that should not be
trusted -- the question at the heart of the original paper.

Outputs (annotation/):
  annotation_sheet.xlsx     -- for annotators (no answer key)
  items.csv                 -- same items in machine-readable form (for the LLM judge)
  answer_key.csv            -- true label, correctness, LIME local fidelity (R^2); NOT for annotators
"""
import os
import warnings

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.model_selection import StratifiedKFold

from data import HEART_DESCRIPTIONS, HEART_LABELS, load_heart
from heart_common import FEATURES, get_model, lime_explainer

warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'annotation')
K, N_SAMPLES, N_WRONG, N_RIGHT = 5, 15000, 20, 40
UNITS = {}  # units are already in HEART_DESCRIPTIONS


def describe_value(f, v):
    if f in HEART_LABELS:
        return HEART_LABELS[f].get(int(v), str(int(v)))
    return f'{v:g}{UNITS.get(f, "")}'


def patient_text(x):
    return '\n'.join(f'{HEART_DESCRIPTIONS[f]}: {describe_value(f, v)}' for f, v in zip(FEATURES, x))


def readable_condition(cond):
    """'thal=7' / 'oldpeak > 1.60' / '0.00 < oldpeak <= 0.80' -> readable text."""
    out = cond
    for f in sorted(FEATURES, key=len, reverse=True):
        if f in cond:
            out = cond.replace(f, HEART_DESCRIPTIONS[f].split(' (')[0])
            break
    return out


def explanation_text(pairs):
    lines = []
    for cond, w in pairs:
        side = 'DISEASE' if w > 0 else 'NO DISEASE'
        lines.append(f'{readable_condition(cond)}  ->  towards {side}  (weight {abs(w):.2f})')
    return '\n'.join(lines)


def main():
    X, y = load_heart()
    X = X.values
    rows = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
        m = get_model('rf_annot').fit(X[tr], y[tr])
        lime = lime_explainer(X[tr])
        for i in te:
            p = m.predict_proba(X[i:i + 1])[0]
            rows.append(dict(idx=int(i), true=int(y[i]), pred=int(p.argmax()), p_disease=float(p[1]),
                             model=m, lime=lime))
    rows.sort(key=lambda r: r['idx'])
    wrong = [r for r in rows if r['pred'] != r['true']]
    right = [r for r in rows if r['pred'] == r['true']]
    rng = np.random.RandomState(1)
    pick = [wrong[i] for i in rng.choice(len(wrong), N_WRONG, replace=False)] + \
           [right[i] for i in rng.choice(len(right), N_RIGHT, replace=False)]
    rng.shuffle(pick)
    print(f'cross-fitted accuracy over 297 patients: {np.mean([r["pred"] == r["true"] for r in rows]):.3f}; '
          f'{len(wrong)} errors')

    items, key = [], []
    for n, r in enumerate(pick, 1):
        x = X[r['idx']]
        e = r['lime'].explain_instance(x, r['model'].predict_proba, num_features=K,
                                       num_samples=N_SAMPLES, labels=(1,))
        pairs = e.as_list(label=1)
        conf = r['p_disease'] if r['pred'] else 1 - r['p_disease']
        item_id = f'H{n:02d}'
        items.append(dict(item_id=item_id, patient=patient_text(x),
                          prediction='DISEASE' if r['pred'] else 'NO DISEASE',
                          confidence=round(conf, 2), explanation=explanation_text(pairs),
                          explanation_raw='; '.join(f'{c}: {w:+.3f}' for c, w in pairs)))
        key.append(dict(item_id=item_id, dataset_row=r['idx'], true_label=r['true'], predicted=r['pred'],
                        correct=int(r['pred'] == r['true']), p_disease=round(r['p_disease'], 4),
                        lime_local_r2=round(float(e.score), 4), lime_intercept=round(float(e.intercept[1]), 4)))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(items).to_csv(os.path.join(OUT, 'items.csv'), index=False)
    pd.DataFrame(key).to_csv(os.path.join(OUT, 'answer_key.csv'), index=False)
    write_workbook(items)
    print(f'wrote {len(items)} items to {OUT}')


INSTRUCTIONS = [
    ('Explanation trust study - annotator sheet', True),
    ('COMP8240 Applications of Data Science, Macquarie University - student project (Sadhvika Sunkara)', False),
    ('', False),
    ('Participant information', True),
    ('You are asked to rate 60 predictions made by a machine-learning model that estimates whether a patient '
     'has heart disease, each shown with an automatic explanation (LIME) of why the model made it. '
     'It takes about 25 minutes. Participation is voluntary; you may stop at any time and ask for your '
     'ratings to be withdrawn until the report is submitted.', False),
    ('The patient records are from the public, de-identified UCI Heart Disease (Cleveland, 1988) dataset. '
     'No information about you is recorded except your ratings, which are stored under an annotator code '
     '(A, B or C) and reported only in aggregate. No medical knowledge is required and this is not a test of you.', False),
    ('Typing your annotator code in cell B14 means you have read this and agree to take part.', False),
    ('', False),
    ('What to do', True),
    ('1. Work through the "Items" sheet top to bottom. Do not discuss items with other annotators.', False),
    ('2. Read the patient record, the model\'s prediction and confidence, and the explanation. The explanation '
     'lists up to 5 facts about the patient and whether each pushed the model towards DISEASE or NO DISEASE; '
     'a larger weight means a stronger push.', False),
    ('3. Fill in the yellow cells only: Q1 trust (1-5) and Q2 (Yes/No). The comment is optional.', False),
    ('', False),
    ('Annotator code (A, B or C):', True),
    ('', False),
    ('Rating scale for Q1 "How much do you trust this prediction?"', True),
    ('1 = strongly distrust   2 = distrust   3 = unsure   4 = trust   5 = strongly trust', False),
    ('Q2 "Does the explanation make sense given the patient record?"  Yes / No', False),
    ('', False),
    ('Do not change any grey cells. Row 2 of the Items sheet is a filled-in EXAMPLE and is not scored.', False),
]


def write_workbook(items):
    wb = Workbook()
    arial = 'Arial'
    yellow = PatternFill('solid', start_color='FFFF00')
    grey = PatternFill('solid', start_color='EEEEEE')
    head = PatternFill('solid', start_color='1F3864')
    thin = Side(style='thin', color='BBBBBB')
    box = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws = wb.active
    ws.title = 'Instructions'
    ws.column_dimensions['A'].width = 110
    ws.column_dimensions['B'].width = 14
    for r, (text, bold) in enumerate(INSTRUCTIONS, 1):
        c = ws.cell(r, 1, text)
        c.font = Font(name=arial, bold=bold, size=12 if r == 1 else 10)
        c.alignment = Alignment(wrap_text=True, vertical='top')
        if len(text) > 110:
            ws.row_dimensions[r].height = 14 * (len(text) // 105 + 1)
    ws['B14'].fill = yellow
    ws['B14'].border = box
    ws['B14'].font = Font(name=arial, bold=True)
    dv_code = DataValidation(type='list', formula1='"A,B,C"', allow_blank=True)
    ws.add_data_validation(dv_code); dv_code.add('B14')

    it = wb.create_sheet('Items')
    cols = ['Item', 'Patient record', 'Model prediction', 'Model confidence',
            'Explanation (LIME, top 5 facts)', 'Q1 Trust (1-5)', 'Q2 Explanation makes sense? (Yes/No)', 'Comment (optional)']
    widths = [7, 42, 14, 12, 62, 11, 16, 30]
    for j, (h, w) in enumerate(zip(cols, widths), 1):
        c = it.cell(1, j, h)
        c.font = Font(name=arial, bold=True, color='FFFFFF'); c.fill = head
        c.alignment = Alignment(wrap_text=True, vertical='center'); c.border = box
        it.column_dimensions[c.column_letter].width = w
    example = dict(item_id='EXAMPLE', patient='age (years): 54\nsex: male\nchest pain type: asymptomatic\n...',
                   prediction='DISEASE', confidence=0.81,
                   explanation='thallium stress test=reversible defect  ->  towards DISEASE  (weight 0.21)\n...')
    for r, item in enumerate([example] + items, 2):
        vals = [item['item_id'], item['patient'], item['prediction'], item['confidence'], item['explanation']]
        for j, v in enumerate(vals, 1):
            c = it.cell(r, j, v)
            c.font = Font(name=arial, size=9, italic=(r == 2))
            c.alignment = Alignment(wrap_text=True, vertical='top')
            c.fill = grey; c.border = box
        it.cell(r, 4).number_format = '0%'
        for j in (6, 7, 8):
            c = it.cell(r, j)
            c.fill = yellow if r > 2 else grey
            c.border = box
            c.font = Font(name=arial, size=10, italic=(r == 2))
            c.alignment = Alignment(horizontal='center', vertical='top', wrap_text=True)
        it.row_dimensions[r].height = 160
    it['F2'], it['G2'], it['H2'] = 4, 'Yes', 'example only - not scored'
    it['F1'].comment = Comment('1 = strongly distrust ... 5 = strongly trust', 'study')
    last = len(items) + 2
    dv1 = DataValidation(type='whole', operator='between', formula1='1', formula2='5', allow_blank=True,
                         showErrorMessage=True, errorTitle='Trust rating', error='Enter a whole number from 1 to 5.')
    dv2 = DataValidation(type='list', formula1='"Yes,No"', allow_blank=True)
    it.add_data_validation(dv1); it.add_data_validation(dv2)
    dv1.add(f'F3:F{last}'); dv2.add(f'G3:G{last}')
    it.freeze_panes = 'B2'
    # progress counter for the annotator
    ws['A22'] = 'Items completed (Q1 filled):'
    ws['A22'].font = Font(name=arial, bold=True)
    ws['B22'] = f"=COUNT(Items!F3:F{last})&\" / {len(items)}\""
    ws['B22'].font = Font(name=arial)
    wb.save(os.path.join(OUT, 'annotation_sheet.xlsx'))


if __name__ == '__main__':
    main()
