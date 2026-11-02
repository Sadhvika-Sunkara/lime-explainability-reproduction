"""Dataset loaders. Every text loader returns (train_docs, y_train, test_docs, y_test)
using the original paper's protocol: 1000 positive + 1000 negative documents,
each class shuffled with random.seed(1) and split 80/20 -> 1600 train / 400 test
(load_datasets.py in marcotcr/lime-experiments)."""
import html
import os
import random
import re

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def _split(pos, neg, seed=1):
    random.seed(seed)
    random.shuffle(pos)
    random.shuffle(neg)
    sp, sn = int(len(pos) * .8), int(len(neg) * .8)
    tr = pos[:sp] + neg[:sn]
    te = pos[sp:] + neg[sn:]
    ytr = np.array([1] * sp + [0] * sn)
    yte = np.array([1] * (len(pos) - sp) + [0] * (len(neg) - sn))
    return tr, ytr, te, yte


def _clean(t):
    t = html.unescape(t)
    t = re.sub(r'<br\s*/?>', ' ', t)
    return ' '.join(t.split())


def load_multidomain(domain):
    """Blitzer et al. (2007) Multi-Domain Sentiment Dataset, 'books' or 'dvd'.
    Expects data/sorted_data_acl/<domain>/{positive,negative}.review (the
    domain_sentiment_data.tar.gz release; reviews are pseudo-XML)."""
    d = os.path.join(ROOT, 'sorted_data_acl', domain)

    def read(f):
        txt = open(os.path.join(d, f), encoding='latin-1').read()
        return [_clean(t) for t in re.findall(r'<review_text>(.*?)</review_text>', txt, re.S)]
    return _split(read('positive.review'), read('negative.review'))


def load_imdb(n_per_class=1000, seed=1):
    """IMDB Large Movie Review Dataset (Maas et al., 2011), 50k-review CSV.
    A balanced random sample of n_per_class reviews per label is drawn so the
    new dataset matches the size and balance of Books/DVDs."""
    df = pd.read_csv(os.path.join(ROOT, 'imdb', 'IMDB-Dataset.csv'))
    rng = np.random.RandomState(seed)
    pos = df[df.sentiment == 'positive'].review.values
    neg = df[df.sentiment == 'negative'].review.values
    pos = [_clean(t) for t in rng.choice(pos, n_per_class, replace=False)]
    neg = [_clean(t) for t in rng.choice(neg, n_per_class, replace=False)]
    return _split(pos, neg, seed)


def load_text(name):
    return {'books': lambda: load_multidomain('books'),
            'dvd': lambda: load_multidomain('dvd'),
            'imdb': load_imdb}[name]()


# ------------------------------------------------------------------ Heart
HEART_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
HEART_CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
HEART_LABELS = {
    'sex': {0: 'female', 1: 'male'},
    'cp': {1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'},
    'fbs': {0: 'fasting sugar <= 120', 1: 'fasting sugar > 120'},
    'restecg': {0: 'normal ECG', 1: 'ST-T abnormality', 2: 'LV hypertrophy'},
    'exang': {0: 'no exercise angina', 1: 'exercise angina'},
    'slope': {1: 'upsloping', 2: 'flat', 3: 'downsloping'},
    'ca': {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels'},
    'thal': {3: 'normal', 6: 'fixed defect', 7: 'reversible defect'},
}
HEART_DESCRIPTIONS = {
    'age': 'age (years)', 'sex': 'sex', 'cp': 'chest pain type',
    'trestbps': 'resting blood pressure (mm Hg)', 'chol': 'serum cholesterol (mg/dl)',
    'fbs': 'fasting blood sugar', 'restecg': 'resting ECG',
    'thalach': 'max heart rate achieved', 'exang': 'exercise-induced angina',
    'oldpeak': 'ST depression (exercise vs rest)', 'slope': 'slope of peak exercise ST',
    'ca': 'major vessels coloured by fluoroscopy', 'thal': 'thallium stress test',
}


def load_heart():
    """UCI Heart Disease, processed Cleveland subset (Janosi et al., 1988).
    303 rows; 6 with '?' in ca/thal are dropped (297 remain); target is
    binarised: num 0 -> 0 (no disease), num 1-4 -> 1 (disease)."""
    df = pd.read_csv(os.path.join(ROOT, 'raw', 'processed.cleveland.data'),
                     header=None, names=HEART_COLS, na_values='?').dropna()
    df = df.reset_index(drop=True)
    y = (df.num > 0).astype(int).values
    X = df[HEART_COLS[:-1]].astype(float)
    return X, y
