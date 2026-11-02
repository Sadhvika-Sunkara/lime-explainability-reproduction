"""
Train a classifier on the UCI Heart Disease (Cleveland) dataset and generate
LIME explanations for individual predictions -- this is the "New Data"
(existing dataset) step from the proposal (Section 5.1 / Table 1).

Note on reading the output: LIME's "+" and "-" weights below are always
relative to the "disease" class (label 1), regardless of which class was
actually predicted -- a "+" means that feature pushed the prediction toward
"disease", a "-" means it pushed toward "no disease", no matter which one
the model actually output for that instance.

HOW TO GET THE DATA:

  Option A (easiest) -- the ucimlrepo helper package:
      pip install ucimlrepo
      # this script will use it automatically.

  Option B -- manual download, if Option A fails or is blocked:
      Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/45/data/processed.cleveland.data" -OutFile "processed.cleveland.data" -UseBasicParsing
      # then run this script with --csv processed.cleveland.data

Then:
      pip install lime scikit-learn numpy pandas
      python lime_heart_disease.py                      # tries ucimlrepo first
      python lime_heart_disease.py --csv processed.cleveland.data   # manual fallback
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from lime.lime_tabular import LimeTabularExplainer

COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

# Categorical columns, by index into the feature list (all COLUMNS minus 'num'),
# with human-readable value labels where the UCI documentation defines a fixed
# coded vocabulary. LIME treats these as discrete categories instead of
# cutting them at arbitrary numeric thresholds.
CATEGORICAL_LABELS = {
    'sex': {0: 'female', 1: 'male'},
    'cp': {1: 'typical angina', 2: 'atypical angina',
           3: 'non-anginal pain', 4: 'asymptomatic'},
    'fbs': {0: 'fasting blood sugar <=120', 1: 'fasting blood sugar >120'},
    'restecg': {0: 'normal', 1: 'ST-T abnormality', 2: 'LV hypertrophy'},
    'exang': {0: 'no exercise angina', 1: 'exercise-induced angina'},
    'slope': {1: 'upsloping', 2: 'flat', 3: 'downsloping'},
    'thal': {3: 'normal', 6: 'fixed defect', 7: 'reversible defect'},
    # 'ca' (0-3 vessels) is ordinal/discrete but has no named categories in
    # the UCI docs, so it's left as a (categorical, unlabeled) feature.
}


def load_via_ucimlrepo():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    df = X.copy()
    df['num'] = y.iloc[:, 0].values
    return df


def load_via_csv(path):
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values='?')
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', default=None,
                     help="path to processed.cleveland.data (manual fallback). "
                          "If omitted, tries ucimlrepo first.")
    ap.add_argument('--n-explain', type=int, default=6,
                     help="how many test predictions to explain")
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    if args.csv:
        df = load_via_csv(args.csv)
        print(f"Loaded {len(df)} rows from {args.csv}")
    else:
        try:
            df = load_via_ucimlrepo()
            print(f"Loaded {len(df)} rows via ucimlrepo")
        except Exception as e:
            raise SystemExit(
                "Could not fetch via ucimlrepo "
                f"({e!r}). Re-run with --csv pointing at a manually "
                "downloaded processed.cleveland.data file -- see the "
                "instructions at the top of this script.")

    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with missing values "
          f"('?' in ca/thal) -> {len(df)} rows remain.")

    # Binarize target: 0 = no disease, 1-4 -> 1 = disease present
    # (standard simplification used throughout the UCI heart-disease
    # literature, since 1-4 encode increasing severity, not distinct classes)
    df['target'] = (df['num'] > 0).astype(int)
    feature_names = [c for c in COLUMNS if c != 'num']
    X = df[feature_names].astype(float).values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                  random_state=args.seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"Test F1 score: {f1_score(y_test, preds):.3f}")
    print(f"Train/test sizes: {len(X_train)} / {len(X_test)}")

    categorical_idx = [feature_names.index(c) for c in CATEGORICAL_LABELS
                        if c in feature_names] + [feature_names.index('ca')]
    categorical_names = {}
    for name, labels in CATEGORICAL_LABELS.items():
        i = feature_names.index(name)
        max_val = int(max(labels.keys()))
        categorical_names[i] = [labels.get(v, str(v)) for v in range(max_val + 1)]

    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names,
        class_names=['no disease', 'disease'],
        categorical_features=sorted(set(categorical_idx)),
        categorical_names=categorical_names,
        discretize_continuous=True, random_state=args.seed)

    n_explain = min(args.n_explain, len(X_test))
    print(f"\n=== LIME explanations for {n_explain} test predictions ===")
    print("('+' pushes toward 'disease', '-' pushes toward 'no disease' -- "
          "always relative to the disease class, regardless of which class "
          "was predicted)")
    for i in range(n_explain):
        prob = clf.predict_proba(X_test[i:i+1])[0]
        pred_label = int(np.argmax(prob))
        true_label = int(y_test[i])
        exp = explainer.explain_instance(
            X_test[i], clf.predict_proba, num_features=6)
        print(f"\n--- Instance {i} ---")
        print(f"True class: {'disease' if true_label else 'no disease'}  |  "
              f"Predicted: {'disease' if pred_label else 'no disease'}  "
              f"(p={prob[pred_label]:.3f})")
        print("Top features driving this prediction:")
        for feat, weight in exp.as_list():
            direction = "+" if weight > 0 else "-"
            print(f"  {direction} {feat}: {weight:+.3f}")


if __name__ == '__main__':
    main()
