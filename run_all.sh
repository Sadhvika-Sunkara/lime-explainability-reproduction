#!/usr/bin/env bash
# Re-runs every experiment in the report and regenerates the tables.
# ~2.5 hours on 2 CPU cores. Requires: python scripts/get_data.py
set -e
export PYTHONWARNINGS=ignore   # silences a scikit-learn/joblib warning repeated by parallel RF workers
cd "$(dirname "$0")"
for d in books_acl dvd_acl books dvd imdb; do
  python src/run_faithfulness.py --dataset $d
  for a in logreg neighbors random_forest svm; do
    python src/run_trust.py --dataset $d --algorithm $a
  done
done
python src/ablation_parzen_sigma.py
for d in books_acl dvd_acl; do python src/run_faithfulness.py --dataset $d --tree-max-depth 10; done
python src/run_heart.py --experiment faithfulness
for a in logreg neighbors random_forest svm; do
  python src/run_heart.py --experiment trust --algorithm $a
done
python src/build_annotation_set.py
if ls annotation/responses/annotator_*.xlsx >/dev/null 2>&1; then python src/analyze_annotations.py; fi
python src/make_tables.py
