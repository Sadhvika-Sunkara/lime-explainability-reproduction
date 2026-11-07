#!/usr/bin/env bash
# Quick check that every pipeline runs end to end (about 5-10 minutes on a laptop CPU).
# Uses small settings, so the numbers are NOT the report's numbers; it writes to
# results_quick/ and leaves results/ (the report's numbers) untouched.
# Needs: pip install -r requirements.txt && python scripts/get_data.py
# No OpenAI key needed.
set -e
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
export LIME_RESULTS_DIR=results_quick
python src/run_faithfulness.py --dataset books_acl --max-test 20 --num-samples 1000
python src/run_trust.py --dataset books_acl --algorithm logreg --max-test 20 --num-samples 1000 --num-rounds 10
python src/run_faithfulness.py --dataset imdb --max-test 20 --num-samples 1000
python src/run_heart.py --experiment trust --algorithm logreg --num-samples 1000 --num-rounds 10
python tests/test_fast_svm.py
echo "Quick check passed. Outputs are in results_quick/ (small settings, not the report's numbers)."
