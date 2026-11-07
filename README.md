# Reproducing and Extending LIME (Ribeiro et al., KDD 2016)

COMP8240 Applications of Data Science, Macquarie University — final project, Sadhvika Sunkara (48768111)

This repository reproduces the two simulated-user experiments of
[“Why Should I Trust You?”: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
(Ribeiro, Singh & Guestrin, KDD 2016) and extends them to new data:

| | Original data (replication) | New data |
|---|---|---|
| **§5.2 Faithfulness** — recall of a model's true features | Books, DVDs (authors' `processed_acl` input; raw text as robustness check) | IMDB (new text domain), UCI Heart Disease (tabular) |
| **§5.3 Trust** — F1 of deciding whether to trust a prediction | Books, DVDs × LR / NN / RF / SVM | IMDB, Heart × LR / NN / RF / SVM |
| **Human trust judgements** | — | 60 Heart Disease predictions + LIME explanations rated by 3 people and an LLM judge |

The report (ACL format) is in [`report/`](report/) and the compiled PDF is `report/main.pdf`.

## Quick start

```bash
git clone https://github.com/Sadhvika-Sunkara/lime-explainability-reproduction
cd lime-explainability-reproduction
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/get_data.py                             # Books/DVDs + IMDB (~100 MB); Heart data is included
bash run_all.sh                                        # every experiment + tables (~2.5 h on 2 CPU cores)
```

To run a single setting (each takes 1–15 minutes):

```bash
python src/run_faithfulness.py --dataset books_acl           # books_acl | dvd_acl (authors' input) | books | dvd (raw text) | imdb
python src/run_trust.py --dataset dvd --algorithm svm        # logreg | neighbors | random_forest | svm
python src/run_heart.py --experiment faithfulness
python src/run_heart.py --experiment trust --algorithm random_forest
python src/make_tables.py                                    # results/*.json -> report/generated/*.tex
```

Each script writes one JSON file to `results/`. Pass `--max-test 20` (text) or `--num-samples 2000` for a quick smoke test.

## Repository layout

```
src/
  explainers.py           Python 3 port of the authors' LIME / Parzen / Greedy / Random explainers
                          (+ FastRBFSVC: exact closed-form SVM probabilities for LIME samples)
  data.py                 loaders; the paper's 1600/400 split protocol for every text dataset
  common.py               vectoriser, the paper's classifier settings, Parzen bandwidths
  run_faithfulness.py     §5.2 on text (port of evaluate_explanations.py)
  run_trust.py            §5.3 on text (port of data_trusting.py)
  heart_common.py         tabular versions of the four explainers (official `lime` package for LIME)
  run_heart.py            §5.2 and §5.3 on Heart Disease (5-fold cross-fitting)
  build_annotation_set.py builds the 60-item human-annotation set and Excel sheet
  llm_judge.py            LLM-as-judge on the same items (OpenAI API)
  analyze_annotations.py  Krippendorff's alpha, weighted kappa, AUC, LLM-vs-human agreement
  ablation_parzen_sigma.py  Parzen bandwidth ablation (explains the Parzen replication gap)
  make_tables.py          all LaTeX tables / figures in the report
scripts/get_data.py       downloads the public datasets
tests/test_fast_svm.py    checks FastRBFSVC against scikit-learn
results/                  one JSON per experiment (the numbers in the report)
annotation/               annotation sheet, items, answer key, completed sheets, LLM judgements
report/                   LaTeX source (ACL template), generated tables, PDF
```

## Why a port, not the original code

The authors' [lime-experiments](https://github.com/marcotcr/lime-experiments) code pins Python 2.7,
numpy 1.14, scipy 0.17 and scikit-learn 0.18, none of which install on a current system, and it
calls APIs that have since been removed (`sklearn.cross_validation`, `print` statements). `src/explainers.py`
re-implements the same algorithms with the same hyperparameters (N = 15,000 samples, kernel width
ρ = 25 on 100 × cosine distance, K-LASSO via the LARS path, the authors' Parzen bandwidths), and the
experiment scripts keep the original classifier settings, seeds, simulated-user rules and metric.
The deviations are listed in §4 of the report:
a bag-of-words RF
(as in the released code), and vectorised and closed-form speed-ups that do not change any computation.

## Data

| Dataset | Source | Licence | In repo? |
|---|---|---|---|
| Multi-Domain Sentiment (Books, DVDs): `processed_acl` (replication) and raw reviews (robustness) | Blitzer et al. 2007, [JHU](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) | research use | no — `scripts/get_data.py` |
| IMDB 50k reviews | Maas et al. 2011, [Stanford](https://ai.stanford.edu/~amaas/data/sentiment/) | research use | no — `scripts/get_data.py` |
| Heart Disease (Cleveland) | Janosi et al. 1988, [UCI](https://archive.ics.uci.edu/dataset/45/heart+disease) | CC BY 4.0 | yes, `data/raw/` |
| Explanation-trust annotations (new) | this project | CC BY 4.0 | yes, `annotation/` |

## Annotation protocol (new dataset)

`src/build_annotation_set.py` picks 20 incorrect and 40 correct cross-fitted Random Forest predictions
(seed 1) and writes `annotation/annotation_sheet.xlsx`, which has a participant-information/consent page and one row per
item: patient record, prediction + confidence, LIME top-5 explanation, and two questions,
trust (1–5) and whether the explanation makes sense (Yes/No). Raters never see the true label (`annotation/answer_key.csv`).
Completed sheets go in `annotation/responses/annotator_{A,B,C}.xlsx`; then:

```bash
export OPENAI_API_KEY=...            # Windows PowerShell: $env:OPENAI_API_KEY="..."
python src/llm_judge.py --model gpt-4o-mini --repeats 3
python src/analyze_annotations.py    # -> results/annotation_analysis.json + report macros
```

## Environment used for the reported results

Python 3.13.18 on Linux x86-64 (2 CPU cores, no GPU); package versions pinned in `requirements.txt`.
Earlier development ran on GitHub Codespaces (Python 3.12) and Windows 11 (Python 3.13).

## Citation

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why Should I Trust You?”: Explaining the Predictions
of Any Classifier. *KDD 2016*, 1135–1144. https://doi.org/10.1145/2939672.2939778
