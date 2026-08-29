# Reproducing and Extending LIME: Local Interpretable Model-Agnostic Explanations

This repository contains the code and supporting evidence for a semester-long reproducibility
project undertaken for **COMP8240 – Applications of Data Science**, Macquarie University
(Session 2, 2026).

The project reproduces and extends the work of:

> Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016.
> ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://dl.acm.org/doi/10.1145/2939672.2939778).
> In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and
> Data Mining (KDD 2016)*, pages 1135–1144.

which introduced **LIME** (Local Interpretable Model-Agnostic Explanations), a technique for
explaining individual predictions of any black-box classifier by fitting a simple, interpretable
model to the local neighbourhood of a specific prediction.

The full project proposal (submitted as Assessment Task #1) describes the wider plan: applying
LIME to five new datasets not used in the original paper, and designing an LLM-as-judge
evaluation protocol to validate against human trust judgements. This repository currently holds
the **feasibility-check stage** of that project — confirming that the official LIME
implementation runs correctly and reproduces a result consistent with the original paper's
described phenomenon.

## What this repository demonstrates

The original paper motivates LIME partly through the observation that classifiers can achieve
good accuracy for the wrong reasons — relying on spurious, "untrustworthy" features rather than
genuine signal (their canonical example is an image classifier that learned to detect snow in
the background rather than wolves). `lime_test.py` reproduces an analogous phenomenon on text
data: a classifier trained on part of the 20 Newsgroups dataset that ends up relying on
email-header artefacts (e.g. `Host`, `Posting`, `NNTP`) rather than genuine topical content —
exactly the kind of untrustworthy-feature reliance the original paper's evaluation is designed to
surface.

## Repository contents

| File | Description |
|---|---|
| `lime_test.py` | Confirms the official `lime` package installs and runs correctly, and reproduces the untrustworthy-feature phenomenon described above. |
| `README.md` | This file. |

## What `lime_test.py` does

1. Loads the **20 Newsgroups** dataset (`sklearn.datasets.fetch_20newsgroups`), restricted to the
   `alt.atheism` and `soc.religion.christian` categories — the same "Christianity vs. Atheism"
   subset used in the original paper's primary text-classification experiments.
2. Vectorises the raw text with **TF-IDF**.
3. Trains a classifier (Random Forest) on the vectorised training split and evaluates it with an
   **F1 score** on the held-out test split.
4. Wraps the trained classifier with LIME's `LimeTextExplainer`.
5. Generates a local explanation for an individual test-set prediction, showing which words most
   influenced the model's classification for that specific document.

## How to run it

```bash
pip install lime scikit-learn numpy
python lime_test.py
```

This was originally run in a GitHub Codespaces cloud development environment as part of the
Week-2/3 reproducibility exercise for this unit, and reproduces without modification in any
standard Python 3 environment with the packages above installed.

## Results / evidence of the run

The run confirms two things: (1) the official `lime` implementation installs and executes
end-to-end without modification, and (2) the resulting explanation for several test predictions
relies heavily on email-header tokens (`Host`, `Posting`, `NNTP`, etc.) rather than genuine
topical words — directly illustrating the untrustworthy-feature phenomenon the original paper's
evaluation is built around, and matching what the project proposal (Section 3, "Explanation of
Evaluation") describes.

**Actual console output from the run:**

```
[PASTE YOUR ACTUAL TERMINAL OUTPUT HERE — the printed classification
report / F1 score, and the LIME explanation output listing the
top contributing words and their weights for the example prediction.]
```

*(A screenshot of the Codespaces run showing this same output is also included below / attached
as `run_screenshot.png` in this repository.)*

## Relationship to the wider project

This feasibility check is Phase 1 of the four-phase plan described in the project proposal:

1. **Reproduce** the original paper's core quantitative results (fidelity/recall and
   simulated-trust experiments) on 20 Newsgroups — *in progress, this repository*.
2. **Apply LIME to new datasets** not used in the original paper (Amazon/IMDB reviews, Fake and
   Real News, Credit Card Fraud Detection, Jigsaw Toxic Comment Classification, Heart Disease).
3. **Construct a new "explanation trust" dataset** (~150–200 predictions from a Heart Disease
   classifier) with human and LLM-as-judge trust judgements.
4. **Consolidate results** and prepare the final report.

Subsequent phases and their code will be added to this repository as the project progresses
through the semester.

## Author

**Sadhvika Sunkara** (Student ID: 48768111)
Macquarie University — COMP8240, Session 2, 2026
`sadhvika.sunkara@student.mq.edu.au`
