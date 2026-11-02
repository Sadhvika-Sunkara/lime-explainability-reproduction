# Earlier attempts (kept for transparency)

These are the scripts from the proposal and project-update stages. They are superseded by `src/`
and are not used for any number in the final report.

| File | Stage | What it did | Why it was superseded |
|---|---|---|---|
| `lime_test_20newsgroups.py` | proposal (Aug 2026) | installed `lime`, explained a Random Forest on 20 Newsgroups (atheism vs. christian), F1 0.925; top features were e-mail header tokens (`Host`, `Posting`, `NNTP`) | feasibility check only |
| `reproduce_trust_experiment.py` + `books_trust_results.txt` | project update (Oct 2026) | §5.3 trust experiment, Books, LR only, 80 test reviews, 1,000 LIME samples, 10 rounds; LIME trust-F1 97.5 ± 0.9 | covered 1 of the paper's 8 settings, no baselines; trust rule compared the local model with/without untrustworthy words rather than against the classifier's own probability; stop-word-filtered vocabulary |
| `lime_heart_disease.py` + `heart_results.txt` | project update | Random Forest on Heart Disease (accuracy 0.867, F1 0.852, 237/60 split) and example LIME explanations | no evaluation of the explanations; became the model used for the annotation set (`src/build_annotation_set.py`) |
