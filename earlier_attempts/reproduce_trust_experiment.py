"""
Modern reproduction of the LIME paper's Section 5.3 "simulated trust" experiment
(Ribeiro, Singh & Guestrin, 2016), adapted to run with current package versions
(lime + scikit-learn, 2026) instead of the frozen 2016 research code the authors
published at https://github.com/marcotcr/lime-experiments (which pins
numpy==1.14.0 / scipy==0.17 / scikit-learn==0.18 and Python 2 syntax and is not
practical to install as-is).

What this script actually does (read this before claiming it "matches the
paper's protocol" -- it follows the same idea, with named adaptations):
  1. Train a classifier (logistic regression, as in the paper) on a domain's
     product reviews (Books or DVDs, from the Blitzer et al. multi-domain
     sentiment dataset -- the raw/"unprocessed" release, where each review is
     wrapped in <review>...<review_text>...</review_text>...</review> tags).
  2. For several rounds, randomly designate a percentage of the vocabulary as
     "untrustworthy" features (simulating spurious/leaked signal the model
     should not be relying on).
  3. Ground truth: a prediction is one we "shouldn't trust" if zeroing out the
     untrustworthy features in that instance flips the CLASSIFIER's prediction.
  4. LIME's judgement, computed the way the paper's own data_trusting.py does
     it: reconstruct LIME's local linear approximation (intercept + feature
     weights) for the instance, then check whether removing the untrustworthy
     features' contribution from that local linear sum flips which side of
     0.5 it falls on. This is a linear-approximation test, not just "does an
     untrustworthy word appear in the top-K list" -- that cruder rule is also
     computed below and reported as a baseline, because it was this script's
     first (weaker) implementation and is informative as a contrast: it has
     near-certain recall but very low precision because, with a large
     percentage of vocabulary marked untrustworthy, almost every explanation
     contains *some* untrustworthy word somewhere in its top-K list even when
     that word isn't actually driving the prediction.
  5. Compare both rules' judgements against ground truth with precision/
     recall/F1. The paper reports F1 on the *trustworthy* predictions; this
     script reports both directions (mistrust-detection and trust-detection)
     so the numbers are unambiguous about which class is being scored.

NOTE on scale: the original paper used 15,000 LIME perturbation samples per
instance, many random-untrustworthy rounds, and the full test set. State your
actual --n-test/--num-samples/--num-rounds values explicitly in your write-up
or video -- this is a deliberate, disclosed substitution, not an attempt to
match the paper's exact numbers (see Section 7 of the proposal).
"""
import argparse
import random
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from lime.lime_text import LimeTextExplainer


def load_multidomain(path_data, seed=1):
    """Parses the Blitzer et al. multi-domain sentiment "unprocessed" XML-ish
    format -- each review wrapped in <review>...<review_text>TEXT</review_text>
    ...</review> -- into raw text documents, and does an 80/20 train/test
    split (same convention as the original paper's load_datasets.py, which
    targeted the older bag-of-words release of this same dataset).
    """
    import os

    def get_reviews(path):
        content = open(path, encoding='utf-8', errors='ignore').read()
        texts = re.findall(r'<review_text>(.*?)</review_text>', content, re.DOTALL)
        return [' '.join(t.strip().split()) for t in texts if t.strip()]

    pos = get_reviews(os.path.join(path_data, 'positive.review'))
    neg = get_reviews(os.path.join(path_data, 'negative.review'))

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    sp, sn = int(len(pos) * .8), int(len(neg) * .8)
    train_data = pos[:sp] + neg[:sn]
    test_data = pos[sp:] + neg[sn:]
    train_labels = np.array([1] * sp + [0] * sn)
    test_labels = np.array([1] * (len(pos) - sp) + [0] * (len(neg) - sn))
    print(f"Parsed {len(pos)} positive and {len(neg)} negative reviews "
          f"from {path_data}")
    return train_data, train_labels, test_data, test_labels


def _score(shouldnt_trust, says_mistrust, label):
    p, r, f1, _ = precision_recall_fscore_support(
        shouldnt_trust, says_mistrust, average='binary', zero_division=0)
    pt, rt, f1t, _ = precision_recall_fscore_support(
        ~shouldnt_trust, ~says_mistrust, average='binary', zero_division=0)
    return {
        f'{label}_mistrust_precision': p, f'{label}_mistrust_recall': r,
        f'{label}_mistrust_f1': f1,
        f'{label}_trust_precision': pt, f'{label}_trust_recall': rt,
        f'{label}_trust_f1': f1t,
    }


def run_experiment(train_data, train_labels, test_data, test_labels,
                    num_features=10, percent_untrustworthy=0.25,
                    num_rounds=10, n_test=60, num_samples=800, seed=1):
    vectorizer = CountVectorizer(lowercase=True, binary=True, min_df=3,
                                  stop_words='english')
    train_vectors = vectorizer.fit_transform(train_data)
    vocab = np.array(vectorizer.get_feature_names_out())

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_vectors, train_labels)

    def predict_proba(raw_texts):
        return clf.predict_proba(vectorizer.transform(raw_texts))

    rng = np.random.RandomState(seed)
    candidate_idx = [i for i, t in enumerate(test_data) if len(t.split()) >= 5]
    n_test = min(n_test, len(candidate_idx))
    idx = rng.choice(candidate_idx, n_test, replace=False)
    subset_texts = [test_data[i] for i in idx]

    test_vectors = vectorizer.transform(subset_texts)
    base_preds = clf.predict(test_vectors)

    print(f"Vocabulary size: {len(vocab)}  |  Train docs: {len(train_data)}  "
          f"|  Test docs used: {n_test}  |  Base test accuracy on subset: "
          f"{(base_preds == test_labels[idx]).mean():.3f}")

    explainer = LimeTextExplainer(class_names=['neg', 'pos'])
    # each entry: (intercept_for_label_1, [(word, weight), ...], set_of_words)
    explanations = []
    print(f"Computing LIME explanations for {n_test} instances "
          f"(num_samples={num_samples})...")
    for i, text in enumerate(subset_texts):
        exp = explainer.explain_instance(
            text, predict_proba, num_features=num_features,
            num_samples=num_samples, labels=(1,))
        word_weights = [(w.lower(), wt) for w, wt in exp.as_list(label=1)]
        icpt = exp.intercept[1]
        explanations.append((icpt, word_weights, {w for w, _ in word_weights}))
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{n_test} done")

    num_untrustworthy = max(1, int(len(vocab) * percent_untrustworthy))
    all_rounds = []

    for r in range(num_rounds):
        round_rng = np.random.RandomState(seed + r)
        untrustworthy_idx = round_rng.choice(len(vocab), num_untrustworthy, replace=False)
        untrustworthy_words = set(vocab[untrustworthy_idx])

        corrupted = test_vectors.copy().tolil()
        corrupted[:, untrustworthy_idx] = 0
        corrupted_preds = clf.predict(corrupted.tocsr())
        shouldnt_trust = (corrupted_preds != base_preds)

        # Baseline rule (this script's original, cruder heuristic): mistrust
        # if ANY untrustworthy word appears anywhere in the top-K explanation,
        # regardless of its weight or sign.
        baseline_mistrust = np.array([
            len(words & untrustworthy_words) > 0
            for _, _, words in explanations
        ])

        # Paper's rule (matches data_trusting.py's trust_fn): reconstruct the
        # local linear model's prediction with and without the untrustworthy
        # features' contribution, and flag mistrust only if that changes
        # which side of 0.5 it lands on.
        linear_mistrust = np.array([
            ((icpt + sum(wt for _, wt in ws)) > 0.5) !=
            ((icpt + sum(wt for w, wt in ws if w not in untrustworthy_words)) > 0.5)
            for icpt, ws, _ in explanations
        ])

        round_result = {'flipped': int(shouldnt_trust.sum())}
        round_result.update(_score(shouldnt_trust, baseline_mistrust, 'baseline'))
        round_result.update(_score(shouldnt_trust, linear_mistrust, 'linear'))
        all_rounds.append(round_result)

    def agg(key):
        vals = [r[key] for r in all_rounds]
        return np.mean(vals), np.std(vals)

    print()
    fm, fs = agg('flipped')
    print(f"Average flipped predictions per round: {fm:.1f} +- {fs:.1f} "
          f"(out of {n_test})")
    print()
    print("=== Baseline rule (any untrustworthy word in top-K explanation) ===")
    for metric in ['mistrust_precision', 'mistrust_recall', 'mistrust_f1']:
        m, s = agg(f'baseline_{metric}')
        print(f"  {metric}: {m:.3f} +- {s:.3f}")
    print()
    print("=== Paper's rule (local linear model, intercept + weights) ===")
    for metric in ['mistrust_precision', 'mistrust_recall', 'mistrust_f1',
                   'trust_precision', 'trust_recall', 'trust_f1']:
        m, s = agg(f'linear_{metric}')
        print(f"  {metric}: {m:.3f} +- {s:.3f}")
    return all_rounds


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', '-d', required=True,
                     help="path containing positive.review / negative.review "
                          "(e.g. sorted_data_acl/books)")
    ap.add_argument('--num-features', '-k', type=int, default=10)
    ap.add_argument('--percent-untrustworthy', '-u', type=float, default=0.25)
    ap.add_argument('--num-rounds', '-r', type=int, default=10)
    ap.add_argument('--n-test', type=int, default=60)
    ap.add_argument('--num-samples', type=int, default=800)
    args = ap.parse_args()

    train_data, train_labels, test_data, test_labels = load_multidomain(args.data_dir)
    run_experiment(train_data, train_labels, test_data, test_labels,
                   num_features=args.num_features,
                   percent_untrustworthy=args.percent_untrustworthy,
                   num_rounds=args.num_rounds, n_test=args.n_test,
                   num_samples=args.num_samples)
