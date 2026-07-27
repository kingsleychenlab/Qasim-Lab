#!/usr/bin/env python3
"""
The decoding stage: EEG -> predicted T5 embedding -> fidelity, for one session.

This is the core of the project. step09/step10 call it once per subject/session,
and the fidelity column that every downstream result rests on comes out of here.
It runs one session at a time on purpose. Ridge is fit within a session, so a
model never sees another session's trials.

    X (576 x 32250 EEG features) --ridge--> Y_hat (576 x 1024)
    fidelity = cosine(Y_hat, true embedding)

The extra metrics are there because raw cosine on its own is misleading. Every
T5 vector shares a large common direction, so predicting roughly "the average
word" already scores ~0.85. The word-specific metrics below ask the question
that actually matters, whether the correct word is ranked above the other 575,
and the shuffled-label control shows what that number looks like when no signal
exists at all.

  raw_cosine            cosine(pred, true)
  centered_cosine       cosine after subtracting the fold's Y_train mean, which
                        removes the shared direction
  true_word_rank        rank of the correct word among all 576 candidate T5
                        embeddings by cosine to the prediction (1 = best)
  true_word_percentile  1 - (rank-1)/575  (higher = better; 0.5 = chance)
  top1/5/10_correct     retrieval hits
  centered_* versions   the same retrieval after centering both the prediction
                        and the candidates by the fold's Y_train mean
  shuffled control      identical CV with Y permuted across trials, same folds.
                        Real must beat shuffled or there is no word-specific
                        decoding, whatever the raw cosine says.

Centering uses the training fold's mean, not the test trials'. Computing it over
all 576 would leak test information into the metric.

CV: 5-fold KFold(shuffle=True, random_state=42), StandardScaler fit on X_train
only, Ridge(alpha=10000, solver="svd"), exactly one out-of-fold prediction per
trial. alpha is deliberately huge because p >> n here (32,250 features from 576
trials); without heavy shrinkage ridge would interpolate the training set.

Outputs (per session, under outputs/subjects/<subject>_ses-<session>/):
  fidelity_results_corrected.csv
  predicted_embeddings_corrected.npy
  ridge_corrected_metadata.json
  ridge_corrected_summary.txt
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from common import Tee, load_word_to_row

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRIALS = 576
Y_DIM = 1024
ALPHA = 10000.0
FOLDS = 5
SEED = 42
SHUFFLE_SEED = 2024


def unit_rows(rows):
    """Scale each row to unit L2 norm. The 1e-12 guards a zero row."""
    return rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12)


def cosine_matrix(preds, candidates, candidates_unit=None):
    """Row-wise cosine between preds (n,d) and candidates (m,d) -> (n,m).

    Pass candidates_unit when the candidate matrix is fixed across calls, so its
    normalization is not recomputed on every fold.
    """
    normed_candidates = unit_rows(candidates) if candidates_unit is None else candidates_unit
    return unit_rows(preds) @ normed_candidates.T


def cv_predict(X, Ymat, splits):
    """Out-of-fold Ridge predictions + per-fold train means + fold assignment.

    Each trial is predicted exactly once, by a model that never saw it. The
    scaler is fit on the training rows alone. Fitting it on all of X first would
    leak the test trials' means and variances into the model.
    """
    pred = np.full((X.shape[0], Ymat.shape[1]), np.nan)
    fold_mean = {}
    fold_assign = np.full(X.shape[0], -1, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splits):
        scaler = StandardScaler().fit(X[train_idx])
        # solver="svd" is the stable choice when p >> n: the normal-equation
        # solvers square the condition number of a 576 x 32250 matrix.
        model = Ridge(alpha=ALPHA, solver="svd")
        model.fit(scaler.transform(X[train_idx]), Ymat[train_idx])
        pred[test_idx] = model.predict(scaler.transform(X[test_idx]))
        # Train-fold mean of Y, used later to center out the shared embedding
        # direction. Taken from the training rows only, for the same reason.
        fold_mean[fold] = Ymat[train_idx].mean(axis=0)
        fold_assign[test_idx] = fold
    return pred, fold_mean, fold_assign


def retrieval_metrics(pred, candidates, correct_idx, fold_mean, fold_assign, candidates_unit=None):
    """Per-trial raw + centered cosine-to-correct, rank, percentile, top-k.

    Done fold by fold because the centering mean is fold-specific: each test
    trial is scored against its own training fold's Y mean, not a global one.

    Rank is the count of candidates scoring strictly above the correct word,
    plus one. Ties therefore favour the correct word, which keeps the better
    rank. That is the conservative direction for a null result: it can only
    flatter the decoder, and the decoder still came out at chance.
    """
    n_trials = pred.shape[0]
    metrics = {k: np.zeros(n_trials) for k in (
        "raw_cosine", "centered_cosine",
        "true_word_rank", "true_word_percentile",
        "top1_correct", "top5_correct", "top10_correct",
        "centered_true_word_rank", "centered_true_word_percentile",
        "centered_top1_correct", "centered_top5_correct", "centered_top10_correct")}

    def score_fold(sims, test_idx, correct_col, cosine_key, prefix):
        """Score one fold's test trials against all 576 candidates at once."""
        correct_sim = sims[np.arange(len(test_idx)), correct_col]
        rank = 1 + (sims > correct_sim[:, None]).sum(axis=1)
        metrics[cosine_key][test_idx] = correct_sim
        metrics[prefix + "true_word_rank"][test_idx] = rank
        metrics[prefix + "true_word_percentile"][test_idx] = 1.0 - (rank - 1) / (N_TRIALS - 1)
        for k in (1, 5, 10):
            metrics[f"{prefix}top{k}_correct"][test_idx] = (rank <= k).astype(int)

    for fold in sorted(set(fold_assign.tolist())):
        test_idx = np.where(fold_assign == fold)[0]
        correct_col = correct_idx[test_idx]
        train_mean = fold_mean[fold]

        # Raw: the candidate matrix is fixed, so reuse its normalized form.
        sims = cosine_matrix(pred[test_idx], candidates, candidates_unit=candidates_unit)
        score_fold(sims, test_idx, correct_col, "raw_cosine", "")

        # Centered: subtracting the fold mean moves the candidates too, so this
        # normalization genuinely has to be redone per fold.
        sims_centered = cosine_matrix(pred[test_idx] - train_mean, candidates - train_mean)
        score_fold(sims_centered, test_idx, correct_col, "centered_cosine", "centered_")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", default=os.path.join(HERE, "outputs/X_eeg.npy"))
    ap.add_argument("--y", default=os.path.join(HERE, "outputs/Y_t5.npy"))
    ap.add_argument("--meta", default=os.path.join(HERE, "outputs/trial_metadata.csv"))
    ap.add_argument("--embeddings", default=os.path.join(HERE, "results/embeddings/peers_t5large_embeddings.npy"))
    ap.add_argument("--order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    ap.add_argument("--out-csv", default=os.path.join(HERE, "outputs/fidelity_results_corrected.csv"))
    ap.add_argument("--out-pred", default=os.path.join(HERE, "outputs/predicted_embeddings_corrected.npy"))
    ap.add_argument("--out-meta", default=os.path.join(HERE, "outputs/ridge_corrected_metadata.json"))
    ap.add_argument("--out-summary", default=os.path.join(HERE, "outputs/ridge_corrected_summary.txt"))
    args = ap.parse_args()

    for path in (args.x, args.y, args.meta, args.embeddings, args.order):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: required input not found: {path}")

    X = np.load(args.x).astype(np.float64)
    Y = np.load(args.y).astype(np.float64)
    meta = pd.read_csv(args.meta)
    candidates = np.load(args.embeddings).astype(np.float64)    # (576, 1024) candidates
    word_to_row = load_word_to_row(args.order)
    subject = str(meta.subject.iloc[0]) if "subject" in meta.columns else "?"
    session = int(meta.session.iloc[0]) if "session" in meta.columns else -1

    report_file = open(args.out_summary, "w")
    log = Tee(report_file)
    check = log.check

    log("=" * 74)
    log(f"CORRECTED DECODING SMOKE TEST — single subject {subject}/ses-{session}")
    log("(NOT final inference; no mixed-effects; word-specific metrics)")
    log("=" * 74)
    log(f"X {X.shape}  Y {Y.shape}  candidates {candidates.shape}  trials {len(meta)}")
    log(f"alpha={ALPHA} folds={FOLDS} seed={SEED} shuffle_seed={SHUFFLE_SEED} solver=svd")

    if not (X.shape[0] == Y.shape[0] == len(meta) == candidates.shape[0] == N_TRIALS):
        sys.exit("ERROR: row counts disagree or != 576.")

    # Correct candidate index per trial (the trial's own word).
    correct_idx = np.array([word_to_row[str(w).upper()] for w in meta.word], dtype=int)
    if not np.allclose(candidates[correct_idx], Y, atol=1e-4):
        log("WARNING: emb[correct_idx] != Y within 1e-4 (unexpected).")

    # Fixed folds, shared by the real and shuffled runs, so the control differs
    # only in the labels and nothing else.
    splits = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(np.arange(N_TRIALS)))

    # The 576 candidates never change, so normalize them once.
    candidates_unit = unit_rows(candidates)

    log("\n[real] running out-of-fold ridge ...")
    pred, fold_mean, fold_assign = cv_predict(X, Y, splits)
    predicted_count = np.zeros(N_TRIALS, dtype=int)
    for fold, (_, test_idx) in enumerate(splits):
        predicted_count[test_idx] += 1
    real_metrics = retrieval_metrics(pred, candidates, correct_idx, fold_mean, fold_assign,
                                     candidates_unit=candidates_unit)

    # Shuffled-label control. Permuting Y across trials destroys the EEG-to-word
    # correspondence while leaving X, the folds, the alpha and the metric
    # untouched, so whatever this scores is what "no signal" looks like for this
    # session. Real must clear it; on this dataset it does not.
    log("[shuffled] running negative control (Y permuted across trials, same folds) ...")
    rng = np.random.RandomState(SHUFFLE_SEED)
    permutation = rng.permutation(N_TRIALS)
    Y_shuffled = Y[permutation]
    pred_shuffled, fold_mean_shuffled, fold_assign_shuffled = cv_predict(X, Y_shuffled, splits)
    # Scored against the true correct word: candidates and correct_idx are
    # deliberately left unpermuted, so this asks whether a model trained on
    # scrambled labels can still find the right word. The answer has to be no.
    shuffled_metrics = retrieval_metrics(pred_shuffled, candidates, correct_idx,
                                         fold_mean_shuffled, fold_assign_shuffled,
                                         candidates_unit=candidates_unit)

    # Assemble the per-trial CSV (real metrics only).
    results = pd.DataFrame({
        "subject": meta.subject, "session": meta.session, "trial": meta.trial,
        "serialpos": meta.serialpos, "word": meta.word, "recalled": meta.recalled,
        "raw_cosine": real_metrics["raw_cosine"],
        "centered_cosine": real_metrics["centered_cosine"],
        "true_word_rank": real_metrics["true_word_rank"].astype(int),
        "true_word_percentile": real_metrics["true_word_percentile"],
        "top1_correct": real_metrics["top1_correct"].astype(int),
        "top5_correct": real_metrics["top5_correct"].astype(int),
        "top10_correct": real_metrics["top10_correct"].astype(int),
        "centered_true_word_rank": real_metrics["centered_true_word_rank"].astype(int),
        "centered_true_word_percentile": real_metrics["centered_true_word_percentile"],
        "centered_top1_correct": real_metrics["centered_top1_correct"].astype(int),
        "centered_top5_correct": real_metrics["centered_top5_correct"].astype(int),
        "centered_top10_correct": real_metrics["centered_top10_correct"].astype(int),
    })

    log("\n=== VALIDATION ===")
    check("fidelity_results_corrected has 576 rows", len(results) == N_TRIALS, f"{len(results)}")
    check("predicted_embeddings_corrected shape 576 x 1024",
          pred.shape == (N_TRIALS, Y_DIM), f"{pred.shape}")
    check("no NaN/Inf in predictions",
          not np.isnan(pred).any() and not np.isinf(pred).any())
    numeric = results.drop(columns=["subject", "word"])
    check("no NaN/Inf anywhere in results",
          not numeric.isna().any().any()
          and np.isfinite(numeric.select_dtypes("number").to_numpy()).all())
    check("every trial predicted exactly once", bool((predicted_count == 1).all()),
          f"unique counts={sorted(set(predicted_count.tolist()))}")
    for col in ("true_word_rank", "centered_true_word_rank"):
        check(f"{col} in [1, 576]",
              bool((results[col] >= 1).all() and (results[col] <= N_TRIALS).all()),
              f"min {results[col].min()} max {results[col].max()}")
    for col in ("true_word_percentile", "centered_true_word_percentile"):
        check(f"{col} in [0, 1]",
              bool((results[col] >= 0).all() and (results[col] <= 1).all()),
              f"min {results[col].min():.4f} max {results[col].max():.4f}")
    for col in ("top1_correct", "top5_correct", "top10_correct",
                "centered_top1_correct", "centered_top5_correct", "centered_top10_correct"):
        check(f"{col} is 0/1", set(results[col].unique()) <= {0, 1},
              f"values={sorted(results[col].unique().tolist())}")

    log("\n=== REMEMBERED vs FORGOTTEN (real labels) ===")
    recalled_mask = results.recalled == 1
    metric_cols = ["raw_cosine", "centered_cosine", "true_word_rank",
                   "true_word_percentile", "top1_correct", "top5_correct",
                   "top10_correct", "centered_true_word_rank",
                   "centered_true_word_percentile", "centered_top1_correct",
                   "centered_top5_correct", "centered_top10_correct"]
    log(f"n recalled={int(recalled_mask.sum())}  n forgotten={int((~recalled_mask).sum())}")
    log(f"{'metric':<32} {'remembered':>11} {'forgotten':>11} {'rem-forg':>10}")
    remembered_vs_forgotten = {}
    for metric in metric_cols:
        remembered_mean = float(results.loc[recalled_mask, metric].mean())
        forgotten_mean = float(results.loc[~recalled_mask, metric].mean())
        remembered_vs_forgotten[metric] = {"remembered": remembered_mean,
                                           "forgotten": forgotten_mean,
                                           "diff": remembered_mean - forgotten_mean}
        log(f"{metric:<32} {remembered_mean:>11.4f} {forgotten_mean:>11.4f} {remembered_mean-forgotten_mean:>+10.4f}")

    log("\n=== REAL vs SHUFFLED-LABEL CONTROL (retrieval, all trials) ===")
    log("chance: mean rank ~288.5, mean percentile ~0.5, top1 ~0.0017, "
        "top5 ~0.0087, top10 ~0.0174")
    comparison_cols = ["true_word_rank", "true_word_percentile", "top1_correct",
                       "top5_correct", "top10_correct", "centered_true_word_rank",
                       "centered_true_word_percentile", "centered_top1_correct",
                       "centered_top5_correct", "centered_top10_correct"]
    log(f"{'metric':<32} {'real':>11} {'shuffled':>11}")
    real_vs_shuffled = {}
    for metric in comparison_cols:
        real_mean = float(np.mean(real_metrics[metric]))
        shuffled_mean = float(np.mean(shuffled_metrics[metric]))
        real_vs_shuffled[metric] = {"real": real_mean, "shuffled": shuffled_mean}
        log(f"{metric:<32} {real_mean:>11.4f} {shuffled_mean:>11.4f}")

    results.to_csv(args.out_csv, index=False)
    np.save(args.out_pred, pred.astype(np.float32))
    meta_json = {
        "kind": "single-subject CORRECTED decoding smoke test (NOT final inference)",
        "subject": subject, "session": session, "task": "ltpFR2",
        "n_trials": N_TRIALS, "alpha": ALPHA, "n_folds": FOLDS,
        "kfold_shuffle": True, "seed": SEED, "shuffle_seed": SHUFFLE_SEED,
        "ridge_solver": "svd", "n_candidates": int(candidates.shape[0]),
        "metrics_overall": {metric: float(np.mean(real_metrics[metric])) for metric in metric_cols},
        "remembered_vs_forgotten": remembered_vs_forgotten,
        "real_vs_shuffled": real_vs_shuffled,
        "n_recalled": int(recalled_mask.sum()), "n_forgotten": int((~recalled_mask).sum()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Smoke test only. Single subject. No mixed-effects model, "
                      "no significance testing. Not final inference.",
    }
    json.dump(meta_json, open(args.out_meta, "w"), indent=2)

    log("\nwrote:")
    for path in (args.out_csv, args.out_pred, args.out_meta, args.out_summary):
        log(f"  {os.path.relpath(path, HERE)}")
    log("\n*** SINGLE-SUBJECT CORRECTED DECODING SMOKE TEST — not final inference, "
        "no mixed-effects. ***")
    log("STATUS: " + ("OK" if log.fail == 0 else f"FAILED ({log.fail} checks)"))
    report_file.close()
    sys.exit(0 if log.fail == 0 else 1)


if __name__ == "__main__":
    main()
