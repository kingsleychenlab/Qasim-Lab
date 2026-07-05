#!/usr/bin/env python3
"""
Corrected decoding-metric analysis — single subject LTP269/ses-20.

** Single-subject CORRECTED DECODING SMOKE TEST — NOT final inference. **
No mixed-effects model, no significance testing. Replaces the inflated
raw-cosine-only metric with word-specific decoding metrics:

  - raw_cosine            cosine(pred, true)
  - centered_cosine       cosine after subtracting the fold's Y_train mean
  - true_word_rank        rank of the correct word among all 576 candidate
                          T5 embeddings by cosine to the prediction (1 = best)
  - true_word_percentile  1 - (rank-1)/575  (higher = better)
  - top1/5/10_correct     retrieval hits
  - centered_* versions   same retrieval after centering pred + candidates by
                          the fold's Y_train mean
  - shuffled-label control: identical CV with Y permuted across trials
                          (same folds) — real must beat shuffled if there is
                          genuine word-specific decoding.

CV: 5-fold KFold(shuffle=True, random_state=42), StandardScaler on X_train
only, Ridge(alpha=10000, solver="svd"), one out-of-fold prediction per trial.

Outputs:
  outputs/fidelity_results_corrected.csv
  outputs/predicted_embeddings_corrected.npy
  outputs/ridge_corrected_metadata.json
  outputs/ridge_corrected_summary.txt
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRIALS = 576
Y_DIM = 1024
ALPHA = 10000.0
FOLDS = 5
SEED = 42
SHUFFLE_SEED = 2024


def cosine_matrix(P, C):
    """Row-wise cosine between P (n,d) and C (m,d) -> (n,m)."""
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return Pn @ Cn.T


def cv_predict(X, Ymat, splits):
    """Out-of-fold Ridge predictions + per-fold train means + fold assignment."""
    pred = np.full((X.shape[0], Ymat.shape[1]), np.nan)
    fold_mean = {}
    fold_assign = np.full(X.shape[0], -1, dtype=int)
    for f, (tr, te) in enumerate(splits):
        scaler = StandardScaler().fit(X[tr])
        model = Ridge(alpha=ALPHA, solver="svd")
        model.fit(scaler.transform(X[tr]), Ymat[tr])
        pred[te] = model.predict(scaler.transform(X[te]))
        fold_mean[f] = Ymat[tr].mean(axis=0)
        fold_assign[te] = f
    return pred, fold_mean, fold_assign


def retrieval_metrics(pred, emb, correct_idx, fold_mean, fold_assign):
    """Per-trial raw + centered cosine-to-correct, rank, percentile, top-k.
    Computed per fold so centering uses that fold's Y_train mean."""
    n = pred.shape[0]
    out = {k: np.zeros(n) for k in (
        "raw_cosine", "centered_cosine",
        "true_word_rank", "true_word_percentile",
        "top1_correct", "top5_correct", "top10_correct",
        "centered_true_word_rank", "centered_true_word_percentile",
        "centered_top1_correct", "centered_top5_correct", "centered_top10_correct")}

    for f in sorted(set(fold_assign.tolist())):
        te = np.where(fold_assign == f)[0]
        mean_f = fold_mean[f]

        # ---- raw ----
        sims = cosine_matrix(pred[te], emb)                 # (nte, 576)
        # ---- centered ----
        sims_c = cosine_matrix(pred[te] - mean_f, emb - mean_f)

        for j, i in enumerate(te):
            ci = correct_idx[i]
            s = sims[j]
            sc = sims_c[j]
            corr = s[ci]
            corr_c = sc[ci]
            rank = int(1 + np.sum(s > corr))
            rank_c = int(1 + np.sum(sc > corr_c))
            out["raw_cosine"][i] = corr
            out["centered_cosine"][i] = corr_c
            out["true_word_rank"][i] = rank
            out["true_word_percentile"][i] = 1.0 - (rank - 1) / (N_TRIALS - 1)
            out["top1_correct"][i] = int(rank <= 1)
            out["top5_correct"][i] = int(rank <= 5)
            out["top10_correct"][i] = int(rank <= 10)
            out["centered_true_word_rank"][i] = rank_c
            out["centered_true_word_percentile"][i] = 1.0 - (rank_c - 1) / (N_TRIALS - 1)
            out["centered_top1_correct"][i] = int(rank_c <= 1)
            out["centered_top5_correct"][i] = int(rank_c <= 5)
            out["centered_top10_correct"][i] = int(rank_c <= 10)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", default=os.path.join(HERE, "outputs/X_eeg.npy"))
    ap.add_argument("--y", default=os.path.join(HERE, "outputs/Y_t5.npy"))
    ap.add_argument("--meta", default=os.path.join(HERE, "outputs/trial_metadata.csv"))
    ap.add_argument("--embeddings", default=os.path.join(HERE, "peers_t5large_embeddings.npy"))
    ap.add_argument("--order", default=os.path.join(HERE, "peers_word_order.csv"))
    ap.add_argument("--out-csv", default=os.path.join(HERE, "outputs/fidelity_results_corrected.csv"))
    ap.add_argument("--out-pred", default=os.path.join(HERE, "outputs/predicted_embeddings_corrected.npy"))
    ap.add_argument("--out-meta", default=os.path.join(HERE, "outputs/ridge_corrected_metadata.json"))
    ap.add_argument("--out-summary", default=os.path.join(HERE, "outputs/ridge_corrected_summary.txt"))
    args = ap.parse_args()

    for p in (args.x, args.y, args.meta, args.embeddings, args.order):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: required input not found: {p}")

    X = np.load(args.x).astype(np.float64)
    Y = np.load(args.y).astype(np.float64)
    meta = pd.read_csv(args.meta)
    emb = np.load(args.embeddings).astype(np.float64)          # (576, 1024) candidates
    order = pd.read_csv(args.order)
    word_to_row = dict(zip(order.word.str.upper(), order.row_index.astype(int)))
    subj = str(meta.subject.iloc[0]) if "subject" in meta.columns else "?"
    sess = int(meta.session.iloc[0]) if "session" in meta.columns else -1

    fh = open(args.out_summary, "w")

    def log(*p):
        line = " ".join(str(x) for x in p)
        print(line)
        fh.write(line + "\n")

    fails = [0]

    def check(label, cond, detail=""):
        if not cond:
            fails[0] += 1
        log(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    log("=" * 74)
    log(f"CORRECTED DECODING SMOKE TEST — single subject {subj}/ses-{sess}")
    log("(NOT final inference; no mixed-effects; word-specific metrics)")
    log("=" * 74)
    log(f"X {X.shape}  Y {Y.shape}  candidates {emb.shape}  trials {len(meta)}")
    log(f"alpha={ALPHA} folds={FOLDS} seed={SEED} shuffle_seed={SHUFFLE_SEED} solver=svd")

    if not (X.shape[0] == Y.shape[0] == len(meta) == emb.shape[0] == N_TRIALS):
        sys.exit("ERROR: row counts disagree or != 576.")

    # correct candidate index per trial (the trial's own word)
    correct_idx = np.array([word_to_row[str(w).upper()] for w in meta.word], dtype=int)
    # sanity: candidate at correct_idx must equal the built target Y
    if not np.allclose(emb[correct_idx], Y, atol=1e-4):
        log("WARNING: emb[correct_idx] != Y within 1e-4 (unexpected).")

    splits = list(KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(np.arange(N_TRIALS)))

    # ---------------- REAL labels ----------------
    log("\n[real] running out-of-fold ridge ...")
    pred, fold_mean, fold_assign = cv_predict(X, Y, splits)
    predicted_count = np.zeros(N_TRIALS, dtype=int)
    for f, (_, te) in enumerate(splits):
        predicted_count[te] += 1
    real = retrieval_metrics(pred, emb, correct_idx, fold_mean, fold_assign)

    # ---------------- SHUFFLED-label control ----------------
    log("[shuffled] running negative control (Y permuted across trials, same folds) ...")
    rng = np.random.RandomState(SHUFFLE_SEED)
    perm = rng.permutation(N_TRIALS)
    Y_shuf = Y[perm]
    pred_s, fold_mean_s, fold_assign_s = cv_predict(X, Y_shuf, splits)
    # Evaluate against the TRUE correct word (candidates/correct_idx unchanged).
    shuf = retrieval_metrics(pred_s, emb, correct_idx, fold_mean_s, fold_assign_s)

    # ---------------- assemble CSV (real metrics only) ----------------
    res = pd.DataFrame({
        "subject": meta.subject, "session": meta.session, "trial": meta.trial,
        "serialpos": meta.serialpos, "word": meta.word, "recalled": meta.recalled,
        "raw_cosine": real["raw_cosine"],
        "centered_cosine": real["centered_cosine"],
        "true_word_rank": real["true_word_rank"].astype(int),
        "true_word_percentile": real["true_word_percentile"],
        "top1_correct": real["top1_correct"].astype(int),
        "top5_correct": real["top5_correct"].astype(int),
        "top10_correct": real["top10_correct"].astype(int),
        "centered_true_word_rank": real["centered_true_word_rank"].astype(int),
        "centered_true_word_percentile": real["centered_true_word_percentile"],
        "centered_top1_correct": real["centered_top1_correct"].astype(int),
        "centered_top5_correct": real["centered_top5_correct"].astype(int),
        "centered_top10_correct": real["centered_top10_correct"].astype(int),
    })

    # ---------------- validation ----------------
    log("\n=== VALIDATION ===")
    check("fidelity_results_corrected has 576 rows", len(res) == N_TRIALS, f"{len(res)}")
    check("predicted_embeddings_corrected shape 576 x 1024",
          pred.shape == (N_TRIALS, Y_DIM), f"{pred.shape}")
    check("no NaN/Inf in predictions",
          not np.isnan(pred).any() and not np.isinf(pred).any())
    numeric = res.drop(columns=["subject", "word"])
    check("no NaN/Inf anywhere in results",
          not numeric.isna().any().any()
          and np.isfinite(numeric.select_dtypes("number").to_numpy()).all())
    check("every trial predicted exactly once", bool((predicted_count == 1).all()),
          f"unique counts={sorted(set(predicted_count.tolist()))}")
    for col in ("true_word_rank", "centered_true_word_rank"):
        check(f"{col} in [1, 576]",
              bool((res[col] >= 1).all() and (res[col] <= N_TRIALS).all()),
              f"min {res[col].min()} max {res[col].max()}")
    for col in ("true_word_percentile", "centered_true_word_percentile"):
        check(f"{col} in [0, 1]",
              bool((res[col] >= 0).all() and (res[col] <= 1).all()),
              f"min {res[col].min():.4f} max {res[col].max():.4f}")
    for col in ("top1_correct", "top5_correct", "top10_correct",
                "centered_top1_correct", "centered_top5_correct", "centered_top10_correct"):
        check(f"{col} is 0/1", set(res[col].unique()) <= {0, 1},
              f"values={sorted(res[col].unique().tolist())}")

    # ---------------- remembered vs forgotten ----------------
    log("\n=== REMEMBERED vs FORGOTTEN (real labels) ===")
    rec_mask = res.recalled == 1
    metric_cols = ["raw_cosine", "centered_cosine", "true_word_rank",
                   "true_word_percentile", "top1_correct", "top5_correct",
                   "top10_correct", "centered_true_word_rank",
                   "centered_true_word_percentile", "centered_top1_correct",
                   "centered_top5_correct", "centered_top10_correct"]
    log(f"n recalled={int(rec_mask.sum())}  n forgotten={int((~rec_mask).sum())}")
    log(f"{'metric':<32} {'remembered':>11} {'forgotten':>11} {'rem-forg':>10}")
    rem_forg = {}
    for c in metric_cols:
        rm = float(res.loc[rec_mask, c].mean())
        fm = float(res.loc[~rec_mask, c].mean())
        rem_forg[c] = {"remembered": rm, "forgotten": fm, "diff": rm - fm}
        log(f"{c:<32} {rm:>11.4f} {fm:>11.4f} {rm-fm:>+10.4f}")

    # ---------------- real vs shuffled ----------------
    log("\n=== REAL vs SHUFFLED-LABEL CONTROL (retrieval, all trials) ===")
    log("chance: mean rank ~288.5, mean percentile ~0.5, top1 ~0.0017, "
        "top5 ~0.0087, top10 ~0.0174")
    cmp_cols = ["true_word_rank", "true_word_percentile", "top1_correct",
                "top5_correct", "top10_correct", "centered_true_word_rank",
                "centered_true_word_percentile", "centered_top1_correct",
                "centered_top5_correct", "centered_top10_correct"]
    log(f"{'metric':<32} {'real':>11} {'shuffled':>11}")
    real_vs_shuf = {}
    for c in cmp_cols:
        rmean = float(np.mean(real[c]))
        smean = float(np.mean(shuf[c]))
        real_vs_shuf[c] = {"real": rmean, "shuffled": smean}
        log(f"{c:<32} {rmean:>11.4f} {smean:>11.4f}")

    # ---------------- save ----------------
    res.to_csv(args.out_csv, index=False)
    np.save(args.out_pred, pred.astype(np.float32))
    meta_json = {
        "kind": "single-subject CORRECTED decoding smoke test (NOT final inference)",
        "subject": subj, "session": sess, "task": "ltpFR2",
        "n_trials": N_TRIALS, "alpha": ALPHA, "n_folds": FOLDS,
        "kfold_shuffle": True, "seed": SEED, "shuffle_seed": SHUFFLE_SEED,
        "ridge_solver": "svd", "n_candidates": int(emb.shape[0]),
        "metrics_overall": {c: float(np.mean(real[c])) for c in metric_cols},
        "remembered_vs_forgotten": rem_forg,
        "real_vs_shuffled": real_vs_shuf,
        "n_recalled": int(rec_mask.sum()), "n_forgotten": int((~rec_mask).sum()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Smoke test only. Single subject. No mixed-effects model, "
                      "no significance testing. Not final inference.",
    }
    json.dump(meta_json, open(args.out_meta, "w"), indent=2)

    log("\nwrote:")
    for p in (args.out_csv, args.out_pred, args.out_meta, args.out_summary):
        log(f"  {os.path.relpath(p, HERE)}")
    log("\n*** SINGLE-SUBJECT CORRECTED DECODING SMOKE TEST — not final inference, "
        "no mixed-effects. ***")
    log("STATUS: " + ("OK" if fails[0] == 0 else f"FAILED ({fails[0]} checks)"))
    fh.close()
    sys.exit(0 if fails[0] == 0 else 1)


if __name__ == "__main__":
    main()
