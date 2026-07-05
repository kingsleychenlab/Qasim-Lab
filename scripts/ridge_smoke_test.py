#!/usr/bin/env python3
"""
Ridge-regression SMOKE TEST (single subject: LTP269, ses-20).

** This is a smoke test, NOT the final analysis and NOT final inference. **
No mixed-effects model. One subject, held-out-trial K-fold cross-validation,
strong ridge regularization. Purpose: sanity-check that raw 300-800 ms EEG
carries *any* linearly decodable signal about the T5 target, and whether
remembered trials decode better than forgotten ones.

Pipeline (per fold):
  1. KFold split over trials (train/test disjoint — never test on a trained trial).
  2. StandardScaler fit on TRAIN X only, applied to train & test.
  3. Ridge(alpha) fit X_train -> Y_train (1024-dim multi-output).
  4. Predict held-out Y (1024-dim) for the test trials.
Each trial receives exactly one out-of-fold prediction.
Fidelity = cosine similarity(predicted, true) per held-out trial.

Outputs:
  outputs/fidelity_results.csv        (subject,session,trial,serialpos,word,recalled,embedding_fidelity)
  outputs/predicted_embeddings.npy    (576 x 1024)
  outputs/ridge_cv_metadata.json
  outputs/ridge_smoke_summary.txt
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


class Tee:
    def __init__(self, fh):
        self.fh = fh
        self.fail = 0

    def __call__(self, *p):
        line = " ".join(str(x) for x in p)
        print(line)
        self.fh.write(line + "\n")

    def check(self, label, cond, detail=""):
        if not cond:
            self.fail += 1
        self(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


def row_cosine(A, B):
    """Row-wise cosine similarity between two (n, d) arrays."""
    an = np.linalg.norm(A, axis=1)
    bn = np.linalg.norm(B, axis=1)
    denom = an * bn
    out = np.full(A.shape[0], np.nan)
    ok = denom > 0
    out[ok] = np.sum(A[ok] * B[ok], axis=1) / denom[ok]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", default=os.path.join(HERE, "outputs/X_eeg.npy"))
    ap.add_argument("--y", default=os.path.join(HERE, "outputs/Y_t5.npy"))
    ap.add_argument("--meta", default=os.path.join(HERE, "outputs/trial_metadata.csv"))
    ap.add_argument("--alpha", type=float, default=10000.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default=os.path.join(HERE, "outputs/fidelity_results.csv"))
    ap.add_argument("--out-pred", default=os.path.join(HERE, "outputs/predicted_embeddings.npy"))
    ap.add_argument("--out-meta", default=os.path.join(HERE, "outputs/ridge_cv_metadata.json"))
    ap.add_argument("--out-summary", default=os.path.join(HERE, "outputs/ridge_smoke_summary.txt"))
    args = ap.parse_args()

    for p in (args.x, args.y, args.meta):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: required input not found: {p}")

    X = np.load(args.x).astype(np.float64)
    Y = np.load(args.y).astype(np.float64)
    meta = pd.read_csv(args.meta)

    with open(args.out_summary, "w") as fh:
        log = Tee(fh)
        log("=" * 72)
        log("RIDGE SMOKE TEST — single subject LTP269/ses-20 (NOT final analysis)")
        log("=" * 72)
        log(f"X_eeg: {X.shape}   Y_t5: {Y.shape}   trials: {len(meta)}")
        log(f"alpha={args.alpha}  folds={args.folds}  seed={args.seed}  "
            f"solver=svd (n<<p)")

        if not (X.shape[0] == Y.shape[0] == len(meta) == N_TRIALS):
            sys.exit("ERROR: row counts disagree or != 576.")

        subj = set(meta.subject.astype(str).unique())
        if subj != {"LTP269"}:
            log(f"WARNING: expected only subject LTP269, found {subj}")

        # -----------------------------------------------------------------
        # Out-of-fold prediction
        # -----------------------------------------------------------------
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        pred = np.full((N_TRIALS, Y_DIM), np.nan, dtype=np.float64)
        predicted_count = np.zeros(N_TRIALS, dtype=int)
        fold_assign = np.full(N_TRIALS, -1, dtype=int)

        for fold, (tr, te) in enumerate(kf.split(np.arange(N_TRIALS))):
            # standardize on TRAIN ONLY
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            Xte = scaler.transform(X[te])
            model = Ridge(alpha=args.alpha, solver="svd")
            model.fit(Xtr, Y[tr])
            pred[te] = model.predict(Xte)
            predicted_count[te] += 1
            fold_assign[te] = fold
            log(f"  fold {fold}: train={len(tr)} test={len(te)}")

        # -----------------------------------------------------------------
        # Fidelity = cosine(pred, true) per held-out trial
        # -----------------------------------------------------------------
        fidelity = row_cosine(pred, Y)

        res = pd.DataFrame({
            "subject": meta.subject,
            "session": meta.session,
            "trial": meta.trial,
            "serialpos": meta.serialpos,
            "word": meta.word,
            "recalled": meta.recalled,
            "embedding_fidelity": fidelity,
        })

        # -----------------------------------------------------------------
        # Validation
        # -----------------------------------------------------------------
        log("\n=== VALIDATION ===")
        log.check("fidelity_results has 576 rows", len(res) == N_TRIALS, f"{len(res)}")
        log.check("predicted_embeddings shape 576 x 1024", pred.shape == (N_TRIALS, Y_DIM),
                  f"{pred.shape}")
        log.check("every trial has exactly one held-out prediction",
                  bool((predicted_count == 1).all()),
                  f"counts unique={sorted(set(predicted_count.tolist()))}")
        log.check("predictions have no NaN/Inf",
                  not np.isnan(pred).any() and not np.isinf(pred).any())
        log.check("fidelity has no NaN/Inf",
                  not np.isnan(fidelity).any() and not np.isinf(fidelity).any())

        # -----------------------------------------------------------------
        # Remembered vs forgotten
        # -----------------------------------------------------------------
        rec = res[res.recalled == 1].embedding_fidelity
        forg = res[res.recalled == 0].embedding_fidelity
        mean_rec = float(rec.mean())
        mean_forg = float(forg.mean())
        diff = mean_rec - mean_forg

        log("\n=== FIDELITY (cosine sim of held-out predictions) ===")
        log(f"overall mean fidelity     : {fidelity.mean():.4f}  "
            f"(sd {fidelity.std():.4f}, min {fidelity.min():.4f}, max {fidelity.max():.4f})")
        log(f"recalled  (n={len(rec)})   mean fidelity: {mean_rec:.4f}")
        log(f"forgotten (n={len(forg)})  mean fidelity: {mean_forg:.4f}")
        log(f"remembered - forgotten     : {diff:+.4f}")

        # -----------------------------------------------------------------
        # Save
        # -----------------------------------------------------------------
        res.to_csv(args.out_csv, index=False)
        np.save(args.out_pred, pred.astype(np.float32))

        meta_json = {
            "kind": "single-subject ridge smoke test (NOT final analysis)",
            "subject": "LTP269",
            "session": 20,
            "task": "ltpFR2",
            "n_trials": N_TRIALS,
            "x_shape": list(X.shape),
            "y_shape": list(Y.shape),
            "alpha": args.alpha,
            "n_folds": args.folds,
            "kfold_shuffle": True,
            "seed": args.seed,
            "ridge_solver": "svd",
            "standardize": "StandardScaler fit on train fold only",
            "cv": "held-out-trial KFold; train/test disjoint",
            "fidelity_metric": "cosine similarity(predicted, true) per held-out trial",
            "mean_fidelity_overall": float(fidelity.mean()),
            "mean_fidelity_recalled": mean_rec,
            "mean_fidelity_forgotten": mean_forg,
            "remembered_minus_forgotten": diff,
            "n_recalled": int(len(rec)),
            "n_forgotten": int(len(forg)),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "disclaimer": "Smoke test only. Not final inference. No mixed-effects "
                          "model. No significance testing performed here.",
        }
        json.dump(meta_json, open(args.out_meta, "w"), indent=2)

        log("\nwrote:")
        log(f"  {os.path.relpath(args.out_csv, HERE)}")
        log(f"  {os.path.relpath(args.out_pred, HERE)}")
        log(f"  {os.path.relpath(args.out_meta, HERE)}")
        log(f"  {os.path.relpath(args.out_summary, HERE)}")
        log("\n*** SMOKE TEST ONLY — single subject, not final inference. "
            "Do NOT interpret as the mixed-effects result. ***")
        log("STATUS: " + ("OK" if log.fail == 0 else f"FAILED ({log.fail} checks)"))

    sys.exit(0 if log.fail == 0 else 1)


if __name__ == "__main__":
    main()
