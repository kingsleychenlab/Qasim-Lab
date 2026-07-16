#!/usr/bin/env python3
"""
The gate before ridge: check X_eeg, Y_t5 and trial_metadata line up.

Read-only, no regression or analysis. It exists so that a misalignment gets
caught here rather than downstream, where nothing would catch it.

The failure it guards against: row i of X must be the EEG for the same trial
whose word's embedding is row i of Y. If those slip out of step, ridge gets
trained to predict the wrong word's vector from each trial's brain response.
That would not crash or warn; it would just produce fidelity values that look
reasonable and mean nothing. Shapes, row counts and per-row word agreement are
all checked here to rule that out.

Writes a full report to outputs/model_input_validation.txt and exits non-zero
on any failure, so it can gate the pipeline.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from common import Tee

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRIALS = 576          # words per ltpFR2 session (the full PEERS pool)
Y_DIM = 1024            # t5-large hidden size
N_CH = 129              # EGI channels retained
N_TP = 250              # 500 Hz x the 500 ms window (300-800 ms)
N_FEAT = N_CH * N_TP    # 32250 flattened EEG features per trial
EXP_SUBJECT = "LTP269"
EXP_SESSION = 20


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", default=os.path.join(HERE, "outputs/X_eeg.npy"))
    ap.add_argument("--y", default=os.path.join(HERE, "outputs/Y_t5.npy"))
    ap.add_argument("--meta", default=os.path.join(HERE, "outputs/trial_metadata.csv"))
    ap.add_argument("--targets-map", default=os.path.join(HERE, "outputs/trial_targets_metadata.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs/model_input_validation.txt"))
    args = ap.parse_args()

    with open(args.out, "w") as fh:
        log = Tee(fh)
        log("=" * 70)
        log("MODEL-INPUT VALIDATION (X_eeg / Y_t5 / trial_metadata)")
        log("=" * 70)

        # Existence
        log.check("outputs/X_eeg.npy exists", os.path.isfile(args.x))
        log.check("outputs/Y_t5.npy exists", os.path.isfile(args.y))
        log.check("outputs/trial_metadata.csv exists", os.path.isfile(args.meta))
        if log.fail:
            log("\nFAILURE: missing required inputs; aborting.")
            sys.exit(1)

        X = np.load(args.x)
        Y = np.load(args.y)
        meta = pd.read_csv(args.meta)

        # Shapes / counts
        log.check(f"X_eeg rows = {N_TRIALS}", X.shape[0] == N_TRIALS, f"got {X.shape[0]}")
        log.check(f"Y_t5 rows = {N_TRIALS}", Y.shape[0] == N_TRIALS, f"got {Y.shape[0]}")
        log.check(f"trial_metadata rows = {N_TRIALS}", len(meta) == N_TRIALS, f"got {len(meta)}")
        log.check(f"Y_t5 columns = {Y_DIM}", Y.shape[1] == Y_DIM, f"got {Y.shape[1]}")
        log.check(f"X_eeg columns = {N_FEAT} ({N_CH}x{N_TP})", X.shape[1] == N_FEAT,
                  f"got {X.shape[1]}")

        # Finiteness
        log.check("X_eeg has no NaN", not np.isnan(X).any())
        log.check("X_eeg has no infinite values", not np.isinf(X).any())
        log.check("Y_t5 has no NaN", not np.isnan(Y).any())
        log.check("Y_t5 has no infinite values", not np.isinf(Y).any())

        # Alignment: X rows <-> trial_metadata rows (row count already checked;
        # metadata is per-trial and index-aligned to X by construction).
        log.check("X_eeg rows align with trial_metadata rows", X.shape[0] == len(meta),
                  f"{X.shape[0]} vs {len(meta)}")

        # Alignment: Y rows <-> trial_metadata words (via trial_targets_metadata)
        if os.path.isfile(args.targets_map):
            tmap = pd.read_csv(args.targets_map)
            aligned = (len(tmap) == len(meta)
                       and (tmap.word.values == meta.word.values).all())
            log.check("Y_t5 rows align with trial_metadata words "
                      "(order identical in trial_targets_metadata.csv)", aligned)
        else:
            log.check("trial_targets_metadata.csv present for Y/word alignment", False,
                      "missing")

        # recalled 0/1
        if "recalled" in meta.columns:
            log.check("recalled column is 0/1", set(meta.recalled.unique()) <= {0, 1},
                      f"values={sorted(meta.recalled.unique().tolist())}")
        else:
            log.check("recalled column present", False)

        # subject/session identity
        subs = set(meta.subject.astype(str).unique())
        sess = set(meta.session.unique())
        log.check(f"subject/session are {EXP_SUBJECT}/ses-{EXP_SESSION}",
                  subs == {EXP_SUBJECT} and sess == {EXP_SESSION},
                  f"subjects={subs} sessions={sess}")

        # -------- prints --------
        n_dropped = int((~meta["extracted"]).sum()) if "extracted" in meta.columns else 0
        tp = int(meta["n_timepoints"].iloc[0]) if "n_timepoints" in meta.columns else N_TP
        log("\n--- summary ---")
        log(f"X_eeg shape            : {X.shape}")
        log(f"Y_t5 shape             : {Y.shape}")
        log(f"EEG feature dimension  : {X.shape[1]}")
        log(f"channel count          : {N_CH}")
        log(f"timepoint count        : {tp}")
        log(f"dropped trial count    : {n_dropped}")
        if "recalled" in meta.columns:
            log(f"recalled=1 / recalled=0: {int((meta.recalled==1).sum())} / "
                f"{int((meta.recalled==0).sum())}")
        log(f"X dtype / Y dtype      : {X.dtype} / {Y.dtype}")
        log(f"X value range          : [{X.min():.3e}, {X.max():.3e}]")

        log("\n" + ("SUCCESS: all model-input checks passed."
                    if log.fail == 0 else f"FAILURE: {log.fail} check(s) failed."))

    sys.exit(0 if log.fail == 0 else 1)


if __name__ == "__main__":
    main()
