#!/usr/bin/env python3
"""
Audit outputs/X_eeg.npy against the original EDF slices for the canonical
session (sub-LTP269 / ses-20 / task-ltpFR2).

For rows 0, 100, 300, 575 it independently re-reads the EDF with MNE, re-picks
the same 129 EEG channels, re-computes the sample-based window, extracts the
raw slice, flattens channel-major (C-order), and compares to X_eeg[row].

Window (sfreq=500, 0.300-0.800 s):
    start_sample = sample + int(0.300*sfreq)   # +150
    stop_sample  = sample + int(0.800*sfreq)   # +400  (exclusive)  -> 250 tp

Report -> outputs/eeg_extraction_audit.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRIALS = 576
N_CH = 129
N_TP = 250
N_FEAT = N_CH * N_TP  # 32250
AUDIT_ROWS = [0, 100, 300, 575]
WIN_START = 0.300
WIN_STOP = 0.800


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x", default=os.path.join(HERE, "outputs/X_eeg.npy"))
    ap.add_argument("--meta", default=os.path.join(HERE, "outputs/trial_metadata.csv"))
    ap.add_argument("--trials", default=os.path.join(HERE, "outputs/encoding_trials.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs/eeg_extraction_audit.txt"))
    args = ap.parse_args()

    import mne

    for p in (args.x, args.meta, args.trials):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: required input not found: {p}")

    X = np.load(args.x)
    meta = pd.read_csv(args.meta)
    trials = pd.read_csv(args.trials)

    # canonical EDF from the eeg_file column
    eeg_files = trials.eeg_file.unique()
    if len(eeg_files) != 1:
        sys.exit(f"ERROR: expected one eeg_file, found {len(eeg_files)}: {eeg_files}")
    eeg_rel = eeg_files[0]
    eeg_path = eeg_rel if os.path.isabs(eeg_rel) else os.path.join(HERE, eeg_rel)
    if not os.path.isfile(eeg_path):
        sys.exit(f"ERROR: EDF not found: {eeg_path}")

    with open(args.out, "w") as fh:
        log = Tee(fh)
        log("=" * 74)
        log("EEG EXTRACTION AUDIT — sub-LTP269 / ses-20 / task-ltpFR2")
        log("=" * 74)
        log(f"X_eeg          : {args.x}  shape={X.shape} dtype={X.dtype}")
        log(f"trial_metadata : {args.meta}  rows={len(meta)}")
        log(f"EDF            : {eeg_rel}")

        log(f"\nloading EDF with MNE ...")
        raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        n_times_total = int(raw.n_times)
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
        n_channels = len(eeg_picks)
        data = raw.get_data(picks=eeg_picks)  # (129, n_times) float64
        start_off = int(WIN_START * sfreq)
        stop_off = int(WIN_STOP * sfreq)
        log(f"sfreq={sfreq}  n_times={n_times_total}  EEG channels picked={n_channels}")
        log(f"window offsets: +{start_off} .. +{stop_off} (exclusive) "
            f"-> {stop_off - start_off} timepoints")

        log.check("picked 129 EEG channels", n_channels == N_CH, f"got {n_channels}")

        # =================================================================
        # Spot-check specific rows against freshly-extracted slices
        # =================================================================
        log("\n" + "-" * 74)
        log("PER-ROW SLICE COMPARISON (rows 0, 100, 300, 575)")
        log("-" * 74)
        worst = 0.0
        for r in AUDIT_ROWS:
            row = meta.iloc[r]
            s = int(row["sample"])
            start = s + start_off
            stop = s + stop_off
            seg = data[:, start:stop]  # (129, 250) float64
            shape_ok = seg.shape == (N_CH, N_TP)
            flat = seg.reshape(-1)  # C-order, channel-major

            stored = X[r].astype(np.float64)
            # Raw (stored float32 vs recomputed float64) and exact-after-cast diff.
            max_abs = float(np.max(np.abs(flat - stored)))
            mean_abs = float(np.mean(np.abs(flat - stored)))
            exact_after_cast = bool(np.array_equal(seg.reshape(-1).astype(np.float32),
                                                   X[r]))
            worst = max(worst, max_abs)

            log(f"\nrow {r:>3}: word={row.get('word')!r} trial={int(row['trial'])} "
                f"serialpos={int(row['serialpos'])} sample={s}")
            log(f"   start_sample={start}  stop_sample={stop}  "
                f"reextracted shape={seg.shape}")
            log(f"   max|reextracted - X_eeg[{r}]| = {max_abs:.3e}")
            log(f"   mean|reextracted - X_eeg[{r}]| = {mean_abs:.3e}")
            log(f"   exact match after float32 cast: {exact_after_cast}")
            log.check(f"row {r}: reextracted shape is 129x250", shape_ok, f"{seg.shape}")
            log.check(f"row {r}: matches X_eeg (float32-exact)", exact_after_cast,
                      f"max abs diff {max_abs:.3e}")

        log(f"\nworst max-abs diff across audited rows (float64 vs stored float32): "
            f"{worst:.3e}")

        # =================================================================
        # Full-matrix validation
        # =================================================================
        log("\n" + "=" * 74)
        log("FULL VALIDATION (all 576 rows)")
        log("=" * 74)
        log.check("X_eeg shape is 576 x 32250", X.shape == (N_TRIALS, N_FEAT), f"{X.shape}")

        if "n_timepoints" in meta.columns:
            log.check("every row has n_timepoints = 250",
                      bool((meta.n_timepoints == N_TP).all()),
                      f"unique={sorted(meta.n_timepoints.unique().tolist())}")
        if "extracted" in meta.columns:
            log.check("every row has extracted = True",
                      bool(meta.extracted.astype(bool).all()),
                      f"{int(meta.extracted.astype(bool).sum())}/{len(meta)}")
        if "drop_reason" in meta.columns:
            dr = meta.drop_reason.fillna("").astype(str).str.strip()
            n_dropped = int((dr != "").sum())
            log.check("drop_reason empty/none for all rows", n_dropped == 0,
                      f"{n_dropped} rows with a drop_reason")

        log.check("no NaN values", not np.isnan(X).any())
        log.check("no infinite values", not np.isinf(X).any())
        row_all_zero = (X == 0).all(axis=1)
        log.check("no all-zero rows", not row_all_zero.any(),
                  f"{int(row_all_zero.sum())} all-zero rows")

        n_dropped_total = 0
        if "extracted" in meta.columns:
            n_dropped_total = int((~meta.extracted.astype(bool)).sum())

        log("\n--- summary ---")
        log(f"X_eeg shape          : {X.shape}")
        log(f"dropped trials       : {n_dropped_total}")
        log(f"valid rows           : {N_TRIALS - n_dropped_total}/{N_TRIALS}")
        log(f"worst per-row max diff: {worst:.3e} (float32 storage rounding)")

        log("")
        if log.fail == 0:
            log("SUCCESS: EEG extraction audit passed")
        else:
            log(f"FAILURE: EEG extraction audit failed ({log.fail} checks)")

    sys.exit(0 if log.fail == 0 else 1)


if __name__ == "__main__":
    main()
