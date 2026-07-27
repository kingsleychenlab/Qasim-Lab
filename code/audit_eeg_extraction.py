#!/usr/bin/env python3
"""
Re-cut EEG windows straight from the EDF and check X_eeg matches.

This catches something nothing else can: an X_eeg that has the right shape, no
NaNs, but was cut from the wrong samples. A window off by even a few samples
still gives a valid 576 x 32250 matrix, and ridge would fit it fine. The
fidelity numbers would just describe the wrong slice of brain activity.

So I don't trust any of step07's own bookkeeping here. This re-reads the EDF
with MNE, re-picks the same 129 channels, recomputes the window from the event
sample, and compares the resulting bytes to the stored row. Only a few rows (0,
100, 300, 575) get checked. An indexing or flattening error is systematic and
would show up on any of them, and re-reading a ~500 MB EDF is the slow part.

Window arithmetic being verified (sfreq=500, 0.300-0.800 s):
    start_sample = sample + int(0.300*sfreq)   # +150
    stop_sample  = sample + int(0.800*sfreq)   # +400, exclusive -> 250 timepoints

It also re-checks the flattening order. step08 only sees a flat 32250-vector, so
channel-major (C-order) vs time-major is invisible downstream, but it silently
permutes every feature.

Report -> outputs/eeg_extraction_audit.txt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from common import Tee

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRIALS = 576
N_CH = 129
N_TP = 250
N_FEAT = N_CH * N_TP  # 32250
AUDIT_ROWS = [0, 100, 300, 575]
WIN_START = 0.300
WIN_STOP = 0.800


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", default=os.path.join(HERE, "outputs/X_eeg.npy"))
    parser.add_argument("--meta", default=os.path.join(HERE, "outputs/trial_metadata.csv"))
    parser.add_argument("--trials", default=os.path.join(HERE, "outputs/encoding_trials.csv"))
    parser.add_argument("--out", default=os.path.join(HERE, "outputs/eeg_extraction_audit.txt"))
    args = parser.parse_args()

    import mne

    for path in (args.x, args.meta, args.trials):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: required input not found: {path}")

    X = np.load(args.x)
    meta = pd.read_csv(args.meta)
    trials = pd.read_csv(args.trials)

    # canonical EDF from the eeg_file column
    eeg_files = trials.eeg_file.unique()
    if len(eeg_files) != 1:
        sys.exit(f"ERROR: expected one eeg_file, found {len(eeg_files)}: {eeg_files}")
    eeg_file_rel = eeg_files[0]
    eeg_file_path = eeg_file_rel if os.path.isabs(eeg_file_rel) else os.path.join(HERE, eeg_file_rel)
    if not os.path.isfile(eeg_file_path):
        sys.exit(f"ERROR: EDF not found: {eeg_file_path}")

    with open(args.out, "w") as fh:
        log = Tee(fh)
        log("=" * 74)
        log("EEG EXTRACTION AUDIT — sub-LTP269 / ses-20 / task-ltpFR2")
        log("=" * 74)
        log(f"X_eeg          : {args.x}  shape={X.shape} dtype={X.dtype}")
        log(f"trial_metadata : {args.meta}  rows={len(meta)}")
        log(f"EDF            : {eeg_file_rel}")

        log("\nloading EDF with MNE ...")
        raw = mne.io.read_raw_edf(eeg_file_path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        n_times_total = int(raw.n_times)
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
        n_channels = len(eeg_picks)
        data = raw.get_data(picks=eeg_picks)  # (129, n_times) float64
        start_offset = int(WIN_START * sfreq)
        stop_offset = int(WIN_STOP * sfreq)
        log(f"sfreq={sfreq}  n_times={n_times_total}  EEG channels picked={n_channels}")
        log(f"window offsets: +{start_offset} .. +{stop_offset} (exclusive) "
            f"-> {stop_offset - start_offset} timepoints")

        log.check("picked 129 EEG channels", n_channels == N_CH, f"got {n_channels}")

        # Spot-check specific rows against freshly-extracted slices.
        log("\n" + "-" * 74)
        log("PER-ROW SLICE COMPARISON (rows 0, 100, 300, 575)")
        log("-" * 74)
        worst = 0.0
        for row_idx in AUDIT_ROWS:
            meta_row = meta.iloc[row_idx]
            sample = int(meta_row["sample"])
            start_sample = sample + start_offset
            stop_sample = sample + stop_offset
            segment = data[:, start_sample:stop_sample]  # (129, 250) float64
            shape_matches = segment.shape == (N_CH, N_TP)
            flat_segment = segment.reshape(-1)  # C-order, channel-major

            stored_row = X[row_idx].astype(np.float64)
            # Raw (stored float32 vs recomputed float64) and exact-after-cast diff.
            max_abs = float(np.max(np.abs(flat_segment - stored_row)))
            mean_abs = float(np.mean(np.abs(flat_segment - stored_row)))
            exact_after_cast = bool(np.array_equal(segment.reshape(-1).astype(np.float32),
                                                   X[row_idx]))
            worst = max(worst, max_abs)

            log(f"\nrow {row_idx:>3}: word={meta_row.get('word')!r} trial={int(meta_row['trial'])} "
                f"serialpos={int(meta_row['serialpos'])} sample={sample}")
            log(f"   start_sample={start_sample}  stop_sample={stop_sample}  "
                f"reextracted shape={segment.shape}")
            log(f"   max|reextracted - X_eeg[{row_idx}]| = {max_abs:.3e}")
            log(f"   mean|reextracted - X_eeg[{row_idx}]| = {mean_abs:.3e}")
            log(f"   exact match after float32 cast: {exact_after_cast}")
            log.check(f"row {row_idx}: reextracted shape is 129x250", shape_matches, f"{segment.shape}")
            log.check(f"row {row_idx}: matches X_eeg (float32-exact)", exact_after_cast,
                      f"max abs diff {max_abs:.3e}")

        log(f"\nworst max-abs diff across audited rows (float64 vs stored float32): "
            f"{worst:.3e}")

        log("\n" + "=" * 74)
        log("FULL VALIDATION (all 576 rows)")
        log("=" * 74)
        log.check("X_eeg shape is 576 x 32250", X.shape == (N_TRIALS, N_FEAT), f"{X.shape}")

        if "n_timepoints" in meta.columns:
            log.check("every row has n_timepoints = 250",
                      bool((meta.n_timepoints == N_TP).all()),
                      f"unique={sorted(meta.n_timepoints.unique().tolist())}")
        else:
            log.warn("n_timepoints column absent", "cannot verify per-row timepoints")
        if "extracted" in meta.columns:
            log.check("every row has extracted = True",
                      bool(meta.extracted.astype(bool).all()),
                      f"{int(meta.extracted.astype(bool).sum())}/{len(meta)}")
        else:
            log.warn("extracted column absent", "cannot verify all rows extracted")
        if "drop_reason" in meta.columns:
            drop_reasons = meta.drop_reason.fillna("").astype(str).str.strip()
            n_dropped = int((drop_reasons != "").sum())
            log.check("drop_reason empty/none for all rows", n_dropped == 0,
                      f"{n_dropped} rows with a drop_reason")
        else:
            log.warn("drop_reason column absent", "cannot verify no drops")

        log.check("no NaN values", not np.isnan(X).any())
        log.check("no infinite values", not np.isinf(X).any())
        all_zero_rows = (X == 0).all(axis=1)
        log.check("no all-zero rows", not all_zero_rows.any(),
                  f"{int(all_zero_rows.sum())} all-zero rows")

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
