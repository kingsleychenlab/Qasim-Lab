#!/usr/bin/env python3
"""
Stage B: extract raw 300-800 ms EEG windows for each encoding trial.

For every row of outputs/encoding_trials.csv, pull the raw EEG segment
[sample + int(win_start*sfreq), sample + int(win_stop*sfreq)) from the EDF and
flatten it (channels x timepoints) into one feature vector. The first pass keeps
the raw 500 Hz data, with no filtering, baseline correction, or resampling
unless the corresponding flag is passed.

Window / indexing (sfreq = 500 Hz, defaults 0.300-0.800 s):
    start_sample = sample + int(0.300 * sfreq)   # = sample + 150
    stop_sample  = sample + int(0.800 * sfreq)   # = sample + 400
    segment      = data[:, start_sample:stop_sample]   # stop exclusive (Python slice)
    -> timepoints = 400 - 150 = 250  (stop is exclusive, so 250 not 251)

Flattening: numpy C-order reshape of a (channels, timepoints) block, i.e.
    [ch0_t0, ch0_t1, ..., ch0_t249, ch1_t0, ...]  (channel-major).

Outputs:
    outputs/X_eeg.npy                 (576 x 32250, float32)
    outputs/trial_metadata.csv        (encoding_trials + start/stop/n_timepoints)
    outputs/eeg_feature_metadata.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", default=os.path.join(HERE, "outputs/encoding_trials.csv"))
    ap.add_argument("--out-x", default=os.path.join(HERE, "outputs/X_eeg.npy"))
    ap.add_argument("--out-meta-csv", default=os.path.join(HERE, "outputs/trial_metadata.csv"))
    ap.add_argument("--out-meta-json", default=os.path.join(HERE, "outputs/eeg_feature_metadata.json"))
    ap.add_argument("--win-start", type=float, default=0.300)
    ap.add_argument("--win-stop", type=float, default=0.800)
    # Preprocessing is off by default (the first pass keeps raw 500 Hz data).
    ap.add_argument("--filter", nargs=2, type=float, metavar=("LFREQ", "HFREQ"),
                    default=None, help="Optional band-pass (l_freq h_freq). Default: none.")
    ap.add_argument("--baseline", nargs=2, type=float, metavar=("BSTART", "BSTOP"),
                    default=None, help="Optional baseline window (s rel. to onset). Default: none.")
    ap.add_argument("--resample", type=float, default=None,
                    help="Optional resample rate (Hz). Default: none (keep 500 Hz).")
    args = ap.parse_args()

    import mne

    trials = pd.read_csv(args.trials)
    print(f"encoding_trials: {len(trials)} rows")

    eeg_files = trials.eeg_file.unique()
    if len(eeg_files) != 1:
        sys.exit(f"ERROR: expected a single EEG file, found {len(eeg_files)}: {eeg_files}")
    eeg_rel = eeg_files[0]
    eeg_path = eeg_rel if os.path.isabs(eeg_rel) else os.path.join(HERE, eeg_rel)
    if not os.path.isfile(eeg_path):
        sys.exit(f"ERROR: EEG file not found: {eeg_path}")

    # -----------------------------------------------------------------
    # Load EDF (preload into memory so slicing is exact & fast)
    # -----------------------------------------------------------------
    print(f"loading EDF: {eeg_path}")
    raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_times_total = int(raw.n_times)
    print(f"sfreq={sfreq} n_times={n_times_total} n_ch(all)={len(raw.ch_names)}")

    # Optional preprocessing (all off unless a flag is passed)
    applied = {"filter": None, "baseline": None, "resample": None}
    if args.filter is not None:
        l, h = args.filter
        print(f"applying band-pass {l}-{h} Hz")
        raw.filter(l_freq=l, h_freq=h, verbose="ERROR")
        applied["filter"] = [l, h]
    if args.resample is not None:
        print(f"resampling to {args.resample} Hz (adjusts sample indices!)")
        # Resampling changes the sample grid; recompute sample indices accordingly.
        orig_sfreq = sfreq
        raw.resample(args.resample, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        n_times_total = int(raw.n_times)
        trials = trials.copy()
        trials["sample"] = (trials["sample"] * (sfreq / orig_sfreq)).round().astype(int)
        applied["resample"] = args.resample

    # Pick EEG channels only.
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    n_channels = len(eeg_picks)
    print(f"picked EEG channels: {n_channels}")

    data = raw.get_data(picks=eeg_picks)  # (n_channels, n_times), volts, float64

    win_start_off = int(args.win_start * sfreq)
    win_stop_off = int(args.win_stop * sfreq)
    n_timepoints = win_stop_off - win_start_off
    n_features = n_channels * n_timepoints
    print(f"window offsets: start=+{win_start_off}, stop=+{win_stop_off} "
          f"(stop EXCLUSIVE) -> {n_timepoints} timepoints")
    print(f"feature dim = {n_channels} ch x {n_timepoints} tp = {n_features}")

    # -----------------------------------------------------------------
    # Extract per-trial (sample-based, integer indexing, no float onsets)
    # -----------------------------------------------------------------
    X = np.zeros((len(trials), n_features), dtype=np.float32)
    meta_rows = []
    dropped = []
    for i, row in trials.reset_index(drop=True).iterrows():
        s = int(row["sample"])
        start = s + win_start_off
        stop = s + win_stop_off
        reason = ""
        if s < 0:
            reason = f"sentinel sample {s} < 0"
        elif start < 0:
            reason = f"start_sample {start} < 0"
        elif stop > n_times_total:
            reason = f"stop_sample {stop} > n_times {n_times_total}"

        if reason:
            dropped.append((i, row.get("word"), reason))
        else:
            seg = data[:, start:stop]  # (n_channels, n_timepoints)
            if seg.shape[1] != n_timepoints:
                reason = f"segment length {seg.shape[1]} != {n_timepoints}"
                dropped.append((i, row.get("word"), reason))
            else:
                # C-order flatten: channel-major (ch0 all tp, then ch1, ...)
                X[i] = seg.reshape(-1).astype(np.float32)

        m = row.to_dict()
        m.update({"start_sample": start, "stop_sample": stop,
                  "n_timepoints": n_timepoints,
                  "extracted": reason == "", "drop_reason": reason})
        meta_rows.append(m)

    meta_df = pd.DataFrame(meta_rows)

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"trials total   : {len(trials)}")
    print(f"trials extracted: {int(meta_df.extracted.sum())}")
    print(f"trials dropped : {len(dropped)}")
    if dropped:
        print("DROPPED (index, word, reason):")
        for d in dropped:
            print(f"   {d[0]}  {d[1]!r}  {d[2]}")
    else:
        print("no trials dropped.")
    print(f"X_eeg shape    : {X.shape}")

    # Basic integrity
    has_nan = bool(np.isnan(X).any())
    has_inf = bool(np.isinf(X).any())
    print(f"NaN: {has_nan}   Inf: {has_inf}")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    np.save(args.out_x, X)
    meta_df.to_csv(args.out_meta_csv, index=False)

    feat_meta = {
        "description": "Raw EEG 300-800 ms windows per encoding trial, "
                       "flattened channels x timepoints (C-order, channel-major).",
        "source_trials": os.path.relpath(args.trials, HERE),
        "eeg_file": eeg_rel,
        "sfreq": sfreq,
        "n_times_total": n_times_total,
        "n_channels": n_channels,
        "channel_names": ch_names,
        "win_start_s": args.win_start,
        "win_stop_s": args.win_stop,
        "win_start_offset_samples": win_start_off,
        "win_stop_offset_samples": win_stop_off,
        "stop_exclusive": True,
        "n_timepoints": n_timepoints,
        "n_features": n_features,
        "X_shape": list(X.shape),
        "dtype": str(X.dtype),
        "flatten_order": "C (channel-major: [ch0_t0..ch0_tN, ch1_t0..])",
        "units": "volts (raw EDF scaling from MNE)",
        "preprocessing_applied": applied,
        "raw_500hz_first_pass": applied == {"filter": None, "baseline": None, "resample": None},
        "n_trials_total": int(len(trials)),
        "n_trials_extracted": int(meta_df.extracted.sum()),
        "n_trials_dropped": len(dropped),
        "dropped": [{"index": int(d[0]), "word": d[1], "reason": d[2]} for d in dropped],
        "no_nan": not has_nan,
        "no_inf": not has_inf,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(feat_meta, open(args.out_meta_json, "w"), indent=2)

    print(f"\nwrote {args.out_x} {X.shape}")
    print(f"wrote {args.out_meta_csv}")
    print(f"wrote {args.out_meta_json}")
    if dropped:
        print("STATUS: completed WITH dropped trials (see above).")
        sys.exit(2)
    print("STATUS: OK (all 576 trials extracted, raw 500 Hz).")


if __name__ == "__main__":
    main()
