#!/usr/bin/env python3
"""
Validate the canonical encoding_trials.csv before anything downstream trusts it.

Read-only: no EEG extraction, no X_eeg.npy / Y_t5.npy, no training. Prints to
stdout only and exits non-zero on any failure, so it can gate the pipeline.

Checks the session that the EXPECTED_* constants below pin (sub-LTP269 /
ses-20 / task-ltpFR2 -- the session with 576 words, full PEERS coverage and
576/576 valid 300-800 ms windows):
  - 576 rows
  - every word exists in peers_word_order.csv (100% coverage)
  - recalled is only 0 / 1
  - no missing onset / sample
  - no sentinel values (onset <= 0 or sample < 0)
  - all eeg_file paths exist on disk
"""

import argparse
import os
import sys

import pandas as pd

from common import peers_word_set

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_ROWS = 576
EXPECTED_SUBJECT = "LTP269"
EXPECTED_SESSION = 20
EXPECTED_TASK = "task-ltpFR2"
EXPECTED_EEG = "sub-LTP269/ses-20/eeg/sub-LTP269_ses-20_task-ltpFR2_eeg.edf"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=os.path.join(HERE, "outputs", "encoding_trials.csv"))
    ap.add_argument("--peers-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    ap.add_argument("--win-start", type=float, default=0.300)
    ap.add_argument("--win-stop", type=float, default=0.800)
    args = ap.parse_args()

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        if not cond:
            ok = False
        print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    df = pd.read_csv(args.csv)
    peers = peers_word_set(args.peers_order)

    print(f"csv   : {args.csv}")
    print(f"rows  : {len(df)}   columns: {list(df.columns)}\n")

    # 1. row count
    check(f"encoding_trials.csv has {EXPECTED_ROWS} rows", len(df) == EXPECTED_ROWS,
          f"got {len(df)}")

    # 2. word coverage
    in_peers = df.word.str.upper().isin(peers)
    missing = sorted(set(df.word.str.upper()[~in_peers]))
    check("all words exist in peers_word_order.csv",
          bool(in_peers.all()),
          f"{int(in_peers.sum())}/{len(df)} in peers; missing={missing[:10]}")

    # 3. recalled only 0/1
    check("recalled is only 0 or 1", set(df.recalled.unique()) <= {0, 1},
          f"values={sorted(df.recalled.unique().tolist())}")

    # 4. no missing onset/sample
    check("no missing onset", bool(df.onset.notna().all()),
          f"{int(df.onset.isna().sum())} missing")
    check("no missing sample", bool(df["sample"].notna().all()),
          f"{int(df['sample'].isna().sum())} missing")

    # 5. no sentinel values (onset <= 0 or sample < 0)
    bad_onset = int((df.onset <= 0).sum())
    bad_sample = int((df["sample"] < 0).sum())
    check("no sentinel onset (all onset > 0)", bad_onset == 0, f"{bad_onset} rows onset<=0")
    check("no sentinel sample (all sample >= 0)", bad_sample == 0, f"{bad_sample} rows sample<0")

    # 6. all eeg_file paths exist
    missing_paths = []
    for eeg_file in df.eeg_file.unique():
        full_path = eeg_file if os.path.isabs(eeg_file) else os.path.join(HERE, eeg_file)
        if not os.path.isfile(full_path):
            missing_paths.append(eeg_file)
    check("all eeg_file paths exist on disk", not missing_paths,
          f"missing: {missing_paths}")

    # 6b. window fit: the full [win_start, win_stop] window must sit inside the
    #     recording for every row, checked directly via MNE. Also re-checks no
    #     sentinels.
    if not missing_paths:
        import mne
        eeg_path = df.eeg_file.iloc[0]
        eeg_full = eeg_path if os.path.isabs(eeg_path) else os.path.join(HERE, eeg_path)
        raw = mne.io.read_raw_edf(eeg_full, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        start_s = df["sample"] + int(args.win_start * sfreq)
        stop_s = df["sample"] + int(args.win_stop * sfreq)
        win_ok = (df["sample"] >= 0) & (df.onset > 0) & (start_s >= 0) & (stop_s < n_times)
        check(f"all 576 EEG windows [{int(args.win_start*1000)}-"
              f"{int(args.win_stop*1000)}ms] fit inside recording "
              f"(n_times={n_times}, sfreq={sfreq:g})",
              bool(win_ok.all()),
              f"{int(win_ok.sum())}/{len(df)} fit")
    else:
        check("EEG window-fit check", False, "eeg file missing; cannot verify")

    # 7. session identity
    subs = set(df.subject.astype(str).unique())
    sess = set(df.session.unique())
    tasks_ok = df.eeg_file.str.contains(EXPECTED_TASK).all() and \
        df.event_file.str.contains(EXPECTED_TASK).all()
    check(f"session is {EXPECTED_SUBJECT}/ses-{EXPECTED_SESSION}/{EXPECTED_TASK}",
          subs == {EXPECTED_SUBJECT} and sess == {EXPECTED_SESSION}
          and tasks_ok and df.eeg_file.str.contains(EXPECTED_EEG).all(),
          f"subjects={subs} sessions={sess} task_ok={bool(tasks_ok)}")

    # Extra context
    print(f"\nrecalled=1: {int((df.recalled==1).sum())}   "
          f"recalled=0: {int((df.recalled==0).sum())}   "
          f"recall rate: {df.recalled.mean():.4f}")
    print(f"unique words: {df.word.nunique()}   trials: {df.trial.nunique()}")

    print("\n" + ("SUCCESS: all canonical checks passed." if ok
                  else "FAILURE: one or more checks failed."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
