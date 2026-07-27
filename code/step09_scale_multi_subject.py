#!/usr/bin/env python3
"""
Run the whole single-session pipeline across many subjects, one session each.

This is what gets from "it works on one session" to a result worth reporting.
It picks subjects, runs the same five stages on each session, and concatenates
the per-trial fidelity tables into one table for the model.

The session selection matters. A session is used only if it is complete on every
axis the pipeline needs:

    task == ltpFR2       the 576-word pool the T5 embeddings were built from;
                         plain ltpFR uses a different, larger pool
    WORD events == 576   the full list was actually presented
    peers coverage 576   every presented word has an embedding
    576 valid windows    the full 300-800 ms window fits inside the recording
                         for every trial (truncated EDFs fail here)

All four are screened from the EDF header before downloading, since the EDFs are
large and most candidate sessions fail. Sessions are dropped only for being
incomplete, never for any reason tied to the outcome, so this selection cannot
bias the memory result.

Per session it runs the exact validated chain:

    step03_download_session.py  (--task ltpFR2, sidecars + one EDF only)
    step05_create_encoding_trials.py  -> encoding_trials.csv (recall from REC_WORD/item_num)
    step06_build_trial_targets.py     -> Y_t5.npy (validated PEERS T5 embeddings)
    step07_extract_eeg_features.py    -> X_eeg.npy (sample-based 300-800 ms, raw 500 Hz)
    step08_run_ridge_cv.py -> fidelity_results_corrected.csv (+shuffled control)

Per-session outputs go under outputs/subjects/<subject>_ses-<session>/.
A combined table is written to outputs/all_subjects_fidelity_results.csv.

This does not run the mixed-effects memory model.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from common import Tee, peers_word_set

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
sys.path.insert(0, os.path.join(HERE, "code"))
import step02_find_sessions as F  # noqa: E402

DATASET = "ds004395"
COMBINED_COLS = [
    "subject", "session", "trial", "serialpos", "word", "recalled",
    "raw_cosine", "centered_cosine",
    "true_word_rank", "true_word_percentile",
    "top1_correct", "top5_correct", "top10_correct",
    "centered_true_word_rank", "centered_true_word_percentile",
    "centered_top1_correct", "centered_top5_correct", "centered_top10_correct",
]


def run(cmd, log):
    log("    $ " + " ".join(os.path.basename(part) if part.endswith(".py") else part for part in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"    !! exit {result.returncode}")
        tail = (result.stdout + result.stderr).strip().splitlines()[-8:]
        for line in tail:
            log("       " + line)
    return result.returncode


def select_sessions(task, peers, n_subjects, win_start, win_stop, log):
    """Screen candidates smallest-EDF-first; take the first valid session per
    distinct subject until n_subjects are chosen."""
    log(f"scanning OpenNeuro S3 for task-{task} sessions (metadata only) ...")
    sessions, pages = F.scan_task_sessions(task)
    with_edf = sorted([(key, info) for key, info in sessions.items() if "edf" in info],
                      key=lambda x: x[1]["edf"])
    log(f"scanned {pages} pages; {len(sessions)} sessions with EDFs. "
        f"Screening for {n_subjects} distinct valid subjects ...")
    chosen, seen_subjects = [], set()
    for (sub, ses), info in with_edf:
        if sub in seen_subjects:
            continue
        candidate = F.peek_session(sub, ses, task, peers, info["edf"], win_start, win_stop)
        if not candidate:
            continue
        is_valid = (candidate["n_word"] == 576 and candidate["coverage"] >= 0.999
              and candidate["n_valid_win"] == 576)
        if is_valid:
            chosen.append((sub, ses, info["edf"], candidate))
            seen_subjects.add(sub)
            log(f"  + {sub}/{ses}  EDF {info['edf']/1e6:.0f}MB  "
                f"valid_win {candidate['n_valid_win']}/576")
            if len(chosen) >= n_subjects:
                break
    return chosen


def process_session(sub, ses, task, log):
    """Run the full per-session chain. Returns path to fidelity csv or None."""
    session_dir = os.path.join(HERE, "outputs", "subjects", f"{sub}_{ses}")
    os.makedirs(session_dir, exist_ok=True)
    sub_label = sub.replace("sub-", "")
    ses_label = ses.replace("ses-", "")

    events = os.path.join(HERE, "data", DATASET, sub, ses, "eeg",
                          f"{sub}_{ses}_task-{task}_events.tsv")
    eeg = os.path.join(HERE, "data", DATASET, sub, ses, "eeg",
                       f"{sub}_{ses}_task-{task}_eeg.edf")
    enc = os.path.join(session_dir, "encoding_trials.csv")
    y_path = os.path.join(session_dir, "Y_t5.npy")
    x_path = os.path.join(session_dir, "X_eeg.npy")
    tmeta = os.path.join(session_dir, "trial_metadata.csv")
    fid = os.path.join(session_dir, "fidelity_results_corrected.csv")

    # The per-session chain, run as subprocesses rather than imports. Each stage
    # stays independently runnable and debuggable on one session, and a crash in
    # one session cannot take down the whole scale-up. Everything lands under
    # this session's own directory, so sessions never overwrite each other.
    code_dir = os.path.join(HERE, "code")
    steps = [
        ([PY, os.path.join(code_dir, "step03_download_session.py"),
          "--sub", sub_label, "--ses", ses_label, "--task", task], "download"),
        ([PY, os.path.join(code_dir, "step05_create_encoding_trials.py"),
          "--events", events, "--eeg", eeg,
          "--out-csv", enc, "--out-summary", os.path.join(session_dir, "encoding_trials_summary.txt")],
         "encoding_trials"),
        ([PY, os.path.join(code_dir, "step06_build_trial_targets.py"),
          "--trials", enc, "--out-y", y_path,
          "--out-meta", os.path.join(session_dir, "target_metadata.json"),
          "--out-map", os.path.join(session_dir, "trial_targets_metadata.csv")], "Y_t5"),
        ([PY, os.path.join(code_dir, "step07_extract_eeg_features.py"),
          "--trials", enc, "--out-x", x_path, "--out-meta-csv", tmeta,
          "--out-meta-json", os.path.join(session_dir, "eeg_feature_metadata.json")], "X_eeg"),
        ([PY, os.path.join(code_dir, "step08_run_ridge_cv.py"),
          "--x", x_path, "--y", y_path, "--meta", tmeta,
          "--out-csv", fid,
          "--out-pred", os.path.join(session_dir, "predicted_embeddings_corrected.npy"),
          "--out-meta", os.path.join(session_dir, "ridge_corrected_metadata.json"),
          "--out-summary", os.path.join(session_dir, "ridge_corrected_summary.txt")], "corrected_metrics"),
    ]
    for cmd, name in steps:
        return_code = run(cmd, log)
        # step05_create_encoding_trials exits 2 only if trials dropped; for screened
        # valid sessions it should be 0. Any nonzero on other steps = abort.
        if return_code != 0:
            log(f"    ABORT session {sub}/{ses} at step '{name}' (exit {return_code})")
            return None
    return fid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ltpFR2")
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--peers-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    parser.add_argument("--win-start", type=float, default=0.300)
    parser.add_argument("--win-stop", type=float, default=0.800)
    parser.add_argument("--combined", default=os.path.join(HERE, "outputs/all_subjects_fidelity_results.csv"))
    parser.add_argument("--report", default=os.path.join(HERE, "outputs/all_subjects_summary.txt"))
    args = parser.parse_args()

    peers = peers_word_set(args.peers_order)
    report_file = open(args.report, "w")

    log = Tee(report_file)

    log("=" * 74)
    log("MULTI-SUBJECT ltpFR2 SCALING (NOT final mixed-effects)")
    log("=" * 74)

    chosen = select_sessions(args.task, peers, args.n_subjects,
                             args.win_start, args.win_stop, log)
    if not chosen:
        sys.exit("No valid sessions found.")
    log(f"\nselected {len(chosen)} sessions:")
    for sub, ses, edf, candidate in chosen:
        log(f"  {sub}/{ses}  ({edf/1e6:.0f} MB)")

    fid_paths = []
    for i, (sub, ses, edf, candidate) in enumerate(chosen, 1):
        log(f"\n[{i}/{len(chosen)}] === {sub}/{ses} ===")
        fid_path = process_session(sub, ses, args.task, log)
        if fid_path:
            fid_paths.append(fid_path)
            log(f"    OK -> {os.path.relpath(fid_path, HERE)}")

    if not fid_paths:
        sys.exit("No sessions processed successfully.")

    frames = [pd.read_csv(fid_path) for fid_path in fid_paths]
    combined = pd.concat(frames, ignore_index=True)[COMBINED_COLS]
    combined.to_csv(args.combined, index=False)
    log(f"\ncombined -> {os.path.relpath(args.combined, HERE)}  ({len(combined)} rows)")

    log("\n" + "=" * 74)
    log("COMBINED VALIDATION")
    log("=" * 74)
    check = log.check

    num = combined.drop(columns=["subject", "word"])
    check("no NaN in combined", not combined.isna().any().any())
    check("no Inf in numeric columns",
          bool(np.isfinite(num.select_dtypes("number").to_numpy()).all()))
    check("all rows have subject/session/word/recalled",
          bool(combined[["subject", "session", "word", "recalled"]].notna().all().all()))
    check("recalled is only 0/1", set(combined.recalled.unique()) <= {0, 1},
          f"{sorted(combined.recalled.unique().tolist())}")
    for col in ("true_word_rank", "centered_true_word_rank"):
        check(f"{col} in [1,576]",
              bool((combined[col] >= 1).all() and (combined[col] <= 576).all()),
              f"[{combined[col].min()}, {combined[col].max()}]")
    for col in ("true_word_percentile", "centered_true_word_percentile"):
        check(f"{col} in [0,1]",
              bool((combined[col] >= 0).all() and (combined[col] <= 1).all()))
    for col in ("top1_correct", "top5_correct", "top10_correct",
                "centered_top1_correct", "centered_top5_correct", "centered_top10_correct"):
        check(f"{col} is 0/1", set(combined[col].unique()) <= {0, 1})

    log("\n--- combined summary ---")
    n_sub = combined.subject.nunique()
    n_ses = combined.groupby(["subject", "session"]).ngroups
    log(f"subjects        : {n_sub}")
    log(f"sessions        : {n_ses}")
    log(f"total trials    : {len(combined)}")
    log(f"recalled        : {int((combined.recalled==1).sum())}")
    log(f"forgotten       : {int((combined.recalled==0).sum())}")

    metrics = ["raw_cosine", "centered_cosine", "true_word_rank",
               "true_word_percentile", "top1_correct", "top5_correct",
               "top10_correct", "centered_true_word_rank",
               "centered_true_word_percentile", "centered_top1_correct",
               "centered_top5_correct", "centered_top10_correct"]
    log("\n--- mean CORRECTED metrics by subject ---")
    means = combined.groupby(["subject", "session"])[metrics].mean()
    with pd.option_context("display.width", 220, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.3f}"):
        log(means.to_string())

    # real vs shuffled by subject: read each session's metadata json
    log("\n--- REAL vs SHUFFLED decoding by subject (key retrieval metrics) ---")
    log(f"{'subject/session':22} {'metric':<28} {'real':>9} {'shuffled':>9}")
    key_metrics = ["true_word_percentile", "centered_true_word_percentile",
                   "top5_correct", "top10_correct"]
    for fid_path in fid_paths:
        session_dir = os.path.dirname(fid_path)
        meta_json_path = os.path.join(session_dir, "ridge_corrected_metadata.json")
        if not os.path.isfile(meta_json_path):
            continue
        meta = json.load(open(meta_json_path))
        tag = f"{meta.get('subject')}/ses-{meta.get('session')}"
        real_vs_shuffled = meta.get("real_vs_shuffled", {})
        for key_metric in key_metrics:
            if key_metric in real_vs_shuffled:
                log(f"{tag:22} {key_metric:<28} {real_vs_shuffled[key_metric]['real']:>9.4f} "
                    f"{real_vs_shuffled[key_metric]['shuffled']:>9.4f}")

    log("\n*** MULTI-SUBJECT SMOKE-SCALE — corrected metrics, per-subject "
        "held-out CV. NOT the mixed-effects model. ***")
    log("STATUS: " + ("OK" if log.fail == 0 else f"FAILED ({log.fail} checks)"))
    report_file.close()
    sys.exit(0 if log.fail == 0 else 1)


if __name__ == "__main__":
    main()
