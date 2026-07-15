#!/usr/bin/env python3
"""
Run the whole single-session pipeline across many subjects, one session each.

The bridge between "it works on one session" and a result worth reporting. It
picks subjects, drives the same five stages per session, and concatenates the
per-trial fidelity tables into one table for the model.

Session selection is the part worth understanding. A session is used only if it
is complete on every axis the pipeline needs:

    task == ltpFR2       the 576-word pool the T5 embeddings were built from;
                         plain ltpFR uses a different, larger pool
    WORD events == 576   the full list was actually presented
    peers coverage 576   every presented word has an embedding
    576 valid windows    the full 300-800 ms window fits inside the recording
                         for every trial (truncated EDFs fail here)

All four are screened from the EDF header *before* downloading, since the EDFs
are large and most candidate sessions fail. Sessions are not dropped for any
reason related to the outcome -- only for being incomplete -- so this selection
cannot bias the memory result.

Per session it runs the exact validated chain:

    step03_download_session.py  (--task ltpFR2, sidecars + one EDF only)
    step05_create_encoding_trials.py  -> encoding_trials.csv (recall from REC_WORD/item_num)
    step06_build_trial_targets.py     -> Y_t5.npy (validated PEERS T5 embeddings)
    step07_extract_eeg_features.py    -> X_eeg.npy (sample-based 300-800 ms, raw 500 Hz)
    step08_run_ridge_cv.py -> fidelity_results_corrected.csv (+shuffled control)

Per-session outputs go under outputs/subjects/<subject>_ses-<session>/.
A combined table is written to outputs/all_subjects_fidelity_results.csv.

This does NOT run the mixed-effects memory model.
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
    log("    $ " + " ".join(os.path.basename(c) if c.endswith(".py") else c for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"    !! exit {r.returncode}")
        tail = (r.stdout + r.stderr).strip().splitlines()[-8:]
        for ln in tail:
            log("       " + ln)
    return r.returncode


def select_sessions(task, peers, n_subjects, win_start, win_stop, log):
    """Screen candidates smallest-EDF-first; take the first valid session per
    distinct subject until n_subjects are chosen."""
    log(f"scanning OpenNeuro S3 for task-{task} sessions (metadata only) ...")
    sess, pages = F.scan_task_sessions(task)
    with_edf = sorted([(k, v) for k, v in sess.items() if "edf" in v],
                      key=lambda x: x[1]["edf"])
    log(f"scanned {pages} pages; {len(sess)} sessions with EDFs. "
        f"Screening for {n_subjects} distinct valid subjects ...")
    chosen, seen_subjects = [], set()
    for (sub, ses), v in with_edf:
        if sub in seen_subjects:
            continue
        c = F.peek_session(sub, ses, task, peers, v["edf"], win_start, win_stop)
        if not c:
            continue
        ok = (c["n_word"] == 576 and c["coverage"] >= 0.999
              and c["n_valid_win"] == 576)
        if ok:
            chosen.append((sub, ses, v["edf"], c))
            seen_subjects.add(sub)
            log(f"  + {sub}/{ses}  EDF {v['edf']/1e6:.0f}MB  "
                f"valid_win {c['n_valid_win']}/576")
            if len(chosen) >= n_subjects:
                break
    return chosen


def process_session(sub, ses, task, log):
    """Run the full per-session chain. Returns path to fidelity csv or None."""
    d = os.path.join(HERE, "outputs", "subjects", f"{sub}_{ses}")
    os.makedirs(d, exist_ok=True)
    sub_label = sub.replace("sub-", "")
    ses_label = ses.replace("ses-", "")

    events = os.path.join(HERE, "data", DATASET, sub, ses, "eeg",
                          f"{sub}_{ses}_task-{task}_events.tsv")
    eeg = os.path.join(HERE, "data", DATASET, sub, ses, "eeg",
                       f"{sub}_{ses}_task-{task}_eeg.edf")
    enc = os.path.join(d, "encoding_trials.csv")
    y = os.path.join(d, "Y_t5.npy")
    x = os.path.join(d, "X_eeg.npy")
    tmeta = os.path.join(d, "trial_metadata.csv")
    fid = os.path.join(d, "fidelity_results_corrected.csv")

    # The per-session chain, run as subprocesses rather than imports. Each stage
    # stays independently runnable and debuggable on one session, and a crash in
    # one session cannot take down the whole scale-up. Everything lands under
    # this session's own directory, so sessions never overwrite each other.
    S = os.path.join(HERE, "code")
    steps = [
        ([PY, os.path.join(S, "step03_download_session.py"),
          "--sub", sub_label, "--ses", ses_label, "--task", task], "download"),
        ([PY, os.path.join(S, "step05_create_encoding_trials.py"),
          "--events", events, "--eeg", eeg,
          "--out-csv", enc, "--out-summary", os.path.join(d, "encoding_trials_summary.txt")],
         "encoding_trials"),
        ([PY, os.path.join(S, "step06_build_trial_targets.py"),
          "--trials", enc, "--out-y", y,
          "--out-meta", os.path.join(d, "target_metadata.json"),
          "--out-map", os.path.join(d, "trial_targets_metadata.csv")], "Y_t5"),
        ([PY, os.path.join(S, "step07_extract_eeg_features.py"),
          "--trials", enc, "--out-x", x, "--out-meta-csv", tmeta,
          "--out-meta-json", os.path.join(d, "eeg_feature_metadata.json")], "X_eeg"),
        ([PY, os.path.join(S, "step08_run_ridge_cv.py"),
          "--x", x, "--y", y, "--meta", tmeta,
          "--out-csv", fid,
          "--out-pred", os.path.join(d, "predicted_embeddings_corrected.npy"),
          "--out-meta", os.path.join(d, "ridge_corrected_metadata.json"),
          "--out-summary", os.path.join(d, "ridge_corrected_summary.txt")], "corrected_metrics"),
    ]
    for cmd, name in steps:
        rc = run(cmd, log)
        # step05_create_encoding_trials exits 2 only if trials dropped; for screened
        # valid sessions it should be 0. Any nonzero on other steps = abort.
        if rc != 0:
            log(f"    ABORT session {sub}/{ses} at step '{name}' (exit {rc})")
            return None
    return fid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="ltpFR2")
    ap.add_argument("--n-subjects", type=int, default=5)
    ap.add_argument("--peers-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    ap.add_argument("--win-start", type=float, default=0.300)
    ap.add_argument("--win-stop", type=float, default=0.800)
    ap.add_argument("--combined", default=os.path.join(HERE, "outputs/all_subjects_fidelity_results.csv"))
    ap.add_argument("--report", default=os.path.join(HERE, "outputs/all_subjects_summary.txt"))
    args = ap.parse_args()

    peers = peers_word_set(args.peers_order)
    fh = open(args.report, "w")

    log = Tee(fh)

    log("=" * 74)
    log("MULTI-SUBJECT ltpFR2 SCALING (NOT final mixed-effects)")
    log("=" * 74)

    chosen = select_sessions(args.task, peers, args.n_subjects,
                             args.win_start, args.win_stop, log)
    if not chosen:
        sys.exit("No valid sessions found.")
    log(f"\nselected {len(chosen)} sessions:")
    for sub, ses, edf, c in chosen:
        log(f"  {sub}/{ses}  ({edf/1e6:.0f} MB)")

    # -----------------------------------------------------------------
    # Process each session
    # -----------------------------------------------------------------
    fid_paths = []
    for i, (sub, ses, edf, c) in enumerate(chosen, 1):
        log(f"\n[{i}/{len(chosen)}] === {sub}/{ses} ===")
        p = process_session(sub, ses, args.task, log)
        if p:
            fid_paths.append(p)
            log(f"    OK -> {os.path.relpath(p, HERE)}")

    if not fid_paths:
        sys.exit("No sessions processed successfully.")

    # -----------------------------------------------------------------
    # Combine
    # -----------------------------------------------------------------
    frames = [pd.read_csv(p) for p in fid_paths]
    combined = pd.concat(frames, ignore_index=True)[COMBINED_COLS]
    combined.to_csv(args.combined, index=False)
    log(f"\ncombined -> {os.path.relpath(args.combined, HERE)}  ({len(combined)} rows)")

    # -----------------------------------------------------------------
    # Validate combined
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Summary prints
    # -----------------------------------------------------------------
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
    by = combined.groupby(["subject", "session"])[metrics].mean()
    with pd.option_context("display.width", 220, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.3f}"):
        log(by.to_string())

    # real vs shuffled by subject: read each session's metadata json
    log("\n--- REAL vs SHUFFLED decoding by subject (key retrieval metrics) ---")
    log(f"{'subject/session':22} {'metric':<28} {'real':>9} {'shuffled':>9}")
    key_metrics = ["true_word_percentile", "centered_true_word_percentile",
                   "top5_correct", "top10_correct"]
    for p in fid_paths:
        d = os.path.dirname(p)
        mj = os.path.join(d, "ridge_corrected_metadata.json")
        if not os.path.isfile(mj):
            continue
        m = json.load(open(mj))
        tag = f"{m.get('subject')}/ses-{m.get('session')}"
        rvs = m.get("real_vs_shuffled", {})
        for km in key_metrics:
            if km in rvs:
                log(f"{tag:22} {km:<28} {rvs[km]['real']:>9.4f} "
                    f"{rvs[km]['shuffled']:>9.4f}")

    log("\n*** MULTI-SUBJECT SMOKE-SCALE — corrected metrics, per-subject "
        "held-out CV. NOT the mixed-effects model. ***")
    log("STATUS: " + ("OK" if log.fail == 0 else f"FAILED ({log.fail} checks)"))
    fh.close()
    sys.exit(0 if log.fail == 0 else 1)


if __name__ == "__main__":
    main()
