#!/usr/bin/env python3
"""
Same as step09, but several sessions per subject, which is what lets the
outline's model include a session term.

This is the stage that produced the reported runs. The outline's model is
`recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`, and a
`session` term needs more than one session per subject to estimate. step09 takes
one session each; this takes up to K.

It reuses step09.process_session unchanged instead of reimplementing the chain
(download -> encoding_trials -> Y_t5 -> X_eeg -> ridge metrics). Only the
selection differs. If the two ever drifted apart, the multi-session runs would
no longer be comparable to the single-session ones, and the whole 4 vs 16 vs 32
comparison would be meaningless.

Same validity screen as step09 (task==ltpFR2, WORD==576, peers coverage 576/576,
576/576 valid 300-800 ms windows), checked from the EDF header before download.

Combined table -> outputs/all_sessions_fidelity_results.csv:
  subject,session,trial,serialpos,word,recalled,embedding_fidelity,
  raw_cosine,centered_cosine,true_word_percentile,centered_true_word_percentile
embedding_fidelity == raw_cosine, matching the outline's naming.

Stops here on purpose. Fitting the model is step11's job; keeping the expensive
data-building separate from the cheap model fit means the model can be refit
without re-running any EEG work.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from common import Tee, peers_word_set

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "code"))
import step02_find_sessions as F           # noqa: E402
import step09_scale_multi_subject as M            # noqa: E402

OUT_COLS = ["subject", "session", "trial", "serialpos", "word", "recalled",
            "embedding_fidelity", "raw_cosine", "centered_cosine",
            "true_word_percentile", "centered_true_word_percentile"]


def select_multi(task, peers, n_subjects, per_subject, win_start, win_stop, log,
                 prefer_subjects=None):
    """Pick N subjects, each with up to `per_subject` valid sessions
    (smallest EDF first). If prefer_subjects is given, consider those first
    (maximizes reuse of already-downloaded sessions)."""
    log(f"scanning OpenNeuro S3 for task-{task} sessions (metadata only) ...")
    sess, pages = F.scan_task_sessions(task)
    by_sub = defaultdict(list)
    for (sub, ses), v in sess.items():
        if "edf" in v:
            by_sub[sub].append((ses, v["edf"]))
    # subjects ordered by their smallest EDF (proxy for a complete recording)
    sub_order = sorted(by_sub, key=lambda s: min(e for _, e in by_sub[s]))
    if prefer_subjects:
        pref = [s for s in prefer_subjects if s in by_sub]
        sub_order = pref + [s for s in sub_order if s not in set(pref)]
    log(f"scanned {pages} pages; {len(sess)} sessions; {len(by_sub)} subjects. "
        f"Selecting {n_subjects} subjects x up to {per_subject} valid sessions ...")

    chosen = []
    for sub in sub_order:
        if len({s for s, _ in chosen}) >= n_subjects and sub not in {s for s, _ in chosen}:
            break
        valid_here = []
        for ses, edf in sorted(by_sub[sub], key=lambda x: x[1]):  # smallest first
            c = F.peek_session(sub, ses, task, peers, edf, win_start, win_stop)
            if c and c["n_word"] == 576 and c["coverage"] >= 0.999 and c["n_valid_win"] == 576:
                valid_here.append((ses, edf))
                if len(valid_here) >= per_subject:
                    break
        if len(valid_here) >= min(2, per_subject):   # require multiple sessions
            for ses, edf in valid_here:
                chosen.append((sub, ses))
            log(f"  {sub}: {[s for s, _ in valid_here]}")
        if len({s for s, _ in chosen}) >= n_subjects:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="ltpFR2")
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--sessions-per-subject", type=int, default=2)
    ap.add_argument("--prefer-subjects", default=None,
                    help="Comma list of subject labels to consider first "
                         "(e.g. sub-LTP269,sub-LTP303) to reuse existing downloads.")
    ap.add_argument("--peers-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    ap.add_argument("--win-start", type=float, default=0.300)
    ap.add_argument("--win-stop", type=float, default=0.800)
    ap.add_argument("--combined", default=os.path.join(HERE, "outputs/all_sessions_fidelity_results.csv"))
    ap.add_argument("--report", default=os.path.join(HERE, "outputs/all_sessions_summary.txt"))
    args = ap.parse_args()

    peers = peers_word_set(args.peers_order)
    fh = open(args.report, "w")

    log = Tee(fh)

    log("=" * 74)
    log("MULTI-SESSION ltpFR2 SCALING (session term enabled; NOT final model)")
    log("=" * 74)

    prefer = args.prefer_subjects.split(",") if args.prefer_subjects else None
    chosen = select_multi(args.task, peers, args.n_subjects,
                          args.sessions_per_subject, args.win_start, args.win_stop, log,
                          prefer_subjects=prefer)
    if not chosen:
        sys.exit("No valid sessions found.")
    log(f"\nselected {len(chosen)} sessions across "
        f"{len({s for s,_ in chosen})} subjects:")
    for sub, ses in chosen:
        log(f"  {sub}/{ses}")

    # -----------------------------------------------------------------
    # Process each session (reuse existing fidelity csv if already present)
    # -----------------------------------------------------------------
    fid_paths = []
    for i, (sub, ses) in enumerate(chosen, 1):
        d = os.path.join(HERE, "outputs", "subjects", f"{sub}_{ses}")
        fid = os.path.join(d, "fidelity_results_corrected.csv")
        log(f"\n[{i}/{len(chosen)}] === {sub}/{ses} ===")
        if os.path.isfile(fid):
            log(f"    reuse existing -> {os.path.relpath(fid, HERE)}")
            fid_paths.append(fid)
            continue
        p = M.process_session(sub, ses, args.task, log)
        if p:
            fid_paths.append(p)
            log(f"    OK -> {os.path.relpath(p, HERE)}")

    if not fid_paths:
        sys.exit("No sessions processed successfully.")

    # -----------------------------------------------------------------
    # Combine (embedding_fidelity == raw_cosine)
    # -----------------------------------------------------------------
    frames = []
    for p in fid_paths:
        df = pd.read_csv(p)
        df["embedding_fidelity"] = df["raw_cosine"]
        frames.append(df[OUT_COLS])
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.combined, index=False)
    log(f"\ncombined -> {os.path.relpath(args.combined, HERE)}  ({len(combined)} rows)")

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    log("\n" + "=" * 74)
    log("EXPANDED TABLE VALIDATION")
    log("=" * 74)
    check = log.check

    n_sub = combined.subject.nunique()
    per_sub = combined.groupby("subject").session.nunique()
    n_ses = combined.groupby(["subject", "session"]).ngroups
    multi = int((per_sub >= 2).sum())

    num = combined.drop(columns=["subject", "word"])
    check("multiple subjects", n_sub >= 2, f"{n_sub}")
    check("multiple sessions per subject (>=1 subject with >=2 sessions)",
          multi >= 1, f"{multi} subject(s) with >=2 sessions")
    check("no NaN", not combined.isna().any().any())
    check("no Inf in numeric cols",
          bool(np.isfinite(num.select_dtypes("number").to_numpy()).all()))
    check("recalled only 0/1", set(combined.recalled.unique()) <= {0, 1},
          f"{sorted(combined.recalled.unique().tolist())}")
    req = ["subject", "session", "word", "recalled", "embedding_fidelity"]
    check("every row has subject/session/word/recalled/embedding_fidelity",
          bool(combined[req].notna().all().all()))
    check("embedding_fidelity == raw_cosine",
          bool(np.allclose(combined.embedding_fidelity, combined.raw_cosine)))
    check("true_word_percentile in [0,1]",
          bool((combined.true_word_percentile.between(0, 1)).all()))
    check("centered_true_word_percentile in [0,1]",
          bool((combined.centered_true_word_percentile.between(0, 1)).all()))

    # -----------------------------------------------------------------
    # Summary prints
    # -----------------------------------------------------------------
    log("\n--- expanded table summary ---")
    log(f"number of subjects : {n_sub}")
    log(f"number of sessions : {n_ses}")
    log(f"subjects with >=2 sessions: {multi}")
    log("sessions per subject:")
    for sub, k in per_sub.items():
        ses_list = sorted(combined[combined.subject == sub].session.unique().tolist())
        log(f"   {sub}: {k} sessions {ses_list}")
    log(f"total trials       : {len(combined)}")
    log(f"recalled           : {int((combined.recalled==1).sum())}")
    log(f"forgotten          : {int((combined.recalled==0).sum())}")
    log(f"recall rate        : {combined.recalled.mean():.3f}")

    log("\n--- mean embedding_fidelity (raw_cosine) & centered_true_word_percentile "
        "by subject/session ---")
    by = combined.groupby(["subject", "session"]).agg(
        n=("word", "size"),
        embedding_fidelity=("embedding_fidelity", "mean"),
        centered_true_word_percentile=("centered_true_word_percentile", "mean"))
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.4f}"):
        log(by.to_string())

    log("\n*** EXPANDED MULTI-SESSION TABLE READY. Session term now estimable. "
        "NOT the final mixed-effects model. ***")
    log("STATUS: " + ("OK" if log.fail == 0 else f"FAILED ({log.fail} checks)"))
    fh.close()
    sys.exit(0 if log.fail == 0 else 1)


if __name__ == "__main__":
    main()
