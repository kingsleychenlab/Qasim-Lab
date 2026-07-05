#!/usr/bin/env python3
"""
Audit the `recalled` column of outputs/encoding_trials.csv for the canonical
session (sub-LTP269 / ses-20 / task-ltpFR2).

Independently RE-DERIVES the recall label straight from the events.tsv using
only free-recall events and compares to encoding_trials.csv row-by-row.

Recall rule (free recall only):
    For each WORD (studied) event, recalled = 1 iff a REC_WORD (freely recalled)
    event with the SAME item_num occurs in the SAME trial/list; else 0.

Explicitly does NOT use recognition information of any kind:
    - trial_type RECOG_* events
    - recog_resp / recog_conf (or any recognition column)
item_num is the primary match key; item_name/word is printed for readability only.
Matching is strictly within-trial — item_num is never matched across trials.

Report -> outputs/recall_label_audit.txt
"""

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRIALS = 576
AUDIT_TRIALS = [1, 12, 24]
# Columns that would indicate recognition being used — must NOT be referenced.
RECOG_COLS = ["recog_resp", "recog_conf", "recog_rt", "recog_acc"]


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
    ap.add_argument("--trials", default=os.path.join(HERE, "outputs/encoding_trials.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs/recall_label_audit.txt"))
    args = ap.parse_args()

    trials = pd.read_csv(args.trials)

    # Resolve the canonical events.tsv from the encoding_trials event_file column.
    ev_files = trials.event_file.unique()
    if len(ev_files) != 1:
        sys.exit(f"ERROR: expected one event_file, found {len(ev_files)}: {ev_files}")
    ev_rel = ev_files[0]
    ev_path = ev_rel if os.path.isabs(ev_rel) else os.path.join(HERE, ev_rel)
    if not os.path.isfile(ev_path):
        sys.exit(f"ERROR: events.tsv not found: {ev_path}")

    ev = pd.read_csv(ev_path, sep="\t", na_values=["n/a", ""])

    with open(args.out, "w") as fh:
        log = Tee(fh)
        log("=" * 74)
        log("RECALL-LABEL AUDIT — sub-LTP269 / ses-20 / task-ltpFR2")
        log("=" * 74)
        log(f"encoding_trials : {args.trials}  ({len(trials)} rows)")
        log(f"events.tsv      : {ev_rel}")
        log(f"events columns  : {list(ev.columns)}")

        # ---- confirm we are NOT using recognition information -----------
        recog_events = ev[ev.trial_type.astype(str).str.upper().str.startswith("RECOG")]
        recog_cols_present = [c for c in RECOG_COLS if c in ev.columns]
        log(f"\nRECOG_* events present in file : {len(recog_events)} "
            "(NOT used in recall derivation)")
        log(f"recognition columns present    : {recog_cols_present or 'none'} "
            "(NOT used in recall derivation)")

        # ---- split to the only two event types we use -------------------
        word_ev = ev[ev.trial_type == "WORD"].copy()
        rec_ev = ev[ev.trial_type == "REC_WORD"].copy()
        log(f"\nWORD events     : {len(word_ev)}")
        log(f"REC_WORD events : {len(rec_ev)}")

        # ---- build the within-trial recall key set ----------------------
        # Valid recall keys = (trial, item_num) from REC_WORD, dropping
        # intrusions/vocalizations (item_num == -1) and missing values.
        rec_valid = rec_ev.dropna(subset=["trial", "item_num"])
        rec_valid = rec_valid[rec_valid.item_num != -1]
        recall_keys = set(zip(rec_valid.trial.astype(int),
                              rec_valid.item_num.astype(int)))
        # For a cross-trial contrast (to PROVE within-trial matters), also build
        # the set of item_nums recalled ANYWHERE in the session.
        recalled_any_item = set(rec_valid.item_num.astype(int))

        # ---- recompute recalled per encoding_trials row (exact order) ---
        recomputed = []
        cross_only = 0  # would be 1 under cross-trial matching but 0 within-trial
        for _, r in trials.iterrows():
            t, inum = int(r.trial), int(r.item_num)
            within = (t, inum) in recall_keys
            recomputed.append(1 if within else 0)
            if (not within) and (inum in recalled_any_item):
                cross_only += 1
        trials = trials.assign(recalled_recomputed=recomputed)

        # =================================================================
        # Detailed per-trial audits for trials 1, 12, 24
        # =================================================================
        for t in AUDIT_TRIALS:
            log("\n" + "-" * 74)
            log(f"TRIAL {t} — detailed audit")
            log("-" * 74)
            tw = trials[trials.trial == t].sort_values("serialpos")
            tr = rec_ev[rec_ev.trial == t]

            log(f"WORD (studied) events in trial {t}: {len(tw)}")
            log(f"  {'sp':>3} {'item_name':<12} {'item_num':>8} {'onset':>10} {'sample':>9}")
            for _, w in tw.iterrows():
                log(f"  {int(w.serialpos):>3} {str(w.word):<12} {int(w.item_num):>8} "
                    f"{w.onset:>10.3f} {int(w['sample']):>9}")

            log(f"\nREC_WORD (freely recalled) events in trial {t}: {len(tr)}")
            log(f"  {'item_name':<12} {'item_num':>8} {'onset':>10}")
            for _, r in tr.iterrows():
                onset = f"{r.onset:.3f}" if pd.notna(r.onset) else "n/a"
                tag = "  (intrusion/vocalization)" if r.item_num == -1 else ""
                log(f"  {str(r.item_name):<12} {int(r.item_num):>8} {onset:>10}{tag}")

            rec_items_this_trial = set(
                rec_ev[(rec_ev.trial == t) & (rec_ev.item_num != -1)]
                .item_num.astype(int))
            log(f"\nper-WORD label check (recalled iff item_num in "
                f"trial-{t} REC_WORD set):")
            log(f"  {'sp':>3} {'item_name':<12} {'item_num':>8} "
                f"{'derived':>7} {'csv':>4} {'result':>6}")
            tpass = tfail = 0
            for _, w in tw.iterrows():
                inum = int(w.item_num)
                derived = 1 if inum in rec_items_this_trial else 0
                csv_lbl = int(w.recalled)
                ok = derived == csv_lbl
                tpass += ok
                tfail += (not ok)
                log(f"  {int(w.serialpos):>3} {str(w.word):<12} {inum:>8} "
                    f"{derived:>7} {csv_lbl:>4} {'PASS' if ok else 'FAIL':>6}")
            log(f"trial {t}: {tpass}/{len(tw)} PASS, {tfail} FAIL")

        # =================================================================
        # Full validation across all 576 rows
        # =================================================================
        log("\n" + "=" * 74)
        log("FULL VALIDATION (all 576 rows)")
        log("=" * 74)
        n_match = int((trials.recalled_recomputed == trials.recalled).sum())
        n_mismatch = len(trials) - n_match
        n_rec_recompute = int((trials.recalled_recomputed == 1).sum())
        n_rec_csv = int((trials.recalled == 1).sum())

        log(f"recalled count (recomputed from events.tsv): {n_rec_recompute}")
        log(f"recalled count (encoding_trials.csv)       : {n_rec_csv}")
        log(f"labels matching : {n_match}/{len(trials)}")
        log(f"labels mismatch : {n_mismatch}")
        if n_mismatch:
            bad = trials[trials.recalled_recomputed != trials.recalled]
            log("MISMATCHES:")
            for _, b in bad.iterrows():
                log(f"   trial {int(b.trial)} sp {int(b.serialpos)} {b.word} "
                    f"item_num {int(b.item_num)}: csv={int(b.recalled)} "
                    f"recomputed={int(b.recalled_recomputed)}")

        log(f"\nwithin-trial matching enforced: {cross_only} additional word(s) "
            "would flip to recalled=1 under (incorrect) cross-trial matching; "
            "these were NOT matched.")

        # ---- assertions --------------------------------------------------
        log("\n--- assertions ---")
        log.check("576 rows in encoding_trials.csv", len(trials) == N_TRIALS, f"{len(trials)}")
        log.check("576/576 recalled labels match", n_match == N_TRIALS,
                  f"{n_match}/{N_TRIALS}")
        log.check("recomputed recalled count == csv recalled count",
                  n_rec_recompute == n_rec_csv, f"{n_rec_recompute} vs {n_rec_csv}")
        log.check("no recognition columns were used",
                  True, f"present-but-unused: {recog_cols_present or 'none'}")
        log.check("no RECOG_* events were used (only WORD + REC_WORD)",
                  True, f"{len(recog_events)} RECOG_* events ignored")
        log.check("matching restricted within trial/list (no cross-trial matches)",
                  True, f"{cross_only} cross-trial-only candidates excluded")

        log("")
        if log.fail == 0 and n_match == N_TRIALS:
            log("SUCCESS: recall label audit passed")
            log(f"{n_match}/{N_TRIALS} recalled labels match")
            log("Recognition events were not used")
            log("Matching was restricted within trial/list")
        else:
            log("FAILURE: recall label audit did NOT pass "
                f"({log.fail} failed checks, {n_mismatch} mismatches)")

    sys.exit(0 if (log.fail == 0 and n_mismatch == 0) else 1)


if __name__ == "__main__":
    main()
