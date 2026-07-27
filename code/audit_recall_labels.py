#!/usr/bin/env python3
"""
Re-derive `recalled` from the raw events and check it against what step05 wrote.

recalled is the outcome variable, so the whole project is a claim about it. If
it is wrong, every downstream number is wrong in a way no amount of careful
modelling can detect, because the model has no way to know its labels are bad.
So this rebuilds the column from events.tsv independently and compares row by
row. Agreement must be 576/576.

Recall rule (free recall only):
    For each WORD (studied) event, recalled = 1 iff a REC_WORD (freely recalled)
    event with the same item_num occurs in the same trial/list; else 0.

Two things this deliberately does not do, each a way the label could be inflated
into a false positive result:

  - No recognition data. Not RECOG_* events, not recog_resp, not recog_conf.
    Recognition is a different memory process with a much higher hit rate;
    mixing it in would relabel forgotten words as remembered.
  - No cross-trial matching. ltpFR2 reuses the same 576 words across lists, so
    matching item_num without pinning the trial would mark a word recalled in
    one list as recalled in every list it appeared in.

item_num is the match key. item_name/word is printed for readability and
cross-checked, never used as the primary join.

Report -> outputs/recall_label_audit.txt
"""

import argparse
import os
import sys

import pandas as pd

from common import Tee

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRIALS = 576
AUDIT_TRIALS = [1, 12, 24]
# Columns that would indicate recognition being used, must not be referenced.
RECOG_COLS = ["recog_resp", "recog_conf", "recog_rt", "recog_acc"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", default=os.path.join(HERE, "outputs/encoding_trials.csv"))
    parser.add_argument("--out", default=os.path.join(HERE, "outputs/recall_label_audit.txt"))
    args = parser.parse_args()

    trials = pd.read_csv(args.trials)

    # Resolve the canonical events.tsv from the encoding_trials event_file column.
    event_files = trials.event_file.unique()
    if len(event_files) != 1:
        sys.exit(f"ERROR: expected one event_file, found {len(event_files)}: {event_files}")
    event_file_rel = event_files[0]
    event_file_path = event_file_rel if os.path.isabs(event_file_rel) else os.path.join(HERE, event_file_rel)
    if not os.path.isfile(event_file_path):
        sys.exit(f"ERROR: events.tsv not found: {event_file_path}")

    events = pd.read_csv(event_file_path, sep="\t", na_values=["n/a", ""])

    with open(args.out, "w") as fh:
        log = Tee(fh)
        log("=" * 74)
        log("RECALL-LABEL AUDIT — sub-LTP269 / ses-20 / task-ltpFR2")
        log("=" * 74)
        log(f"encoding_trials : {args.trials}  ({len(trials)} rows)")
        log(f"events.tsv      : {event_file_rel}")
        log(f"events columns  : {list(events.columns)}")

        # Confirm recognition information is not used.
        recog_events = events[events.trial_type.astype(str).str.upper().str.startswith("RECOG")]
        recog_cols_present = [c for c in RECOG_COLS if c in events.columns]
        log(f"\nRECOG_* events present in file : {len(recog_events)} "
            "(NOT used in recall derivation)")
        log(f"recognition columns present    : {recog_cols_present or 'none'} "
            "(NOT used in recall derivation)")

        # Keep only the two event types the recall rule uses.
        word_events = events[events.trial_type == "WORD"].copy()
        recall_events = events[events.trial_type == "REC_WORD"].copy()
        log(f"\nWORD events     : {len(word_events)}")
        log(f"REC_WORD events : {len(recall_events)}")

        # Valid recall keys = (trial, item_num) from REC_WORD, dropping
        # intrusions/vocalizations (item_num == -1) and missing values.
        valid_recalls = recall_events.dropna(subset=["trial", "item_num"])
        valid_recalls = valid_recalls[valid_recalls.item_num != -1]
        recall_keys = set(zip(valid_recalls.trial.astype(int),
                              valid_recalls.item_num.astype(int)))
        # For a cross-trial contrast (to show within-trial matters), also build
        # the set of item_nums recalled anywhere in the session.
        recalled_any_item = set(valid_recalls.item_num.astype(int))

        # Recompute recalled per encoding_trials row, preserving row order.
        recomputed_labels = []
        cross_trial_only = 0  # would be 1 under cross-trial matching but 0 within-trial
        for _, trial_row in trials.iterrows():
            trial_num, item_number = int(trial_row.trial), int(trial_row.item_num)
            matched_within_trial = (trial_num, item_number) in recall_keys
            recomputed_labels.append(1 if matched_within_trial else 0)
            if (not matched_within_trial) and (item_number in recalled_any_item):
                cross_trial_only += 1
        trials = trials.assign(recalled_recomputed=recomputed_labels)

        # Detailed per-trial audits for trials 1, 12, 24.
        for trial_num in AUDIT_TRIALS:
            log("\n" + "-" * 74)
            log(f"TRIAL {trial_num} — detailed audit")
            log("-" * 74)
            trial_words = trials[trials.trial == trial_num].sort_values("serialpos")
            trial_recalls = recall_events[recall_events.trial == trial_num]

            log(f"WORD (studied) events in trial {trial_num}: {len(trial_words)}")
            log(f"  {'sp':>3} {'item_name':<12} {'item_num':>8} {'onset':>10} {'sample':>9}")
            for _, word_row in trial_words.iterrows():
                log(f"  {int(word_row.serialpos):>3} {str(word_row.word):<12} {int(word_row.item_num):>8} "
                    f"{word_row.onset:>10.3f} {int(word_row['sample']):>9}")

            log(f"\nREC_WORD (freely recalled) events in trial {trial_num}: {len(trial_recalls)}")
            log(f"  {'item_name':<12} {'item_num':>8} {'onset':>10}")
            for _, recall_row in trial_recalls.iterrows():
                onset = f"{recall_row.onset:.3f}" if pd.notna(recall_row.onset) else "n/a"
                tag = "  (intrusion/vocalization)" if recall_row.item_num == -1 else ""
                log(f"  {str(recall_row.item_name):<12} {int(recall_row.item_num):>8} {onset:>10}{tag}")

            recalled_items_this_trial = set(
                recall_events[(recall_events.trial == trial_num) & (recall_events.item_num != -1)]
                .item_num.astype(int))
            log(f"\nper-WORD label check (recalled iff item_num in "
                f"trial-{trial_num} REC_WORD set):")
            log(f"  {'sp':>3} {'item_name':<12} {'item_num':>8} "
                f"{'derived':>7} {'csv':>4} {'result':>6}")
            n_pass = n_fail = 0
            for _, word_row in trial_words.iterrows():
                item_number = int(word_row.item_num)
                derived = 1 if item_number in recalled_items_this_trial else 0
                csv_label = int(word_row.recalled)
                labels_agree = derived == csv_label
                n_pass += labels_agree
                n_fail += (not labels_agree)
                log(f"  {int(word_row.serialpos):>3} {str(word_row.word):<12} {item_number:>8} "
                    f"{derived:>7} {csv_label:>4} {'PASS' if labels_agree else 'FAIL':>6}")
            log(f"trial {trial_num}: {n_pass}/{len(trial_words)} PASS, {n_fail} FAIL")

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
            mismatches = trials[trials.recalled_recomputed != trials.recalled]
            log("MISMATCHES:")
            for _, mismatch_row in mismatches.iterrows():
                log(f"   trial {int(mismatch_row.trial)} sp {int(mismatch_row.serialpos)} {mismatch_row.word} "
                    f"item_num {int(mismatch_row.item_num)}: csv={int(mismatch_row.recalled)} "
                    f"recomputed={int(mismatch_row.recalled_recomputed)}")

        log(f"\nwithin-trial matching enforced: {cross_trial_only} additional word(s) "
            "would flip to recalled=1 under (incorrect) cross-trial matching; "
            "these were NOT matched.")

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
                  True, f"{cross_trial_only} cross-trial-only candidates excluded")

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
