#!/usr/bin/env python3
"""
Build the encoding-trials table for one ds004395 (PEERS) session.

One row per WORD-presentation (encoding) event, with a DERIVED free-recall
label. This stage does NOT extract EEG, does NOT train anything, and does NOT
use recognition (RECOG_*) events or `recog_resp` as the recall label.

Recall derivation (free recall only):
    recalled = 1  if a REC_WORD event exists in the SAME trial with the SAME
                  item_num as the presented WORD (item_name used as a backup
                  validation key).
    recalled = 0  otherwise.

No labels are fabricated: recall is derived purely from the presence/absence of
matching REC_WORD events within the same trial.

Output:
    outputs/encoding_trials.csv
    outputs/encoding_trials_summary.txt

Columns:
    subject, session, trial, serialpos, word, item_num, onset, sample,
    recalled, eeg_file, event_file
"""

import argparse
import os
import sys

import pandas as pd

from common import Tee, peers_word_set

WORD = "WORD"
REC_WORD = "REC_WORD"
OUT_COLS = ["subject", "session", "trial", "serialpos", "word", "item_num",
            "onset", "sample", "recalled", "eeg_file", "event_file"]


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__)
    # Canonical session going forward = ltpFR2 sub-LTP269/ses-20:
    # 576 words, 100% peers coverage, 576/576 valid 300-800 ms EEG windows
    # (full recording, EDF 4741 s). Earlier sessions are archived and must not
    # be used: sub-LTP063 (ltpFR, low coverage) and sub-LTP293/ses-3 (ltpFR2 but
    # TRUNCATED EEG, only 449/576 windows fit) under outputs/archive_*/.
    ap.add_argument("--events", default=os.path.join(
        here, "data/ds004395/sub-LTP269/ses-20/eeg/"
        "sub-LTP269_ses-20_task-ltpFR2_events.tsv"))
    ap.add_argument("--eeg", default=os.path.join(
        here, "data/ds004395/sub-LTP269/ses-20/eeg/"
        "sub-LTP269_ses-20_task-ltpFR2_eeg.edf"))
    ap.add_argument("--peers-order", default=os.path.join(here, "results/embeddings/peers_word_order.csv"))
    ap.add_argument("--out-csv", default=os.path.join(here, "outputs/encoding_trials.csv"))
    ap.add_argument("--out-summary", default=os.path.join(
        here, "outputs/encoding_trials_summary.txt"))
    ap.add_argument("--win-start", type=float, default=0.300,
                    help="EEG window start relative to word onset, in seconds.")
    ap.add_argument("--win-stop", type=float, default=0.800,
                    help="EEG window stop relative to word onset, in seconds.")
    args = ap.parse_args()

    # Hard stop if the event file is missing.
    if not os.path.isfile(args.events):
        sys.exit(f"ERROR: event file not found: {args.events}")

    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)

    with open(args.out_summary, "w") as fh:
        log = Tee(fh)
        log.rule("ENCODING-TRIALS BUILD + VALIDATION")
        log(f"event file : {args.events}")
        log(f"eeg file   : {args.eeg}")
        if not os.path.isfile(args.eeg):
            log.warn("EEG file not found on disk (path still recorded in output)",
                     args.eeg)

        # -----------------------------------------------------------------
        # Load events
        # -----------------------------------------------------------------
        ev = pd.read_csv(args.events, sep="\t", na_values=["n/a", ""])
        for col in ("trial_type", "trial", "item_name", "item_num", "onset", "sample"):
            if col not in ev.columns:
                sys.exit(f"ERROR: expected column '{col}' missing from {args.events}")

        word_ev = ev[ev.trial_type == WORD].copy()
        rec_ev = ev[ev.trial_type == REC_WORD].copy()

        log.rule("COUNTS")
        log(f"WORD events     : {len(word_ev)}")
        log(f"REC_WORD events : {len(rec_ev)}")

        # -----------------------------------------------------------------
        # Missing item_num reporting (do NOT silently drop)
        # -----------------------------------------------------------------
        w_missing = word_ev[word_ev.item_num.isna()]
        r_missing = rec_ev[rec_ev.item_num.isna()]
        if len(w_missing):
            log.warn(f"{len(w_missing)} WORD events have missing item_num",
                     f"onsets={w_missing.onset.tolist()[:10]}")
        else:
            log("all WORD events have item_num.")
        if len(r_missing):
            log.warn(f"{len(r_missing)} REC_WORD events have missing item_num "
                     "(cannot be used for matching)",
                     f"onsets={r_missing.onset.tolist()[:10]}")
        else:
            log("all REC_WORD events have item_num.")

        # Intrusions in recall (item_num == -1 per events.json) never match a WORD.
        n_intrusion = int((rec_ev.item_num == -1).sum())
        if n_intrusion:
            log(f"note: {n_intrusion} REC_WORD events are intrusions/vocalizations "
                "(item_num == -1); these never match a presented WORD.")

        # -----------------------------------------------------------------
        # Build the recall lookup.
        #
        # The key is (trial, item_num), not item_num alone. ltpFR2 reuses the
        # same 576-word pool across trials, so a bare item_num would mark a word
        # recalled in list 3 as recalled in list 12 as well -- inflating the
        # recall rate and corrupting the outcome variable.
        #
        # item_num == -1 flags an intrusion or vocalization (per events.json):
        # the subject said something that was not a presented word. Dropped
        # here, since by definition it cannot mark any studied word as recalled.
        # -----------------------------------------------------------------
        rec_valid = rec_ev.dropna(subset=["trial", "item_num"])
        rec_valid = rec_valid[rec_valid.item_num != -1]
        recalled_keys = set(
            zip(rec_valid.trial.astype(int), rec_valid.item_num.astype(int)))
        # Kept only to cross-check the numeric join against the recorded word
        # text. setdefault keeps the first mention: a word recalled twice in one
        # list is still just recalled.
        rec_name_by_key = {}
        for _, r in rec_valid.iterrows():
            rec_name_by_key.setdefault(
                (int(r.trial), int(r.item_num)), str(r.item_name))

        # -----------------------------------------------------------------
        # Build one encoding trial per WORD event
        # -----------------------------------------------------------------
        # serialpos = position within its list, 1-based, ordered by onset.
        # Stable sort so that events sharing an onset keep their file order
        # rather than being permuted arbitrarily.
        word_ev = word_ev.sort_values(["trial", "onset"], kind="stable")
        word_ev["serialpos"] = (
            word_ev.groupby("trial").cumcount() + 1)

        name_mismatches = []
        rows = []
        for _, w in word_ev.iterrows():
            trial = int(w.trial) if pd.notna(w.trial) else None
            inum = int(w.item_num) if pd.notna(w.item_num) else None
            key = (trial, inum)
            recalled = 1 if (trial is not None and inum is not None
                             and key in recalled_keys) else 0

            # Backup validation: if matched by item_num, item_name should agree.
            if recalled and key in rec_name_by_key:
                if str(w.item_name).upper() != rec_name_by_key[key].upper():
                    name_mismatches.append(
                        (trial, inum, str(w.item_name), rec_name_by_key[key]))

            rows.append({
                "subject": w.subject if "subject" in word_ev.columns else None,
                "session": int(w.session) if pd.notna(w.session) else None,
                "trial": trial,
                "serialpos": int(w.serialpos),
                "word": w.item_name,          # preserve UPPERCASE exactly
                "item_num": inum,
                "onset": w.onset,
                "sample": int(w["sample"]) if pd.notna(w["sample"]) else None,
                "recalled": recalled,
                "eeg_file": os.path.relpath(args.eeg, here),
                "event_file": os.path.relpath(args.events, here),
            })

        df = pd.DataFrame(rows, columns=OUT_COLS)

        # -----------------------------------------------------------------
        # Summary stats
        # -----------------------------------------------------------------
        log.rule("RECALL SUMMARY")
        n = len(df)
        n_rec = int((df.recalled == 1).sum())
        n_forg = int((df.recalled == 0).sum())
        log(f"encoding trials (rows) : {n}")
        log(f"recalled (1)           : {n_rec}")
        log(f"forgotten (0)          : {n_forg}")
        log(f"recall rate            : {n_rec / n:.4f}" if n else "recall rate: n/a")

        trials = sorted(df.trial.dropna().unique().tolist())
        log(f"unique trials/lists    : {len(trials)}  -> {trials}")
        per_trial = df.groupby("trial").size()
        log("words per trial:")
        for t, c in per_trial.items():
            log(f"   trial {int(t):>3} : {c} words")
        if per_trial.nunique() > 1:
            log.warn("trials do NOT all have the same word count",
                     f"counts={sorted(per_trial.unique().tolist())}")

        log.rule("FIRST 20 ROWS")
        with pd.option_context("display.max_columns", None, "display.width", 240,
                               "display.max_colwidth", 40):
            log(df.head(20).to_string(index=False))

        # -----------------------------------------------------------------
        # Validation checks
        # -----------------------------------------------------------------
        log.rule("VALIDATION CHECKS")

        log.check("no missing word", df.word.notna().all(),
                  f"{int(df.word.isna().sum())} missing")
        log.check("no missing onset", df.onset.notna().all(),
                  f"{int(df.onset.isna().sum())} missing")
        log.check("no missing sample", df["sample"].notna().all(),
                  f"{int(df['sample'].isna().sum())} missing")

        # EEG-timing validity: onset/sample can be present but INVALID sentinels
        # (onset<=0, sample<0 == -1) meaning the word was presented but NOT
        # captured/synced in this EDF. Critical for later EEG window extraction.
        valid_timing = (df.onset > 0) & (df["sample"] >= 0)
        n_valid = int(valid_timing.sum())
        log.check("all WORD events have VALID EEG timing (onset>0 & sample>=0)",
                  valid_timing.all(),
                  f"{n_valid}/{len(df)} valid ({100*valid_timing.mean():.1f}%); "
                  f"{len(df)-n_valid} rows have sentinel onset<=0/sample<0")
        if not valid_timing.all():
            bad = df[~valid_timing]
            log.warn("rows with invalid EEG timing cannot be aligned to EEG "
                     "(word presented but not captured/synced in this EDF)",
                     f"trials affected: {sorted(bad.trial.unique().tolist())}")

        # EEG WINDOW-FIT check (authoritative, via MNE): the full
        # [win_start, win_stop] window must fit inside the actual recording.
        # start_sample = sample + int(win_start*sfreq)
        # stop_sample  = sample + int(win_stop*sfreq)
        # valid iff sample>=0 & onset>0 & start_sample>=0 & stop_sample < n_times.
        if os.path.isfile(args.eeg):
            try:
                import mne
                raw = mne.io.read_raw_edf(args.eeg, preload=False, verbose="ERROR")
                sfreq = float(raw.info["sfreq"])
                n_times = int(raw.n_times)
                start_s = df["sample"] + int(args.win_start * sfreq)
                stop_s = df["sample"] + int(args.win_stop * sfreq)
                win_ok = (df["sample"] >= 0) & (df.onset > 0) & (start_s >= 0) \
                    & (stop_s < n_times)
                n_win = int(win_ok.sum())
                log.check(
                    f"all WORD events have a full EEG window "
                    f"[{int(args.win_start*1000)}-{int(args.win_stop*1000)}ms] "
                    f"inside the recording (n_times={n_times}, sfreq={sfreq:g})",
                    bool(win_ok.all()),
                    f"{n_win}/{len(df)} fit; {len(df)-n_win} exceed EDF length")
                if not win_ok.all():
                    bad = df[~win_ok].sort_values(["trial", "serialpos"])
                    first = bad.iloc[0]
                    log.warn("session has words whose EEG window falls OUTSIDE the "
                             "recording -> NOT valid for canonical EEG extraction",
                             f"first bad: trial {int(first.trial)} "
                             f"serialpos {int(first.serialpos)} word {first.word!r}; "
                             f"trials affected: {sorted(bad.trial.unique().tolist())}")
            except Exception as e:  # noqa: BLE001
                log.warn("could not run MNE window-fit check", f"{type(e).__name__}: {e}")
        else:
            log.warn("EEG file absent; skipped MNE window-fit check", args.eeg)

        log.check("recalled is only 0 or 1",
                  set(df.recalled.unique()) <= {0, 1},
                  f"values={sorted(df.recalled.unique().tolist())}")

        # Duplicate subject/session/trial/serialpos
        dup_mask = df.duplicated(["subject", "session", "trial", "serialpos"], keep=False)
        log.check("no duplicate subject/session/trial/serialpos",
                  not dup_mask.any(), f"{int(dup_mask.sum())} duplicate rows")
        if dup_mask.any():
            log(df[dup_mask].to_string(index=False))

        # item_name backup validation
        log.check("recalled matches agree on item_name (backup key)",
                  len(name_mismatches) == 0,
                  f"{len(name_mismatches)} mismatches")
        for m in name_mismatches[:10]:
            log(f"    trial {m[0]} item_num {m[1]}: WORD={m[2]!r} REC_WORD={m[3]!r}")

        # Every word exists in peers_word_order.csv
        log.rule("PEERS WORD-LIST CROSS-CHECK")
        if os.path.isfile(args.peers_order):
            peers = peers_word_set(args.peers_order)
            session_words = df.word.str.upper()
            in_peers = session_words.isin(peers)
            missing_words = sorted(set(session_words[~in_peers]))
            log(f"peers list size            : {len(peers)}")
            log(f"encoding rows              : {len(df)}")
            log(f"rows whose word IS in peers: {int(in_peers.sum())} "
                f"({100*in_peers.mean():.1f}%)")
            log(f"unique session words       : {session_words.nunique()}")
            log(f"unique words NOT in peers  : {len(missing_words)}")
            if missing_words:
                log.warn(
                    "not every word is in peers_word_order.csv",
                    "EXPECTED for task-ltpFR: this session uses the ~1638-word "
                    "wasnorm_wordpool, while peers_word_order.csv is the 576-word "
                    "(ltpFR2) pool. No rows dropped; reporting only.")
                log(f"    examples not in peers: {missing_words[:15]}")
            else:
                log.check("every word exists in peers_word_order.csv", True)
        else:
            log.warn("peers_word_order.csv not found; skipped word cross-check",
                     args.peers_order)

        # -----------------------------------------------------------------
        # Write CSV
        # -----------------------------------------------------------------
        df.to_csv(args.out_csv, index=False)
        log.rule("RESULT")
        log(f"wrote {len(df)} rows -> {args.out_csv}")
        log(f"summary -> {args.out_summary}")
        log(f"errors: {log.fail}   warnings: {log.warnings}")
        if log.fail:
            log("STATUS: FAILED validation (see [FAIL] lines above).")
        else:
            log("STATUS: OK (warnings are informational; no rows dropped).")

    sys.exit(1 if log.fail else 0)


if __name__ == "__main__":
    main()
