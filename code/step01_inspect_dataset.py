#!/usr/bin/env python3
"""
Inspect the schema of a locally-downloaded slice of ds004395 (PEERS).

This is a READ-ONLY inspection stage. It does NOT:
  - train any model / ridge regression
  - extract EEG windows
  - fabricate recall / recalled labels
  - assume column names (all columns are discovered from the files themselves)

It looks inside data/ds004395, finds subjects / sessions / EEG files /
*_events.tsv files, reports the events schema, tries to read one EEG file
with MNE (sampling rate + channel count), and writes everything to
outputs/ds004395_inspection_report.txt.

Usage:
    python code/step01_inspect_dataset.py
    python code/step01_inspect_dataset.py --data-root data/ds004395 --out outputs/ds004395_inspection_report.txt
"""

import argparse
import glob
import os
import sys

import pandas as pd

from common import Tee

# ---------------------------------------------------------------------------
# Column-name heuristics. We do NOT assume a fixed schema; we look for columns
# whose names *contain* these hints (case-insensitive) and report candidates.
# ---------------------------------------------------------------------------
WORD_HINTS = ["item_name", "item", "word", "stimulus", "stim_file", "stim", "probe"]
ONSET_HINTS = ["onset", "time", "sample", "latency"]
RECALL_HINTS = ["recall", "recalled", "recog", "remember", "memory",
                "correct", "intrusion", "resp", "acc"]

# Columns the task specifically asked to enumerate unique values for
# (only those actually present are used).
LIKELY_ENUM_COLS = ["trial_type", "type", "event", "item_name", "word",
                    "stimulus", "recalled", "serialpos", "list", "session", "onset"]

EEG_EXTS = [".edf", ".bdf", ".set", ".vhdr", ".fif", ".eeg", ".cnt", ".gdf"]
MAX_UNIQUE_PRINT = 50


def find_candidates(columns, hints):
    cols_l = {c: c.lower() for c in columns}
    found = []
    for c in columns:
        if any(h in cols_l[c] for h in hints):
            found.append(c)
    return found


def describe_unique(log, df, col):
    """Print unique values for a column, capped for high-cardinality columns."""
    series = df[col]
    nun = series.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(series) and nun > MAX_UNIQUE_PRINT:
        log(f"  [{col}] numeric, {nun} unique | "
            f"min={series.min()} max={series.max()} "
            f"mean={series.mean():.3f} (continuous — not enumerating)")
        return
    uniques = series.dropna().unique().tolist()
    if len(uniques) > MAX_UNIQUE_PRINT:
        log(f"  [{col}] {nun} unique (showing first {MAX_UNIQUE_PRINT}):")
        log(f"      {uniques[:MAX_UNIQUE_PRINT]}")
    else:
        # include NaN count for completeness
        na = int(series.isna().sum())
        log(f"  [{col}] {nun} unique (NaN rows: {na}):")
        log(f"      {sorted(map(str, uniques))}")


def inspect_events_file(log, path):
    log.rule(f"EVENTS FILE: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True, na_values=["n/a", ""])
    # Re-read letting pandas infer numerics for stats, but keep string for display safety.
    df_typed = pd.read_csv(path, sep="\t", na_values=["n/a", ""])

    log(f"rows: {len(df)}   columns: {len(df.columns)}")
    log(f"columns: {list(df.columns)}")

    log("\n-- first 30 rows --")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.max_colwidth", 24):
        log(df.head(30).to_string(index=False))

    log("\n-- unique values for likely/enumerable columns --")
    present_enum = [c for c in LIKELY_ENUM_COLS if c in df_typed.columns]
    log(f"(present from requested list: {present_enum})")
    for c in present_enum:
        describe_unique(log, df_typed, c)

    # Also enumerate any obvious categorical trial-type-like column not in the list.
    for c in df_typed.columns:
        if c not in present_enum and df_typed[c].nunique(dropna=True) <= 15 \
                and not pd.api.types.is_float_dtype(df_typed[c]):
            describe_unique(log, df_typed, c)

    log("\n-- column-role candidates (heuristic, names not assumed) --")
    word_cands = find_candidates(df.columns, WORD_HINTS)
    onset_cands = find_candidates(df.columns, ONSET_HINTS)
    recall_cands = find_candidates(df.columns, RECALL_HINTS)
    log(f"  possible WORD / STIMULUS columns : {word_cands}")
    log(f"  possible ONSET / TIME columns    : {onset_cands}")
    log(f"  possible RECALL / MEMORY columns : {recall_cands}")

    # Recall signal can live in trial_type VALUES (e.g. REC_WORD), not just columns.
    for tt_col in ("trial_type", "type", "event"):
        if tt_col in df_typed.columns:
            vals = [str(v) for v in df_typed[tt_col].dropna().unique()]
            rec_like = [v for v in vals if "REC" in v.upper() or "RECALL" in v.upper()]
            if rec_like:
                log(f"  recall-related '{tt_col}' VALUES (outcome derivable, "
                    f"NOT a ready label): {rec_like}")
    log("  NOTE: no assumption is made that any single column is a recall label; "
        "free-recall success must be DERIVED (match presented WORD events to "
        "REC_WORD events per trial). No labels are fabricated here.")
    return df_typed


def inspect_eeg_with_mne(log, eeg_path):
    log.rule(f"EEG FILE (MNE read): {eeg_path}")
    ext = os.path.splitext(eeg_path)[1].lower()
    log(f"detected EEG format: {ext}")
    try:
        import mne
        log(f"mne version: {mne.__version__}")
        readers = {
            ".edf": mne.io.read_raw_edf,
            ".bdf": mne.io.read_raw_bdf,
            ".set": mne.io.read_raw_eeglab,
            ".vhdr": mne.io.read_raw_brainvision,
            ".fif": mne.io.read_raw_fif,
            ".cnt": mne.io.read_raw_cnt,
            ".gdf": mne.io.read_raw_gdf,
        }
        reader = readers.get(ext)
        if reader is None:
            log(f"  no MNE reader mapped for extension {ext}; skipping read.")
            return
        raw = reader(eeg_path, preload=False, verbose="ERROR")
        info = raw.info
        log(f"  sampling rate (sfreq) : {info['sfreq']} Hz")
        log(f"  channel count         : {len(raw.ch_names)}")
        log(f"  n_times               : {raw.n_times}")
        log(f"  duration              : {raw.n_times / info['sfreq']:.1f} s")
        log(f"  highpass / lowpass    : {info['highpass']} / {info['lowpass']} Hz")
        # channel type breakdown
        types = {}
        for t in raw.get_channel_types():
            types[t] = types.get(t, 0) + 1
        log(f"  channel types         : {types}")
        log(f"  first 10 channels     : {raw.ch_names[:10]}")
    except Exception as e:  # noqa: BLE001 - report, don't crash the inspection
        log(f"  MNE read FAILED: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data-root", default=os.path.join(here, "data", "ds004395"))
    ap.add_argument("--out", default=os.path.join(here, "outputs", "ds004395_inspection_report.txt"))
    ap.add_argument("--subject", default=None,
                    help="Restrict inspection to this subject folder, e.g. sub-LTP327 (default: all present).")
    ap.add_argument("--session", default=None,
                    help="Restrict inspection to this session, e.g. ses-4 (default: all present).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as fh:
        log = Tee(fh)
        log.rule("ds004395 (PEERS) INSPECTION REPORT")
        log(f"data root: {args.data_root}")
        log(f"cwd      : {os.getcwd()}")

        if not os.path.isdir(args.data_root):
            log(f"ERROR: data root not found: {args.data_root}")
            log("Download one session first, e.g.:")
            log("  python code/step03_download_session.py --sub LTP063 --ses 15")
            sys.exit(1)

        # --- structure discovery ------------------------------------------
        log.rule("STRUCTURE")
        if args.subject or args.session:
            log(f"filters -> subject={args.subject} session={args.session}")

        def keep(path):
            """Apply optional subject/session filters to a path."""
            if args.subject and f"/{args.subject}/" not in path + "/":
                return False
            if args.session and f"/{args.session}/" not in path + "/":
                return False
            return True

        subjects = sorted(
            d for d in os.listdir(args.data_root)
            if d.startswith("sub-") and os.path.isdir(os.path.join(args.data_root, d))
            and (not args.subject or d == args.subject)
        )
        log(f"subject folders ({len(subjects)}): {subjects}")

        sessions = []
        for sub in subjects:
            for d in sorted(os.listdir(os.path.join(args.data_root, sub))):
                if d.startswith("ses-") and os.path.isdir(os.path.join(args.data_root, sub, d)) \
                        and (not args.session or d == args.session):
                    sessions.append(f"{sub}/{d}")
        log(f"session folders ({len(sessions)}): {sessions}")

        eeg_files = []
        for ext in EEG_EXTS:
            eeg_files += glob.glob(os.path.join(args.data_root, "**", f"*{ext}"), recursive=True)
        eeg_files = sorted(f for f in eeg_files if keep(f))
        log(f"\nEEG data files ({len(eeg_files)}):")
        for f in eeg_files:
            log(f"  {os.path.getsize(f)/1e6:8.1f} MB  {os.path.relpath(f, args.data_root)}")

        event_files = sorted(f for f in glob.glob(
            os.path.join(args.data_root, "**", "*_events.tsv"), recursive=True) if keep(f))
        log(f"\nevents.tsv files ({len(event_files)}):")
        for f in event_files:
            log(f"  {os.path.relpath(f, args.data_root)}")

        # --- events schema -------------------------------------------------
        if not event_files:
            log("\nNo *_events.tsv files found — nothing to inspect.")
        for ef in event_files:
            inspect_events_file(log, ef)

        # --- EEG via MNE ---------------------------------------------------
        if eeg_files:
            inspect_eeg_with_mne(log, eeg_files[0])
        else:
            log.rule("EEG FILE (MNE read)")
            log("No EEG binary present locally (sidecars only?). "
                "Download one with code/step03_download_session.py to read it with MNE.")

        log.rule("END OF REPORT")
        log(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()
