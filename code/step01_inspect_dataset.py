#!/usr/bin/env python3
"""Inspect the schema of a locally-downloaded slice of ds004395 (PEERS).

Read-only. Discovers subjects, sessions, EEG files and *_events.tsv files under
data/ds004395, reports the events schema, reads one EEG file with MNE, and writes
the report to outputs/ds004395_inspection_report.txt. Column names are discovered
from the files, never assumed.

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

# Column roles aren't assumed; we surface any column whose name contains a hint.
WORD_HINTS = ["item_name", "item", "word", "stimulus", "stim_file", "stim", "probe"]
ONSET_HINTS = ["onset", "time", "sample", "latency"]
RECALL_HINTS = ["recall", "recalled", "recog", "remember", "memory",
                "correct", "intrusion", "resp", "acc"]
LIKELY_ENUM_COLS = ["trial_type", "type", "event", "item_name", "word",
                    "stimulus", "recalled", "serialpos", "list", "session", "onset"]
EEG_EXTS = [".edf", ".bdf", ".set", ".vhdr", ".fif", ".eeg", ".cnt", ".gdf"]
MAX_UNIQUE_PRINT = 50


def matches_filters(path, subject, session):
    """True if the path passes the optional subject/session filters."""
    if subject and f"/{subject}/" not in path + "/":
        return False
    if session and f"/{session}/" not in path + "/":
        return False
    return True


def find_subject_dirs(data_root, subject):
    return sorted(
        d for d in os.listdir(data_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(data_root, d))
        and (not subject or d == subject)
    )


def find_session_dirs(data_root, subjects, session):
    sessions = []
    for sub in subjects:
        for d in sorted(os.listdir(os.path.join(data_root, sub))):
            if d.startswith("ses-") and os.path.isdir(os.path.join(data_root, sub, d)) \
                    and (not session or d == session):
                sessions.append(f"{sub}/{d}")
    return sessions


def find_eeg_files(data_root, subject, session):
    files = []
    for ext in EEG_EXTS:
        files += glob.glob(os.path.join(data_root, "**", f"*{ext}"), recursive=True)
    return sorted(f for f in files if matches_filters(f, subject, session))


def find_event_files(data_root, subject, session):
    files = glob.glob(os.path.join(data_root, "**", "*_events.tsv"), recursive=True)
    return sorted(f for f in files if matches_filters(f, subject, session))


def columns_matching(columns, hints):
    """Column names containing any of the hint substrings (case-insensitive)."""
    return [c for c in columns if any(h in c.lower() for h in hints)]


def report_unique_values(log, df, col):
    """Log a column's unique values, capping high-cardinality columns."""
    series = df[col]
    n = series.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(series) and n > MAX_UNIQUE_PRINT:
        log(f"  [{col}] numeric, {n} unique | "
            f"min={series.min()} max={series.max()} "
            f"mean={series.mean():.3f} (continuous — not enumerating)")
        return
    uniques = series.dropna().unique().tolist()
    if len(uniques) > MAX_UNIQUE_PRINT:
        log(f"  [{col}] {n} unique (showing first {MAX_UNIQUE_PRINT}):")
        log(f"      {uniques[:MAX_UNIQUE_PRINT]}")
    else:
        na = int(series.isna().sum())
        log(f"  [{col}] {n} unique (NaN rows: {na}):")
        log(f"      {sorted(map(str, uniques))}")


def report_events_schema(log, path):
    """Report one events.tsv: columns, sample rows, unique values, role candidates."""
    log.rule(f"EVENTS FILE: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True, na_values=["n/a", ""])
    # A second read with inferred dtypes, so numeric stats work; the string read is for display.
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
        report_unique_values(log, df_typed, c)

    # Also surface any small-cardinality categorical column we didn't list explicitly.
    for c in df_typed.columns:
        if c not in present_enum and df_typed[c].nunique(dropna=True) <= 15 \
                and not pd.api.types.is_float_dtype(df_typed[c]):
            report_unique_values(log, df_typed, c)

    log("\n-- column-role candidates (heuristic, names not assumed) --")
    log(f"  possible WORD / STIMULUS columns : {columns_matching(df.columns, WORD_HINTS)}")
    log(f"  possible ONSET / TIME columns    : {columns_matching(df.columns, ONSET_HINTS)}")
    log(f"  possible RECALL / MEMORY columns : {columns_matching(df.columns, RECALL_HINTS)}")

    # Recall can also live in trial_type values (e.g. REC_WORD), not only in column names.
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


def report_eeg_metadata(log, eeg_path):
    """Read one EEG file with MNE and report sampling rate, channels, and duration."""
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

        log.rule("STRUCTURE")
        if args.subject or args.session:
            log(f"filters -> subject={args.subject} session={args.session}")

        subjects = find_subject_dirs(args.data_root, args.subject)
        log(f"subject folders ({len(subjects)}): {subjects}")

        sessions = find_session_dirs(args.data_root, subjects, args.session)
        log(f"session folders ({len(sessions)}): {sessions}")

        eeg_files = find_eeg_files(args.data_root, args.subject, args.session)
        log(f"\nEEG data files ({len(eeg_files)}):")
        for f in eeg_files:
            log(f"  {os.path.getsize(f)/1e6:8.1f} MB  {os.path.relpath(f, args.data_root)}")

        event_files = find_event_files(args.data_root, args.subject, args.session)
        log(f"\nevents.tsv files ({len(event_files)}):")
        for f in event_files:
            log(f"  {os.path.relpath(f, args.data_root)}")

        if not event_files:
            log("\nNo *_events.tsv files found — nothing to inspect.")
        for ef in event_files:
            report_events_schema(log, ef)

        if eeg_files:
            report_eeg_metadata(log, eeg_files[0])
        else:
            log.rule("EEG FILE (MNE read)")
            log("No EEG binary present locally (sidecars only?). "
                "Download one with code/step03_download_session.py to read it with MNE.")

        log.rule("END OF REPORT")
        log(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()
