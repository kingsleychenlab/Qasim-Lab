#!/usr/bin/env python3
"""
Find a task-ltpFR2 session in ds004395 that matches the 576-word PEERS T5
embedding list, download just one such session, and run the inspection and
encoding-trials stages on it.

The reason: task-ltpFR uses the ~1638-word wasnorm_wordpool (only ~35% of a
session's words are in peers_word_order.csv), whereas task-ltpFR2 (PEERS4) uses
the 576-word pool that the T5 embeddings were built from (expected ~100%
coverage).

This script:
  1. Lists ltpFR2 subject/session paths from OpenNeuro's public S3 listing
     (metadata only, no download of the full dataset).
  2. Prints candidate sessions with EEG/events sizes (smallest first).
  3. Downloads one ltpFR2 session (default: smallest EDF; override with --sub/--ses).
  4. Runs code/step01_inspect_dataset.py on that session.
  5. Runs code/step05_create_encoding_trials.py on that session.
  6. Checks word coverage against peers_word_order.csv and reports.

It doesn't create X_eeg.npy / Y_t5.npy and doesn't train anything.
"""

import argparse
import collections
import io
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from common import peers_word_set

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds004395"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def scan_task_sessions(task):
    """Page the whole dataset listing; collect eeg/events keys for `task`."""
    token = None
    sessions = collections.defaultdict(dict)
    task_tag = f"task-{task}"
    pages = 0
    while True:
        params = {"list-type": "2", "prefix": f"{DATASET}/"}
        if token:
            params["continuation-token"] = token
        response = requests.get(BUCKET, params=params, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        pages += 1
        for content in root.findall(f"{NS}Contents"):
            object_key = content.find(f"{NS}Key").text
            if task_tag not in object_key:
                continue
            size = int(content.find(f"{NS}Size").text)
            parts = object_key.split("/")
            if len(parts) < 3:
                continue
            sub_ses = (parts[1], parts[2])  # (sub-XXX, ses-Y)
            if object_key.endswith("_eeg.edf"):
                sessions[sub_ses]["edf"] = size
            elif object_key.endswith("_events.tsv"):
                sessions[sub_ses]["events"] = size
        truncated = root.find(f"{NS}IsTruncated")
        if truncated is not None and truncated.text == "true":
            token = root.find(f"{NS}NextContinuationToken").text
        else:
            break
    return sessions, pages


def events_url(sub, ses, task):
    return f"{BUCKET}/{DATASET}/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_events.tsv"


def edf_url(sub, ses, task):
    return f"{BUCKET}/{DATASET}/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_eeg.edf"


def edf_header_ntimes(url, edf_size):
    """Parse n_times and sfreq from just the EDF header via a small Range request.
    I checked that this matches mne.io.read_raw_edf(...).n_times exactly. Saves
    downloading the full (100s of MB) EDF just to learn its length."""
    response = requests.get(url, headers={"Range": "bytes=0-49999"}, timeout=60)
    response.raise_for_status()
    header_bytes = response.content

    def field(a, z):
        return header_bytes[a:z].decode("ascii", "ignore").strip()

    n_records = int(field(236, 244) or "-1")
    rec_dur = float(field(244, 252))
    n_signals = int(field(252, 256))
    header_nbytes = int(field(184, 192))
    signal_block_start = 256 + n_signals * 216  # start of "samples per record" block
    samples_per_record = [int(field(signal_block_start + i * 8,
                                    signal_block_start + i * 8 + 8))
                          for i in range(n_signals)]
    if n_records <= 0:  # EDF+ may store -1; recover from file size
        n_records = (edf_size - header_nbytes) // (2 * sum(samples_per_record))
    n_times = n_records * samples_per_record[0]
    sfreq = samples_per_record[0] / rec_dur
    return n_times, sfreq


def peek_session(sub, ses, task, peers, edf_size, win_start, win_stop):
    """Fetch events.tsv (tiny) + EDF header (tiny) and compute full window
    validity: a WORD is a valid EEG window only if
        sample>=0 and onset>0 and start_sample>=0 and stop_sample < n_times."""
    response = requests.get(events_url(sub, ses, task), timeout=60)
    if response.status_code != 200:
        return None
    events = pd.read_csv(io.StringIO(response.text), sep="\t", na_values=["n/a", ""])
    word_events = events[events.trial_type == "WORD"].copy()
    rec_events = events[events.trial_type == "REC_WORD"]
    words = word_events.item_name.dropna().str.upper()
    coverage = float(words.isin(peers).mean()) if len(words) else 0.0

    try:
        n_times, sfreq = edf_header_ntimes(edf_url(sub, ses, task), edf_size)
    except Exception as e:  # noqa: BLE001
        return {"n_word": len(word_events), "n_rec": len(rec_events), "coverage": coverage,
                "n_valid_win": 0, "n_times": None, "sfreq": None,
                "err": f"{type(e).__name__}: {e}"}

    start_sample = word_events["sample"] + int(win_start * sfreq)
    stop_sample = word_events["sample"] + int(win_stop * sfreq)
    valid_windows = ((word_events["sample"] >= 0) & (word_events.onset > 0)
                     & (start_sample >= 0) & (stop_sample < n_times))
    return {"n_word": len(word_events), "n_rec": len(rec_events), "coverage": coverage,
            "n_valid_win": int(valid_windows.sum()), "n_times": int(n_times),
            "sfreq": float(sfreq), "err": None}


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def norm_label(v, prefix):
    """Accept '327' / 'LTP327' / 'sub-LTP327' -> 'sub-LTP327' (prefix='sub-LTP')."""
    if v is None:
        return None
    v = str(v)
    if v.startswith("sub-") or v.startswith("ses-"):
        return v
    if prefix == "sub-" :
        return "sub-" + (v if v.upper().startswith("LTP") else "LTP" + v)
    return "ses-" + v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="ltpFR2")
    parser.add_argument("--peers-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    parser.add_argument("--sub", help="Force a subject label, e.g. LTP327 or sub-LTP327")
    parser.add_argument("--ses", help="Force a session label, e.g. 4 or ses-4")
    parser.add_argument("--top", type=int, default=12, help="How many candidates to list")
    parser.add_argument("--win-start", type=float, default=0.300,
                        help="EEG window start relative to word onset, in seconds.")
    parser.add_argument("--win-stop", type=float, default=0.800,
                        help="EEG window stop relative to word onset, in seconds.")
    parser.add_argument("--list-only", action="store_true",
                        help="Only scan + list candidates; do not download or run stages.")
    args = parser.parse_args()
    if bool(args.sub) != bool(args.ses):
        parser.error("--sub and --ses must be given together (or neither).")

    peers = peers_word_set(args.peers_order)
    print(f"peers_word_order.csv: {len(peers)} words")

    # Scan + list candidates
    print(f"\nScanning OpenNeuro S3 listing for task-{args.task} sessions "
          "(metadata only, no bulk download)...")
    sessions, pages = scan_task_sessions(args.task)
    subjects = sorted({sub for sub, _ in sessions})
    print(f"scanned {pages} listing pages; task-{args.task}: "
          f"{len(subjects)} subjects, {len(sessions)} sessions")
    if not sessions:
        sys.exit(f"No task-{args.task} sessions found.")

    sessions_with_edf = [(k, info) for k, info in sessions.items() if "edf" in info]
    sessions_with_edf.sort(key=lambda x: x[1]["edf"])
    print(f"\nSmallest {min(args.top, len(sessions_with_edf))} task-{args.task} sessions by EDF size:")
    print(f"  {'EDF (MB)':>9}  {'events(MB)':>10}  subject/session")
    for (sub, ses), session_info in sessions_with_edf[:args.top]:
        print(f"  {session_info['edf']/1e6:9.1f}  {session_info.get('events', 0)/1e6:10.3f}  {sub}/{ses}")

    if args.list_only:
        return

    # Choose target and verify coverage on the tiny events.tsv first
    if args.sub and args.ses:
        target = (norm_label(args.sub, "sub-"), norm_label(args.ses, "ses-"))
        if target not in sessions:
            sys.exit(f"Requested {target[0]}/{target[1]} is not a task-{args.task} session.")
    else:
        # A session is only acceptable if all 576 WORD events yield a full
        # [win_start, win_stop] EEG window that fits inside the EDF. Small or
        # truncated EDFs fail because their later word windows fall off the end.
        NEED = 576
        target = None
        print(f"\nEvaluating candidates (need WORD=576, cov=576/576, "
              f"valid {int(args.win_start*1000)}-{int(args.win_stop*1000)}ms windows=576/576):")
        print(f"  {'subject/session':22} {'WORD':>4} {'cov':>4} {'valid_win':>9} {'edf_s':>6}  pick?")
        for (sub, ses), session_info in sessions_with_edf:
            candidate = peek_session(sub, ses, args.task, peers, session_info["edf"],
                                     args.win_start, args.win_stop)
            if not candidate:
                continue
            qualifies = (candidate["n_word"] == NEED and candidate["coverage"] >= 0.999
                         and candidate["n_valid_win"] == NEED)
            edf_seconds = f"{candidate['n_times']/candidate['sfreq']:.0f}" if candidate["n_times"] else "err"
            print(f"  {sub}/{ses:8} {candidate['n_word']:4} {int(round(candidate['coverage']*candidate['n_word'])):4} "
                  f"{candidate['n_valid_win']:9} {edf_seconds:>6}  {'<== SELECT' if qualifies else ''}")
            if qualifies:
                target = (sub, ses)
                selected = (session_info, candidate)
                break  # smallest qualifying EDF; stop probing
        if target is None:
            sys.exit("No task-ltpFR2 session with 576/576 valid EEG windows found "
                     "in the candidate list (try increasing --top).")
        sub, ses = target
        print(f"\nSelected: {sub}/{ses} (EDF {selected[0]['edf']/1e6:.1f} MB, "
              f"coverage 576/576, valid windows {selected[1]['n_valid_win']}/576, "
              f"EDF {selected[1]['n_times']/selected[1]['sfreq']:.0f}s)")

    sub, ses = target
    sub_label = sub.replace("sub-", "")
    ses_label = ses.replace("ses-", "")
    tag = f"{sub}_{ses}"

    # Download this 1 session
    run([PY, os.path.join(HERE, "code", "step03_download_session.py"),
         "--sub", sub_label, "--ses", ses_label, "--task", args.task])

    eeg = os.path.join(HERE, "data", DATASET, sub, ses, "eeg",
                       f"{sub}_{ses}_task-{args.task}_eeg.edf")
    events = os.path.join(HERE, "data", DATASET, sub, ses, "eeg",
                          f"{sub}_{ses}_task-{args.task}_events.tsv")

    # Inspect the session
    inspect_out = os.path.join(HERE, "outputs", f"inspection_{tag}.txt")
    run([PY, os.path.join(HERE, "code", "step01_inspect_dataset.py"),
         "--subject", sub, "--session", ses, "--out", inspect_out])

    # Build encoding trials for this session
    enc_csv = os.path.join(HERE, "outputs", f"encoding_trials_{tag}.csv")
    enc_sum = os.path.join(HERE, "outputs", f"encoding_trials_summary_{tag}.txt")
    run([PY, os.path.join(HERE, "code", "step05_create_encoding_trials.py"),
         "--events", events, "--eeg", eeg,
         "--out-csv", enc_csv, "--out-summary", enc_sum])

    # Coverage report (authoritative, from the built table)
    trials = pd.read_csv(enc_csv)
    words = trials.word.str.upper()
    in_peers = words.isin(peers)
    missing = sorted(set(words[~in_peers]))
    valid_timing = (trials.onset > 0) & (trials["sample"] >= 0)

    print("\n" + "=" * 70)
    print("FINAL REPORT — task-ltpFR2 candidate")
    print("=" * 70)
    print(f"subject / session      : {sub} / {ses}")
    print(f"task                   : {args.task}")
    print(f"WORD count             : {len(trials)}")
    print("REC_WORD count         : (see summary) recalled+forgotten below")
    print(f"recalled (1)           : {int((trials.recalled == 1).sum())}")
    print(f"forgotten (0)          : {int((trials.recalled == 0).sum())}")
    print(f"recall rate            : {trials.recalled.mean():.4f}")
    print(f"unique words           : {words.nunique()}")
    print(f"word coverage vs peers : {int(in_peers.sum())}/{len(trials)} "
          f"({100*in_peers.mean():.1f}%)")
    print(f"valid EEG onsets       : {int(valid_timing.sum())}/{len(trials)} "
          f"({100*valid_timing.mean():.1f}%)  <- rows alignable to EEG later")
    print(f"unique words NOT in peers: {len(missing)}")
    if missing:
        print(f"  missing words: {missing[:30]}"
              + (" ..." if len(missing) > 30 else ""))
    else:
        print("  missing words: NONE — full coverage.")
    print("\nartifacts:")
    print(f"  inspection : {os.path.relpath(inspect_out, HERE)}")
    print(f"  encoding   : {os.path.relpath(enc_csv, HERE)}")
    print(f"  enc summary: {os.path.relpath(enc_sum, HERE)}")
    print("\nStopped: no X_eeg.npy / Y_t5.npy, no EEG extraction, no training.")


if __name__ == "__main__":
    main()
