#!/usr/bin/env python3
"""
Find a task-ltpFR2 session in ds004395 that matches the 576-word PEERS T5
embedding list, download ONLY ONE such session, and run the inspection +
encoding-trials stages on it.

Why: task-ltpFR uses the ~1638-word wasnorm_wordpool (only ~35% of a session's
words are in peers_word_order.csv), whereas task-ltpFR2 (PEERS4) uses the
576-word pool that the T5 embeddings were built from (expected ~100% coverage).

This script:
  1. Lists ltpFR2 subject/session paths from OpenNeuro's public S3 listing
     (metadata only — does NOT download the full dataset).
  2. Prints candidate sessions with EEG/events sizes (smallest first).
  3. Downloads ONE ltpFR2 session (default: smallest EDF; override with --sub/--ses).
  4. Runs code/step01_inspect_dataset.py on that session.
  5. Runs code/step05_create_encoding_trials.py on that session.
  6. Checks word coverage against peers_word_order.csv and reports.

It does NOT create X_eeg.npy / Y_t5.npy and does NOT train anything.
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
    sess = collections.defaultdict(dict)
    tag = f"task-{task}"
    pages = 0
    while True:
        params = {"list-type": "2", "prefix": f"{DATASET}/"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        pages += 1
        for c in root.findall(f"{NS}Contents"):
            k = c.find(f"{NS}Key").text
            if tag not in k:
                continue
            size = int(c.find(f"{NS}Size").text)
            parts = k.split("/")
            if len(parts) < 3:
                continue
            key = (parts[1], parts[2])  # (sub-XXX, ses-Y)
            if k.endswith("_eeg.edf"):
                sess[key]["edf"] = size
            elif k.endswith("_events.tsv"):
                sess[key]["events"] = size
        trunc = root.find(f"{NS}IsTruncated")
        if trunc is not None and trunc.text == "true":
            token = root.find(f"{NS}NextContinuationToken").text
        else:
            break
    return sess, pages


def events_url(sub, ses, task):
    return f"{BUCKET}/{DATASET}/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_events.tsv"


def edf_url(sub, ses, task):
    return f"{BUCKET}/{DATASET}/{sub}/{ses}/eeg/{sub}_{ses}_task-{task}_eeg.edf"


def edf_header_ntimes(url, edf_size):
    """Parse n_times & sfreq from just the EDF header via a small Range request.
    Verified to match mne.io.read_raw_edf(...).n_times exactly. Avoids
    downloading the full (100s of MB) EDF just to learn its length."""
    r = requests.get(url, headers={"Range": "bytes=0-49999"}, timeout=60)
    r.raise_for_status()
    b = r.content

    def s(a, z):
        return b[a:z].decode("ascii", "ignore").strip()

    n_records = int(s(236, 244) or "-1")
    rec_dur = float(s(244, 252))
    ns = int(s(252, 256))
    header_nbytes = int(s(184, 192))
    base = 256 + ns * 216  # start of the per-signal "samples per record" block
    spr = [int(s(base + i * 8, base + i * 8 + 8)) for i in range(ns)]
    if n_records <= 0:  # EDF+ may store -1; recover from file size
        n_records = (edf_size - header_nbytes) // (2 * sum(spr))
    n_times = n_records * spr[0]
    sfreq = spr[0] / rec_dur
    return n_times, sfreq


def peek_session(sub, ses, task, peers, edf_size, win_start, win_stop):
    """Fetch events.tsv (tiny) + EDF header (tiny) and compute full window
    validity: a WORD is a valid EEG window only if
        sample>=0 and onset>0 and start_sample>=0 and stop_sample < n_times."""
    r = requests.get(events_url(sub, ses, task), timeout=60)
    if r.status_code != 200:
        return None
    ev = pd.read_csv(io.StringIO(r.text), sep="\t", na_values=["n/a", ""])
    w = ev[ev.trial_type == "WORD"].copy()
    rc = ev[ev.trial_type == "REC_WORD"]
    words = w.item_name.dropna().str.upper()
    cov = float(words.isin(peers).mean()) if len(words) else 0.0

    try:
        n_times, sfreq = edf_header_ntimes(edf_url(sub, ses, task), edf_size)
    except Exception as e:  # noqa: BLE001
        return {"n_word": len(w), "n_rec": len(rc), "coverage": cov,
                "n_valid_win": 0, "n_times": None, "sfreq": None,
                "err": f"{type(e).__name__}: {e}"}

    start = w["sample"] + int(win_start * sfreq)
    stop = w["sample"] + int(win_stop * sfreq)
    valid_win = (w["sample"] >= 0) & (w.onset > 0) & (start >= 0) & (stop < n_times)
    return {"n_word": len(w), "n_rec": len(rc), "coverage": cov,
            "n_valid_win": int(valid_win.sum()), "n_times": int(n_times),
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="ltpFR2")
    ap.add_argument("--peers-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    ap.add_argument("--sub", help="Force a subject label, e.g. LTP327 or sub-LTP327")
    ap.add_argument("--ses", help="Force a session label, e.g. 4 or ses-4")
    ap.add_argument("--top", type=int, default=12, help="How many candidates to list")
    ap.add_argument("--win-start", type=float, default=0.300,
                    help="EEG window start relative to word onset, in seconds.")
    ap.add_argument("--win-stop", type=float, default=0.800,
                    help="EEG window stop relative to word onset, in seconds.")
    ap.add_argument("--list-only", action="store_true",
                    help="Only scan + list candidates; do not download or run stages.")
    args = ap.parse_args()

    peers = peers_word_set(args.peers_order)
    print(f"peers_word_order.csv: {len(peers)} words")

    # ------------------------------------------------------------------
    # 1-2. Scan + list candidates
    # ------------------------------------------------------------------
    print(f"\nScanning OpenNeuro S3 listing for task-{args.task} sessions "
          "(metadata only, no bulk download)...")
    sess, pages = scan_task_sessions(args.task)
    subs = sorted({s for s, _ in sess})
    print(f"scanned {pages} listing pages; task-{args.task}: "
          f"{len(subs)} subjects, {len(sess)} sessions")
    if not sess:
        sys.exit(f"No task-{args.task} sessions found.")

    with_edf = [(k, v) for k, v in sess.items() if "edf" in v]
    with_edf.sort(key=lambda x: x[1]["edf"])
    print(f"\nSmallest {min(args.top, len(with_edf))} task-{args.task} sessions by EDF size:")
    print(f"  {'EDF (MB)':>9}  {'events(MB)':>10}  subject/session")
    for (sub, ses), v in with_edf[:args.top]:
        print(f"  {v['edf']/1e6:9.1f}  {v.get('events', 0)/1e6:10.3f}  {sub}/{ses}")

    if args.list_only:
        return

    # ------------------------------------------------------------------
    # 3. Choose target and verify coverage on the tiny events.tsv first
    # ------------------------------------------------------------------
    if args.sub and args.ses:
        target = (norm_label(args.sub, "sub-"), norm_label(args.ses, "ses-"))
        if target not in sess:
            sys.exit(f"Requested {target[0]}/{target[1]} is not a task-{args.task} session.")
    else:
        # A session is acceptable ONLY if all 576 WORD events yield a full
        # [win_start, win_stop] EEG window that fits inside the EDF:
        #   n_word == 576, coverage == 576/576, AND every window's stop_sample
        #   < raw.n_times (parsed cheaply from the EDF header). Small/truncated
        #   EDFs fail because their later word windows fall off the end.
        NEED = 576
        target = None
        print(f"\nEvaluating candidates (need WORD=576, cov=576/576, "
              f"valid {int(args.win_start*1000)}-{int(args.win_stop*1000)}ms windows=576/576):")
        print(f"  {'subject/session':22} {'WORD':>4} {'cov':>4} {'valid_win':>9} {'edf_s':>6}  pick?")
        for (sub, ses), v in with_edf:
            c = peek_session(sub, ses, args.task, peers, v["edf"],
                             args.win_start, args.win_stop)
            if not c:
                continue
            ok = (c["n_word"] == NEED and c["coverage"] >= 0.999
                  and c["n_valid_win"] == NEED)
            edf_s = f"{c['n_times']/c['sfreq']:.0f}" if c["n_times"] else "err"
            print(f"  {sub}/{ses:8} {c['n_word']:4} {int(round(c['coverage']*c['n_word'])):4} "
                  f"{c['n_valid_win']:9} {edf_s:>6}  {'<== SELECT' if ok else ''}")
            if ok:
                target = (sub, ses)
                sel = (v, c)
                break  # smallest qualifying EDF; stop probing
        if target is None:
            sys.exit("No task-ltpFR2 session with 576/576 valid EEG windows found "
                     "in the candidate list (try increasing --top).")
        sub, ses = target
        print(f"\nSelected: {sub}/{ses} (EDF {sel[0]['edf']/1e6:.1f} MB, "
              f"coverage 576/576, valid windows {sel[1]['n_valid_win']}/576, "
              f"EDF {sel[1]['n_times']/sel[1]['sfreq']:.0f}s)")

    sub, ses = target
    sub_label = sub.replace("sub-", "")
    ses_label = ses.replace("ses-", "")
    tag = f"{sub}_{ses}"

    # ------------------------------------------------------------------
    # 4. Download ONLY this one session
    # ------------------------------------------------------------------
    run([PY, os.path.join(HERE, "code", "step03_download_session.py"),
         "--sub", sub_label, "--ses", ses_label, "--task", args.task])

    eeg = os.path.join(HERE, "data", DATASET, sub, ses, "eeg",
                       f"{sub}_{ses}_task-{args.task}_eeg.edf")
    events = os.path.join(HERE, "data", DATASET, sub, ses, "eeg",
                          f"{sub}_{ses}_task-{args.task}_events.tsv")

    # ------------------------------------------------------------------
    # 5. Inspect this session
    # ------------------------------------------------------------------
    inspect_out = os.path.join(HERE, "outputs", f"inspection_{tag}.txt")
    run([PY, os.path.join(HERE, "code", "step01_inspect_dataset.py"),
         "--subject", sub, "--session", ses, "--out", inspect_out])

    # ------------------------------------------------------------------
    # 6. Build encoding trials for this session
    # ------------------------------------------------------------------
    enc_csv = os.path.join(HERE, "outputs", f"encoding_trials_{tag}.csv")
    enc_sum = os.path.join(HERE, "outputs", f"encoding_trials_summary_{tag}.txt")
    run([PY, os.path.join(HERE, "code", "step05_create_encoding_trials.py"),
         "--events", events, "--eeg", eeg,
         "--out-csv", enc_csv, "--out-summary", enc_sum])

    # ------------------------------------------------------------------
    # 7-8. Coverage report (authoritative, from the built table)
    # ------------------------------------------------------------------
    df = pd.read_csv(enc_csv)
    words = df.word.str.upper()
    in_peers = words.isin(peers)
    missing = sorted(set(words[~in_peers]))
    valid_timing = (df.onset > 0) & (df["sample"] >= 0)

    print("\n" + "=" * 70)
    print("FINAL REPORT — task-ltpFR2 candidate")
    print("=" * 70)
    print(f"subject / session      : {sub} / {ses}")
    print(f"task                   : {args.task}")
    print(f"WORD count             : {len(df)}")
    print("REC_WORD count         : (see summary) recalled+forgotten below")
    print(f"recalled (1)           : {int((df.recalled == 1).sum())}")
    print(f"forgotten (0)          : {int((df.recalled == 0).sum())}")
    print(f"recall rate            : {df.recalled.mean():.4f}")
    print(f"unique words           : {words.nunique()}")
    print(f"word coverage vs peers : {int(in_peers.sum())}/{len(df)} "
          f"({100*in_peers.mean():.1f}%)")
    print(f"valid EEG onsets       : {int(valid_timing.sum())}/{len(df)} "
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
