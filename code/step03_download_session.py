#!/usr/bin/env python3
"""
Fetch one subject/session of ds004395 (PEERS) from OpenNeuro's public S3 bucket.

Anonymous HTTPS, one session at a time (~500 MB EDF plus small sidecars). The
full dataset is 8.7 TB, so downloading it is not an option. The pipeline is
built around pulling only the sessions that pass step09's validity screen, which
is also why that screen reads the EDF header over the network before committing
to the body.

Skips files already on disk, so it's safe to re-run and cheap to resume after
an interrupted scale-up.

Public mirror:
    https://s3.amazonaws.com/openneuro.org/ds004395/...

Usage:
    python code/step03_download_session.py --sub LTP063 --ses 15
    python code/step03_download_session.py --list-subject LTP063   # list sessions+sizes, no download
    python code/step03_download_session.py --sub LTP063 --ses 15 --skip-eeg  # sidecars only
"""

import argparse
import os
import xml.etree.ElementTree as ET

import requests

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds004395"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DEFAULT_DEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# Extensions treated as "large EEG binaries" (skipped when --skip-eeg is set).
EEG_BINARY_EXT = (".edf", ".bdf", ".set", ".fdt", ".eeg", ".vhdr", ".vmrk")


def list_prefix(prefix):
    """Anonymous ListObjectsV2 -> list of (key, size)."""
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall(f"{NS}Contents"):
            out.append((c.find(f"{NS}Key").text, int(c.find(f"{NS}Size").text)))
        trunc = root.find(f"{NS}IsTruncated")
        if trunc is not None and trunc.text == "true":
            token = root.find(f"{NS}NextContinuationToken").text
        else:
            break
    return out


def download_key(key, dest_root, chunk=1024 * 1024):
    """Stream one S3 key to dest_root, preserving the path below the dataset id."""
    rel = key.split(f"{DATASET}/", 1)[1]
    out = os.path.join(dest_root, DATASET, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        print(f"  skip (exists)  {rel}")
        return
    tmp = out + ".part"
    url = f"{BUCKET}/{key}"
    got = 0
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                f.write(c)
                got += len(c)
    os.replace(tmp, out)
    print(f"  ok  {got/1e6:8.2f} MB  {rel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sub", help="Subject label without 'sub-' prefix, e.g. LTP063")
    ap.add_argument("--ses", help="Session label without 'ses-' prefix, e.g. 15")
    ap.add_argument("--dest", default=DEFAULT_DEST, help="Destination root (default: ./data)")
    ap.add_argument("--skip-eeg", action="store_true", help="Download sidecars only, skip large EEG binaries")
    ap.add_argument("--task", default=None,
                    help="Only download files for this task (e.g. ltpFR2). Files for OTHER "
                         "tasks are skipped; generic sidecars with no 'task-' in the name are "
                         "always kept. Important for mixed sessions (e.g. ltpFR2 + VFFR).")
    ap.add_argument("--list-subject", metavar="LABEL",
                    help="List all sessions + EEG file sizes for a subject, then exit")
    args = ap.parse_args()

    # Always fetch the small top-level metadata files.
    top = [f"{DATASET}/{f}" for f in (
        "dataset_description.json", "README", "CHANGES",
        "participants.tsv", "participants.json",
    )]

    if args.list_subject:
        keys = list_prefix(f"{DATASET}/sub-{args.list_subject}/")
        eeg = [(k, s) for k, s in keys if k.endswith(EEG_BINARY_EXT)]
        print(f"Subject sub-{args.list_subject}: {len(keys)} files, {len(eeg)} EEG binaries")
        for k, s in sorted(eeg, key=lambda x: x[1]):
            print(f"  {s/1e6:8.1f} MB  {k}")
        if eeg:
            smallest = min(eeg, key=lambda x: x[1])
            print(f"\nSmallest EEG file: {smallest[0]} ({smallest[1]/1e6:.1f} MB)")
        return

    if not args.sub or not args.ses:
        ap.error("--sub and --ses are required (unless using --list-subject)")

    prefix = f"{DATASET}/sub-{args.sub}/ses-{args.ses}/"
    keys = list_prefix(prefix)
    if not keys:
        ap.error(f"No files found under {prefix} (check --sub / --ses).")

    to_get = top + [k for k, _ in keys]
    if args.task:
        # Keep files for this task + generic files that carry no 'task-' label
        # (electrodes, coordsystem, scans, top-level metadata). Drop other tasks.
        to_get = [k for k in to_get
                  if ("task-" not in os.path.basename(k)) or (f"task-{args.task}" in k)]
    if args.skip_eeg:
        to_get = [k for k in to_get if not k.endswith(EEG_BINARY_EXT)]

    print(f"Downloading {len(to_get)} files for sub-{args.sub}/ses-{args.ses} "
          f"into {args.dest}/{DATASET}/ ...")
    for k in to_get:
        download_key(k, args.dest)
    print("Done.")


if __name__ == "__main__":
    main()
