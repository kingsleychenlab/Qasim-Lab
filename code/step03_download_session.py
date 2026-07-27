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
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = requests.get(BUCKET, params=params, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        for content in root.findall(f"{NS}Contents"):
            keys.append((content.find(f"{NS}Key").text, int(content.find(f"{NS}Size").text)))
        truncated = root.find(f"{NS}IsTruncated")
        if truncated is not None and truncated.text == "true":
            token = root.find(f"{NS}NextContinuationToken").text
        else:
            break
    return keys


def download_key(key, dest_root, chunk=1024 * 1024):
    """Stream one S3 key to dest_root, preserving the path below the dataset id."""
    rel_path = key.split(f"{DATASET}/", 1)[1]
    dest_path = os.path.join(dest_root, DATASET, rel_path)
    dataset_root = os.path.realpath(os.path.join(dest_root, DATASET))
    if os.path.commonpath([dataset_root, os.path.realpath(dest_path)]) != dataset_root:
        raise ValueError(f"refusing to write outside dataset dir: {key}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"  skip (exists)  {rel_path}")
        return
    tmp_path = dest_path + ".part"
    url = f"{BUCKET}/{key}"
    downloaded_bytes = 0
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk_data in response.iter_content(chunk_size=chunk):
                f.write(chunk_data)
                downloaded_bytes += len(chunk_data)
    os.replace(tmp_path, dest_path)
    print(f"  ok  {downloaded_bytes/1e6:8.2f} MB  {rel_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", help="Subject label without 'sub-' prefix, e.g. LTP063")
    parser.add_argument("--ses", help="Session label without 'ses-' prefix, e.g. 15")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Destination root (default: ./data)")
    parser.add_argument("--skip-eeg", action="store_true", help="Download sidecars only, skip large EEG binaries")
    parser.add_argument("--task", default=None,
                        help="Only download files for this task (e.g. ltpFR2). Files for OTHER "
                             "tasks are skipped; generic sidecars with no 'task-' in the name are "
                             "always kept. Important for mixed sessions (e.g. ltpFR2 + VFFR).")
    parser.add_argument("--list-subject", metavar="LABEL",
                        help="List all sessions + EEG file sizes for a subject, then exit")
    args = parser.parse_args()

    # Always fetch the small top-level metadata files.
    top_level_keys = [f"{DATASET}/{f}" for f in (
        "dataset_description.json", "README", "CHANGES",
        "participants.tsv", "participants.json",
    )]

    if args.list_subject:
        keys = list_prefix(f"{DATASET}/sub-{args.list_subject}/")
        eeg_files = [(key, size) for key, size in keys if key.endswith(EEG_BINARY_EXT)]
        print(f"Subject sub-{args.list_subject}: {len(keys)} files, {len(eeg_files)} EEG binaries")
        for key, size in sorted(eeg_files, key=lambda x: x[1]):
            print(f"  {size/1e6:8.1f} MB  {key}")
        if eeg_files:
            smallest = min(eeg_files, key=lambda x: x[1])
            print(f"\nSmallest EEG file: {smallest[0]} ({smallest[1]/1e6:.1f} MB)")
        return

    if not args.sub or not args.ses:
        parser.error("--sub and --ses are required (unless using --list-subject)")

    prefix = f"{DATASET}/sub-{args.sub}/ses-{args.ses}/"
    keys = list_prefix(prefix)
    if not keys:
        parser.error(f"No files found under {prefix} (check --sub / --ses).")

    keys_to_get = top_level_keys + [key for key, _ in keys]
    if args.task:
        # Keep files for this task + generic files that carry no 'task-' label
        # (electrodes, coordsystem, scans, top-level metadata). Drop other tasks.
        keys_to_get = [key for key in keys_to_get
                       if ("task-" not in os.path.basename(key)) or (f"task-{args.task}" in key)]
    if args.skip_eeg:
        keys_to_get = [key for key in keys_to_get if not key.endswith(EEG_BINARY_EXT)]

    print(f"Downloading {len(keys_to_get)} files for sub-{args.sub}/ses-{args.ses} "
          f"into {args.dest}/{DATASET}/ ...")
    for key in keys_to_get:
        download_key(key, args.dest)
    print("Done.")


if __name__ == "__main__":
    main()
