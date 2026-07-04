#!/usr/bin/env python3
"""
Verify the PEERS T5-large embedding outputs.

Checks:
  - matrix shape is (576, 1024)
  - there are 576 row mappings
  - row indices go from 0 to 575
  - no NaN values
  - no infinite values
Then prints the first 5 words with row indices and vector-norm statistics.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

EXPECTED_WORDS = 576
EXPECTED_DIM = 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="peers_t5large_embeddings.npy")
    parser.add_argument("--order", default="peers_word_order.csv")
    parser.add_argument("--metadata", default="embedding_metadata.json")
    args = parser.parse_args()

    ok = True

    def check(label, condition):
        nonlocal ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            ok = False
        print(f"[{status}] {label}")
        return condition

    # ------------------------------------------------------------------
    # Load matrix
    # ------------------------------------------------------------------
    matrix = np.load(args.matrix)
    check(
        f"matrix shape is ({EXPECTED_WORDS}, {EXPECTED_DIM}) -> got {matrix.shape}",
        matrix.shape == (EXPECTED_WORDS, EXPECTED_DIM),
    )

    # ------------------------------------------------------------------
    # Load order mapping
    # ------------------------------------------------------------------
    order = pd.read_csv(args.order)
    check(
        f"row mapping count is {EXPECTED_WORDS} -> got {len(order)}",
        len(order) == EXPECTED_WORDS,
    )
    check(
        "order CSV has columns row_index, word",
        list(order.columns) == ["row_index", "word"],
    )

    indices = order["row_index"].tolist()
    check(
        f"row indices go from 0 to {EXPECTED_WORDS - 1}",
        indices == list(range(EXPECTED_WORDS)),
    )

    # ------------------------------------------------------------------
    # NaN / Inf checks
    # ------------------------------------------------------------------
    check("no NaN values", not np.isnan(matrix).any())
    check("no infinite values", not np.isinf(matrix).any())

    # ------------------------------------------------------------------
    # Optional metadata
    # ------------------------------------------------------------------
    if os.path.exists(args.metadata):
        with open(args.metadata) as f:
            meta = json.load(f)
        print("\n--- metadata ---")
        for key in (
            "model_name",
            "model_class",
            "encoder_layer_used",
            "encoder_layer_count",
            "num_words",
            "embedding_dim",
            "matrix_shape",
            "eos_excluded",
            "padding_excluded",
            "timestamp_utc",
        ):
            if key in meta:
                print(f"  {key}: {meta[key]}")
        check("metadata eos_excluded is True", meta.get("eos_excluded") is True)
        check("metadata padding_excluded is True", meta.get("padding_excluded") is True)
    else:
        print(f"\n(metadata file '{args.metadata}' not found; skipping metadata checks)")

    # ------------------------------------------------------------------
    # First 5 words
    # ------------------------------------------------------------------
    print("\n--- first 5 words ---")
    for _, row in order.head(5).iterrows():
        print(f"  [{int(row['row_index'])}] {row['word']}")

    # ------------------------------------------------------------------
    # Norm statistics
    # ------------------------------------------------------------------
    norms = np.linalg.norm(matrix, axis=1)
    print("\n--- vector norm statistics ---")
    print(f"  min : {norms.min():.6f}")
    print(f"  max : {norms.max():.6f}")
    print(f"  mean: {norms.mean():.6f}")

    print()
    if ok:
        print("SUCCESS: all checks passed.")
        sys.exit(0)
    else:
        print("FAILURE: one or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
