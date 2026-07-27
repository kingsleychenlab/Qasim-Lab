#!/usr/bin/env python3
"""
Build the regression targets: for each encoding trial, the T5 vector of the word
that was actually on screen.

This is the join the rest of the project depends on. step04 produced 576
embeddings in one order; step05 produced this session's 576 trials in
presentation order. The two orders are unrelated, so I look up every trial's
word by name in peers_word_order.csv and pull its row from the embedding matrix.

    trial i shows word "OCEAN"
      -> peers_word_order says OCEAN is row 412
      -> Y[i] = embeddings[412]

Row i of Y has to line up with row i of X. Otherwise ridge gets trained to
predict the wrong word's embedding from a trial's EEG, and it does so silently,
with no error and a result that looks plausible. So this stage does nothing but
the lookup, and it refuses to continue on any unmatched word instead of dropping
it.

Outputs:
    outputs/Y_t5.npy                    (576 x 1024, float32) targets in trial order
    outputs/target_metadata.json        provenance
    outputs/trial_targets_metadata.csv  the per-trial word -> peers row mapping,
                                        kept so the join can be audited later
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIM = 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", default=os.path.join(HERE, "outputs/encoding_trials.csv"))
    parser.add_argument("--peers-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    parser.add_argument("--embeddings", default=os.path.join(HERE, "results/embeddings/peers_t5large_embeddings.npy"))
    parser.add_argument("--emb-metadata", default=os.path.join(HERE, "results/embeddings/embedding_metadata.json"))
    parser.add_argument("--out-y", default=os.path.join(HERE, "outputs/Y_t5.npy"))
    parser.add_argument("--out-meta", default=os.path.join(HERE, "outputs/target_metadata.json"))
    parser.add_argument("--out-map", default=os.path.join(HERE, "outputs/trial_targets_metadata.csv"))
    args = parser.parse_args()

    for path in (args.trials, args.peers_order, args.embeddings):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: required input not found: {path}")

    trials = pd.read_csv(args.trials)
    order = pd.read_csv(args.peers_order)
    embeddings = np.load(args.embeddings)

    print(f"encoding_trials: {len(trials)} rows")
    print(f"peers_word_order: {len(order)} rows")
    print(f"embeddings: {embeddings.shape} {embeddings.dtype}")

    if embeddings.shape[1] != EMB_DIM:
        sys.exit(f"ERROR: embeddings have {embeddings.shape[1]} dims, expected {EMB_DIM}")
    if len(order) != embeddings.shape[0]:
        sys.exit(f"ERROR: peers_word_order rows ({len(order)}) != embedding rows "
                 f"({embeddings.shape[0]})")

    # Case-fold before joining, since the events files and the word pool don't
    # agree on casing. I check uniqueness after folding, not before: two words
    # differing only in case would collapse into one key here and quietly give
    # all of their trials the same embedding.
    order["_wU"] = order.word.str.upper()
    if order["_wU"].duplicated().any():
        dups = order.loc[order["_wU"].duplicated(), "_wU"].tolist()
        sys.exit(f"ERROR: peers_word_order has duplicate words: {dups[:10]}")
    word_to_row = dict(zip(order["_wU"], order.row_index.astype(int)))

    # Build Y in encoding_trials order, indexing by position rather than a pandas
    # merge: a merge can reorder or duplicate rows and silently break the X/Y row
    # correspondence that ridge depends on.
    Y = np.zeros((len(trials), EMB_DIM), dtype=np.float32)
    rows = []
    missing = []
    for i, word in enumerate(trials.word):
        word_upper = str(word).upper()
        if word_upper not in word_to_row:
            missing.append((i, word))
            rows.append({"trial_row_index": i, "word": word, "peers_row_index": -1,
                         "matched": False, "embedding_l2_norm": np.nan})
            continue
        peers_row = word_to_row[word_upper]
        Y[i] = embeddings[peers_row]
        rows.append({"trial_row_index": i, "word": word, "peers_row_index": peers_row,
                     "matched": True,
                     "embedding_l2_norm": float(np.linalg.norm(embeddings[peers_row]))})

    if missing:
        sys.exit(f"ERROR: {len(missing)} trial words not found in peers_word_order: "
                 f"{missing[:10]}. Refusing to write partial targets.")

    map_df = pd.DataFrame(rows)

    print("\n=== VALIDATION ===")
    checks = []

    def record_check(label, cond, detail=""):
        checks.append((label, bool(cond)))
        print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    record_check(f"Y_t5 shape is ({len(trials)}, {EMB_DIM})", Y.shape == (len(trials), EMB_DIM),
       f"got {Y.shape}")
    record_check("row order matches encoding_trials.csv exactly",
       (map_df.word.values == trials.word.values).all())
    record_check("every trial word matched a peers_word_order row", len(missing) == 0)
    record_check("no NaN values", not np.isnan(Y).any())
    record_check("no infinite values", not np.isinf(Y).any())
    norms = np.linalg.norm(Y, axis=1)
    record_check("no zero vectors", (norms > 0).all(), f"min norm {norms.min():.4f}")

    print("\nfirst 10 word -> embedding-row matches:")
    for i in range(min(10, len(trials))):
        match_row = map_df.iloc[i]
        print(f"  trial[{i:>3}] {match_row.word:<12} -> peers_row {int(match_row.peers_row_index):>3}  "
              f"||emb||={match_row.embedding_l2_norm:.3f}")

    if not all(ok for _, ok in checks):
        sys.exit("\nFAILURE: target validation failed; nothing written.")

    np.save(args.out_y, Y)
    map_df.to_csv(args.out_map, index=False)

    emb_meta = {}
    if os.path.isfile(args.emb_metadata):
        emb_meta = json.load(open(args.emb_metadata))

    target_meta = {
        "description": "T5-large targets aligned to encoding_trials.csv row order.",
        "source_embeddings": os.path.relpath(args.embeddings, HERE),
        "source_trials": os.path.relpath(args.trials, HERE),
        "peers_word_order": os.path.relpath(args.peers_order, HERE),
        "model_name": emb_meta.get("model_name"),
        "encoder_layer_used": emb_meta.get("encoder_layer_used"),
        "n_trials": int(len(trials)),
        "embedding_dim": EMB_DIM,
        "Y_shape": list(Y.shape),
        "dtype": str(Y.dtype),
        "order_source": "encoding_trials.csv (exact row order preserved)",
        "all_words_matched": True,
        "no_nan": bool(not np.isnan(Y).any()),
        "no_inf": bool(not np.isinf(Y).any()),
        "no_zero_vectors": bool((norms > 0).all()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(target_meta, open(args.out_meta, "w"), indent=2)

    print(f"\nwrote {args.out_y} {Y.shape}")
    print(f"wrote {args.out_map}")
    print(f"wrote {args.out_meta}")
    print("STATUS: OK")


if __name__ == "__main__":
    main()
