#!/usr/bin/env python3
"""
Build the regression targets: for each encoding trial, the T5 vector of the word
that was actually on screen.

The join that makes the whole project work. step04 produced 576 embeddings in
one order; step05 produced this session's 576 trials in presentation order.
Those two orders are unrelated, so every trial's word is looked up by name in
peers_word_order.csv and its row pulled from the embedding matrix.

    trial i shows word "OCEAN"
      -> peers_word_order says OCEAN is row 412
      -> Y[i] = embeddings[412]

Row i of Y must correspond to row i of X, or ridge would be trained to predict
the wrong word's embedding from a trial's EEG -- silently, with no error and a
plausible-looking result. That is why this stage does nothing but the lookup,
and why it refuses to continue on any unmatched word rather than dropping it.

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", default=os.path.join(HERE, "outputs/encoding_trials.csv"))
    ap.add_argument("--peers-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    ap.add_argument("--embeddings", default=os.path.join(HERE, "results/embeddings/peers_t5large_embeddings.npy"))
    ap.add_argument("--emb-metadata", default=os.path.join(HERE, "results/embeddings/embedding_metadata.json"))
    ap.add_argument("--out-y", default=os.path.join(HERE, "outputs/Y_t5.npy"))
    ap.add_argument("--out-meta", default=os.path.join(HERE, "outputs/target_metadata.json"))
    ap.add_argument("--out-map", default=os.path.join(HERE, "outputs/trial_targets_metadata.csv"))
    args = ap.parse_args()

    for p in (args.trials, args.peers_order, args.embeddings):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: required input not found: {p}")

    trials = pd.read_csv(args.trials)
    order = pd.read_csv(args.peers_order)
    emb = np.load(args.embeddings)

    print(f"encoding_trials: {len(trials)} rows")
    print(f"peers_word_order: {len(order)} rows")
    print(f"embeddings: {emb.shape} {emb.dtype}")

    if emb.shape[1] != EMB_DIM:
        sys.exit(f"ERROR: embeddings have {emb.shape[1]} dims, expected {EMB_DIM}")
    if len(order) != emb.shape[0]:
        sys.exit(f"ERROR: peers_word_order rows ({len(order)}) != embedding rows "
                 f"({emb.shape[0]})")

    # Case-fold before joining: the events files and the word pool do not agree
    # on casing. Uniqueness is enforced *after* folding, because two words that
    # differ only in case would collapse into one key here and silently give
    # every one of their trials the same embedding.
    order["_wU"] = order.word.str.upper()
    if order["_wU"].duplicated().any():
        dups = order.loc[order["_wU"].duplicated(), "_wU"].tolist()
        sys.exit(f"ERROR: peers_word_order has duplicate words: {dups[:10]}")
    word_to_row = dict(zip(order["_wU"], order.row_index.astype(int)))

    # -----------------------------------------------------------------
    # Build Y in EXACT encoding_trials order.
    # Indexed by position, never by a pandas merge: a merge can reorder or
    # duplicate rows, which would break the X/Y row correspondence that ridge
    # depends on and that nothing downstream would catch.
    # -----------------------------------------------------------------
    Y = np.zeros((len(trials), EMB_DIM), dtype=np.float32)
    rows = []
    missing = []
    for i, w in enumerate(trials.word):
        wU = str(w).upper()
        if wU not in word_to_row:
            missing.append((i, w))
            rows.append({"trial_row_index": i, "word": w, "peers_row_index": -1,
                         "matched": False, "embedding_l2_norm": np.nan})
            continue
        pr = word_to_row[wU]
        Y[i] = emb[pr]
        rows.append({"trial_row_index": i, "word": w, "peers_row_index": pr,
                     "matched": True,
                     "embedding_l2_norm": float(np.linalg.norm(emb[pr]))})

    if missing:
        sys.exit(f"ERROR: {len(missing)} trial words not found in peers_word_order: "
                 f"{missing[:10]}. Refusing to write partial targets.")

    map_df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    print("\n=== VALIDATION ===")
    checks = []

    def ck(label, cond, detail=""):
        checks.append((label, bool(cond)))
        print(f"[{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))

    ck(f"Y_t5 shape is ({len(trials)}, {EMB_DIM})", Y.shape == (len(trials), EMB_DIM),
       f"got {Y.shape}")
    ck("row order matches encoding_trials.csv exactly",
       (map_df.word.values == trials.word.values).all())
    ck("every trial word matched a peers_word_order row", len(missing) == 0)
    ck("no NaN values", not np.isnan(Y).any())
    ck("no infinite values", not np.isinf(Y).any())
    norms = np.linalg.norm(Y, axis=1)
    ck("no zero vectors", (norms > 0).all(), f"min norm {norms.min():.4f}")

    print("\nfirst 10 word -> embedding-row matches:")
    for i in range(min(10, len(trials))):
        r = map_df.iloc[i]
        print(f"  trial[{i:>3}] {r.word:<12} -> peers_row {int(r.peers_row_index):>3}  "
              f"||emb||={r.embedding_l2_norm:.3f}")

    if not all(ok for _, ok in checks):
        sys.exit("\nFAILURE: target validation failed; nothing written.")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
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
