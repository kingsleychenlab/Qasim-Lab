#!/usr/bin/env python3
"""
Independent T5-large embedding recompute for the reproducibility rerun.
Mirrors the original pipeline exactly:
  google-t5/t5-large, T5EncoderModel (encoder only), hidden_states[12],
  average subword tokens, exclude EOS (id 1) and pad (id 0), float32, 576x1024.

Recomputes in the canonical peers_word_order.csv order so rows align with the
current saved matrix. Writes into rerun_full_validation/outputs/.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 1024
LAYER = 12


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default=os.path.join(HERE, "peers_word_order.csv"))
    ap.add_argument("--model", default="google-t5/t5-large")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-matrix", required=True)
    ap.add_argument("--out-order", required=True)
    ap.add_argument("--out-metadata", required=True)
    args = ap.parse_args()

    order = pd.read_csv(args.order).sort_values("row_index").reset_index(drop=True)
    words = order.word.tolist()
    assert len(words) == 576, f"expected 576 words, got {len(words)}"

    dev = device()
    print(f"device={dev}; loading {args.model}")
    tok = T5Tokenizer.from_pretrained(args.model)
    model = T5EncoderModel.from_pretrained(args.model, output_hidden_states=True).to(dev).eval()
    eos, pad = tok.eos_token_id, tok.pad_token_id
    n_layers = int(model.config.num_layers)
    print(f"encoder layers={n_layers}; using hidden_states[{LAYER}]; eos={eos} pad={pad}")

    emb = np.zeros((576, DIM), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, 576, args.batch_size):
            batch = words[s:s + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            ids = enc["input_ids"].to(dev)
            att = enc["attention_mask"].to(dev)
            out = model(input_ids=ids, attention_mask=att, output_hidden_states=True)
            hid = out.hidden_states[LAYER]
            for i in range(len(batch)):
                mask = att[i].bool() & (ids[i] != eos) & (ids[i] != pad)
                emb[s + i] = hid[i][mask].mean(0).float().cpu().numpy().astype(np.float32)

    np.save(args.out_matrix, emb)
    order[["row_index", "word"]].to_csv(args.out_order, index=False)
    meta = {
        "model_name": args.model, "model_class": "T5EncoderModel", "encoder_only": True,
        "encoder_layer_used": LAYER, "encoder_layer_count": n_layers,
        "eos_excluded": True, "padding_excluded": True,
        "eos_token_id": eos, "pad_token_id": pad,
        "num_words": 576, "embedding_dim": DIM, "matrix_shape": list(emb.shape),
        "dtype": str(emb.dtype), "device": str(dev),
        "torch": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "note": "independent reproducibility rerun recompute",
    }
    json.dump(meta, open(args.out_metadata, "w"), indent=2)
    norms = np.linalg.norm(emb, axis=1)
    print(f"saved {emb.shape} float32; norms min {norms.min():.1f} "
          f"mean {norms.mean():.1f} max {norms.max():.1f}")
    print(f"NaN={np.isnan(emb).any()} Inf={np.isinf(emb).any()} "
          f"zero_rows={(norms==0).sum()}")


if __name__ == "__main__":
    main()
