#!/usr/bin/env python3
"""
Clean-room T5 embedding step for the independent redo.
Starts from peers_words.csv ONLY. Follows the outline literally:
  T5-large, encoder only (T5EncoderModel), middle encoder layer hidden_states[12],
  average subword tokens, exclude EOS + pad, one 1024-d vector per word.
Saves a 576x1024 matrix and a word/row_index CSV.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM, LAYER = 1024, 12


def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default=os.path.join(HERE, "peers_words.csv"))
    ap.add_argument("--out-matrix", required=True)
    ap.add_argument("--out-order", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    words = pd.read_csv(a.words)["word"].astype(str).str.strip().tolist()
    assert len(words) == 576, f"expected 576 words, got {len(words)}"

    d = dev()
    tok = T5Tokenizer.from_pretrained("google-t5/t5-large")
    model = T5EncoderModel.from_pretrained("google-t5/t5-large",
                                           output_hidden_states=True).to(d).eval()
    eos, pad = tok.eos_token_id, tok.pad_token_id
    print(f"device={d} layers={model.config.num_layers} using hidden_states[{LAYER}] "
          f"eos={eos} pad={pad}")

    emb = np.zeros((576, DIM), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, 576, a.batch_size):
            b = words[s:s + a.batch_size]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True)
            ids, att = enc["input_ids"].to(d), enc["attention_mask"].to(d)
            hid = model(input_ids=ids, attention_mask=att,
                        output_hidden_states=True).hidden_states[LAYER]
            for i in range(len(b)):
                m = att[i].bool() & (ids[i] != eos) & (ids[i] != pad)
                emb[s + i] = hid[i][m].mean(0).float().cpu().numpy().astype(np.float32)

    np.save(a.out_matrix, emb)
    pd.DataFrame({"word": words, "row_index": range(576)}).to_csv(a.out_order, index=False)
    n = np.linalg.norm(emb, axis=1)
    json.dump({"model_name": "google-t5/t5-large", "model_class": "T5EncoderModel",
               "encoder_only": True, "encoder_layer_used": LAYER, "eos_excluded": True,
               "padding_excluded": True, "matrix_shape": list(emb.shape),
               "dtype": str(emb.dtype), "device": str(d), "torch": torch.__version__,
               "source": "peers_words.csv", "timestamp_utc": datetime.now(timezone.utc).isoformat()},
              open(a.out_meta, "w"), indent=2)
    print(f"saved {emb.shape} float32 norms[min {n.min():.1f} mean {n.mean():.1f} max {n.max():.1f}] "
          f"NaN={np.isnan(emb).any()} zero_rows={(n==0).sum()}")


if __name__ == "__main__":
    main()
