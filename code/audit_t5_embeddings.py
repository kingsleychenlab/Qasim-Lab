#!/usr/bin/env python3
"""
Spot-check the saved T5 embeddings by rebuilding a few from scratch.

Loads t5-large afresh, recomputes a handful of words, and compares against the
stored rows of peers_t5large_embeddings.npy. Catches the failure step04's own
validation cannot: a matrix that is internally consistent (right shape, no NaNs,
plausible norms) but was built with the wrong layer, the wrong pooling, or the
EOS token left in. Any of those would still produce a perfectly well-formed
576 x 1024 file, and every downstream number would be quietly wrong.

Recompute path deliberately mirrors the extractor exactly:
  - google-t5/t5-large via T5EncoderModel (encoder only, never the decoder)
  - output_hidden_states=True, take hidden_states[12] (middle of 24 layers)
  - drop EOS (</s>) and pad, average the remaining subword tokens

Reads the words back through peers_word_order.csv, so it also checks the
word -> row mapping the rest of the pipeline depends on.

Interpreting the output: max abs diff should be ~1e-6 or below. Small nonzero
values are expected -- float32 storage and nondeterministic GPU kernels -- so
the thresholds are <1e-5 OK, <1e-3 close, anything larger means the saved matrix
was not produced by this recipe.

Only a few words are checked because loading t5-large is the expensive part;
a pooling or layer error would show up on any word, not just some.
"""

import argparse
import os
import sys

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from common import load_word_to_row

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = ["ACTOR", "AIRPLANE", "ZIPPER"]
LAYER = 12


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google-t5/t5-large")
    ap.add_argument("--embeddings", default=os.path.join(HERE, "results/embeddings/peers_t5large_embeddings.npy"))
    ap.add_argument("--order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--words", nargs="+", default=WORDS)
    ap.add_argument("--device", default=None, help="force cpu/mps/cuda (default: auto)")
    args = ap.parse_args()

    for p in (args.embeddings, args.order):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: required input not found: {p}")

    saved = np.load(args.embeddings)
    word_to_row = load_word_to_row(args.order)

    device = torch.device(args.device) if args.device else select_device()
    print(f"device: {device}")
    print(f"loading tokenizer + T5EncoderModel: {args.model}")
    tok = T5Tokenizer.from_pretrained(args.model)
    model = T5EncoderModel.from_pretrained(args.model, output_hidden_states=True)
    model.to(device).eval()

    eos_id, pad_id = tok.eos_token_id, tok.pad_token_id
    n_layers = int(model.config.num_layers)
    print(f"encoder layers: {n_layers}; using hidden_states[{args.layer}]  "
          f"(eos_id={eos_id}, pad_id={pad_id})\n")

    max_diffs = []
    with torch.no_grad():
        for w in args.words:
            wU = w.upper()
            if wU not in word_to_row:
                print(f"[SKIP] {w!r} not in peers_word_order.csv")
                continue
            enc = tok([w], return_tensors="pt", padding=True, truncation=True)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            # hidden_states is num_layers+1 long: index 0 is the embedding
            # output, 1..24 are the encoder blocks. So [12] is the middle block,
            # not the 12th of 25.
            layer_hidden = out.hidden_states[args.layer][0]  # (seq, 1024)

            # The word's own subword tokens only. T5 appends </s> to every
            # sequence, and averaging it in would drag all 576 vectors toward a
            # shared constant -- inflating every cosine downstream.
            valid = attn[0].bool() & (input_ids[0] != eos_id) & (input_ids[0] != pad_id)
            vec = layer_hidden[valid].mean(dim=0).float().cpu().numpy().astype(np.float32)

            row = word_to_row[wU]
            saved_vec = saved[row]
            max_abs = float(np.max(np.abs(vec - saved_vec)))
            max_diffs.append(max_abs)

            toks = tok.convert_ids_to_tokens(input_ids[0].tolist())
            included = [t for t, k in zip(toks, valid.tolist()) if k]
            status = "OK (<1e-5)" if max_abs < 1e-5 else \
                     ("close (<1e-3)" if max_abs < 1e-3 else "LARGE")
            print(f"{w:<10} peers_row={row:<3} tokens_avgd={included}")
            print(f"           max|recomputed - saved| = {max_abs:.3e}   [{status}]\n")

    if max_diffs:
        print(f"overall max abs diff across audited words: {max(max_diffs):.3e}")
        print("expected: near 0 (~<1e-5) if saved embeddings are reproducible.")


if __name__ == "__main__":
    main()
