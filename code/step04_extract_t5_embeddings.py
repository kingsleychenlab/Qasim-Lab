#!/usr/bin/env python3
"""
Extract T5-large encoder embeddings for a PEERS word list.

For each word:
  1. Tokenize with the T5 SentencePiece tokenizer.
  2. Run the T5 *encoder* (T5EncoderModel, NOT the decoder,
     NOT T5ForConditionalGeneration) with output_hidden_states=True.
  3. Take the middle encoder hidden state (default: hidden_states[12],
     the 12th of 24 encoder layers for t5-large; layer 0 is the
     embedding output).
  4. Average the subword-token embeddings for the word, EXCLUDING:
        - padding tokens
        - the T5 end-of-sequence token </s> (eos_token_id)
  5. Save one 1024-dim vector per word.

Reference:
  https://huggingface.co/docs/transformers/model_doc/t5
  https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_WORDS = 576
EXPECTED_DIM = 1024


def get_versions():
    """Best-effort package version collection for metadata."""
    versions = {}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:
        versions["transformers"] = None
    versions["torch"] = getattr(torch, "__version__", None)
    versions["numpy"] = getattr(np, "__version__", None)
    versions["pandas"] = getattr(pd, "__version__", None)
    try:
        import sentencepiece

        versions["sentencepiece"] = getattr(sentencepiece, "__version__", None)
    except Exception:
        versions["sentencepiece"] = None
    versions["python"] = sys.version.split()[0]
    return versions


def select_device():
    """CUDA first, Apple Silicon MPS second, CPU otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_words(input_path):
    """Load the PEERS word list, strip whitespace, preserve order, enforce count."""
    df = pd.read_csv(input_path)
    if "word" not in df.columns:
        raise ValueError(
            f"Input CSV '{input_path}' must have a 'word' column. "
            f"Found columns: {list(df.columns)}"
        )
    # Strip whitespace but PRESERVE the original order exactly.
    words = [str(w).strip() for w in df["word"].tolist()]

    # Do not silently continue if the word count is not 576.
    if len(words) != EXPECTED_WORDS:
        raise ValueError(
            f"Expected exactly {EXPECTED_WORDS} PEERS words, but found {len(words)} "
            f"in '{input_path}'. Refusing to continue."
        )

    empty = [i for i, w in enumerate(words) if w == "" or w.lower() == "nan"]
    if empty:
        raise ValueError(f"Found empty/NaN words at row indices: {empty}")

    return words


def resolve_middle_layer(model, requested_layer):
    """
    Determine which hidden_states index to use as the middle encoder layer.

    hidden_states has length num_layers + 1:
        index 0            -> embedding output
        index 1..num_layers -> output of each encoder block

    For t5-large, num_layers == 24, so the middle layer is index 12.
    We honor --layer if provided, otherwise derive num_layers // 2 from config.
    """
    num_layers = int(model.config.num_layers)
    derived_middle = num_layers // 2
    layer = requested_layer if requested_layer is not None else derived_middle

    if layer < 0 or layer > num_layers:
        raise ValueError(
            f"Requested layer {layer} is out of range. Valid hidden_states "
            f"indices are 0..{num_layers} (0 = embeddings, {num_layers} = last)."
        )
    return num_layers, derived_middle, layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(HERE, "results/embeddings/peers_words.csv"))
    parser.add_argument("--out-matrix", default=os.path.join(HERE, "results/embeddings/peers_t5large_embeddings.npy"))
    parser.add_argument("--out-order", default=os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
    parser.add_argument("--out-csv", default=os.path.join(HERE, "results/embeddings/peers_t5large_embeddings.csv"))
    parser.add_argument("--out-metadata", default=os.path.join(HERE, "results/embeddings/embedding_metadata.json"))
    parser.add_argument("--model", default="google-t5/t5-large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="hidden_states index to use (0=embeddings). Default 12 = middle of t5-large's 24 layers.",
    )
    parser.add_argument("--debug-first", type=int, default=5,
                        help="Print tokenization debug for the first N words.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load words
    # ------------------------------------------------------------------
    words = load_words(args.input)
    print(f"Loaded {len(words)} words from {args.input} (order preserved).")

    # ------------------------------------------------------------------
    # 2. Device + model
    # ------------------------------------------------------------------
    device = select_device()
    print(f"Using device: {device}")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    print(f"Loading T5EncoderModel (encoder only): {args.model}")
    model = T5EncoderModel.from_pretrained(args.model, output_hidden_states=True)
    model.to(device)
    model.eval()

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    print(f"eos_token_id = {eos_id} ({tokenizer.eos_token!r}), "
          f"pad_token_id = {pad_id} ({tokenizer.pad_token!r})")

    num_layers, derived_middle, layer = resolve_middle_layer(model, args.layer)
    print(f"Model encoder layer count (config.num_layers): {num_layers}")
    print(f"hidden_states length will be {num_layers + 1} "
          f"(index 0 = embeddings, index {num_layers} = final layer).")
    print(f"Derived middle layer = {derived_middle}. Using hidden_states[{layer}].")
    if layer != derived_middle:
        print(f"NOTE: requested layer {layer} differs from derived middle {derived_middle}.")

    # ------------------------------------------------------------------
    # 3. Batched extraction
    # ------------------------------------------------------------------
    embeddings = np.zeros((len(words), EXPECTED_DIM), dtype=np.float32)
    debug_budget = max(0, args.debug_first)

    with torch.no_grad():
        for start in tqdm(range(0, len(words), args.batch_size), desc="Encoding"):
            batch_words = words[start:start + args.batch_size]

            enc = tokenizer(
                batch_words,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # hidden_states: tuple of (num_layers + 1) tensors,
            # each (batch, seq_len, hidden_size).
            layer_hidden = outputs.hidden_states[layer]  # (batch, seq, 1024)

            for i in range(len(batch_words)):
                global_idx = start + i

                # Clear, explicit mask:
                #   keep real tokens, drop EOS, drop padding.
                valid_mask = (
                    attention_mask[i].bool()
                    & (input_ids[i] != eos_id)
                    & (input_ids[i] != pad_id)
                )

                if valid_mask.sum().item() == 0:
                    raise RuntimeError(
                        f"No valid (non-EOS, non-pad) tokens for word "
                        f"{global_idx!r}: {batch_words[i]!r}"
                    )

                word_vector = layer_hidden[i][valid_mask].mean(dim=0)
                embeddings[global_idx] = word_vector.detach().cpu().numpy()

                # ---- Optional debugging output for the first N words ----
                if debug_budget > 0:
                    ids = input_ids[i].tolist()
                    toks = tokenizer.convert_ids_to_tokens(ids)
                    kept = valid_mask.tolist()
                    included = [t for t, k in zip(toks, kept) if k]
                    excluded = [t for t, k in zip(toks, kept) if not k]
                    print("\n--- DEBUG word "
                          f"[{global_idx}] {batch_words[i]!r} ---")
                    print(f"  input_ids : {ids}")
                    print(f"  tokens    : {toks}")
                    print(f"  included in mean : {included}")
                    print(f"  excluded (EOS/pad): {excluded}")
                    print(f"  '{tokenizer.eos_token}' excluded: "
                          f"{tokenizer.eos_token in excluded}")
                    debug_budget -= 1

    # ------------------------------------------------------------------
    # 4. Shape check
    # ------------------------------------------------------------------
    if embeddings.shape != (EXPECTED_WORDS, EXPECTED_DIM):
        raise RuntimeError(
            f"Embedding matrix has shape {embeddings.shape}, "
            f"expected ({EXPECTED_WORDS}, {EXPECTED_DIM})."
        )
    print(f"\nEmbedding matrix shape: {embeddings.shape}  (OK)")

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    np.save(args.out_matrix, embeddings)
    print(f"Saved matrix -> {args.out_matrix}")

    order_df = pd.DataFrame({"row_index": range(len(words)), "word": words})
    order_df.to_csv(args.out_order, index=False)
    print(f"Saved word order -> {args.out_order}")

    dim_cols = [f"dim_{d}" for d in range(EXPECTED_DIM)]
    full_df = pd.DataFrame(embeddings, columns=dim_cols)
    full_df.insert(0, "word", words)
    full_df.insert(0, "row_index", range(len(words)))
    full_df.to_csv(args.out_csv, index=False)
    print(f"Saved full embedding CSV -> {args.out_csv}")

    metadata = {
        "model_name": args.model,
        "encoder_layer_used": layer,
        "encoder_layer_count": num_layers,
        "derived_middle_layer": derived_middle,
        "hidden_states_note": (
            "hidden_states[0] is the embedding output; "
            f"hidden_states[1..{num_layers}] are encoder blocks. "
            f"hidden_states[{layer}] used as the middle encoder layer."
        ),
        "num_words": len(words),
        "embedding_dim": EXPECTED_DIM,
        "matrix_shape": list(embeddings.shape),
        "eos_excluded": True,
        "padding_excluded": True,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "encoder_only": True,
        "model_class": "T5EncoderModel",
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": get_versions(),
    }
    with open(args.out_metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {args.out_metadata}")

    print("\nDone.")


if __name__ == "__main__":
    main()
