# PEERS T5-large Encoder Embeddings

Extract a single 1024-dimensional embedding for each of the 576 PEERS words using
the **T5-large encoder**, taken from the **middle encoder layer**.

## What the script does

`extract_t5_peers_embeddings.py`:

1. Loads `peers_words.csv` (one column: `word`).
2. Confirms there are **exactly 576** words (it refuses to continue otherwise).
3. Strips whitespace but **preserves the original PEERS word order exactly**.
4. Loads the `google-t5/t5-large` tokenizer and **`T5EncoderModel`**.
5. Selects a device: **CUDA → Apple Silicon MPS → CPU**.
6. Tokenizes words in batches and runs the encoder with `output_hidden_states=True`.
7. Selects the **middle encoder hidden state** (`hidden_states[12]` by default).
8. For each word, averages the subword-token embeddings while **excluding the EOS
   token `</s>` and any padding tokens**.
9. Saves a `576 × 1024` matrix plus a word-order CSV, a full human-readable CSV,
   and a metadata JSON.

## Why `T5EncoderModel`?

T5 is an encoder–decoder model. We only want the **encoder's** contextual
representations of each word — not generated text — so we load
[`T5EncoderModel`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel),
which is the encoder stack **without any decoder**. Using
`T5ForConditionalGeneration` would attach the decoder and language-model head,
which we do not use and which changes the forward signature.

## Why is EOS excluded?

The T5 tokenizer appends an end-of-sequence token `</s>` (`eos_token_id`) to every
input. That token is a sequence delimiter, **not part of the word**, so including
it in the mean would contaminate the word's representation. We also exclude
**padding** tokens (added to make batched sequences equal length), which carry no
word content. Only the real subword tokens of the word are averaged.

## What does `hidden_states[12]` mean?

With `output_hidden_states=True`, the encoder returns a tuple of length
`num_layers + 1`:

- `hidden_states[0]` → the **embedding output** (before any encoder block).
- `hidden_states[1] … hidden_states[24]` → the outputs of encoder blocks 1–24.

t5-large has **24 encoder layers**, so the **middle** layer is
`hidden_states[12]`. The script prints `config.num_layers` and derives the middle
as `num_layers // 2`; you can override with `--layer`.

## Input CSV format

`peers_words.csv` must contain a single `word` column:

```
word
apple
chair
river
...
```

It must contain exactly 576 rows (excluding the header).

## Output files

| File | Description |
| --- | --- |
| `peers_t5large_embeddings.npy` | NumPy array, shape `(576, 1024)`, `float32`. |
| `peers_word_order.csv` | Columns `row_index,word` — maps each row to its word. |
| `peers_t5large_embeddings.csv` | Human-readable: `row_index,word,dim_0,…,dim_1023`. |
| `embedding_metadata.json` | Model name, layer used, word count, dim, shape, EOS/padding exclusion flags, timestamp, package versions. |

Expected matrix shape: **576 × 1024**.

## Install

```bash
pip install -r requirements.txt
```

> A T5-large download is ~3 GB. Running on CPU works but is slower than GPU/MPS.

## Run the extractor

```bash
python extract_t5_peers_embeddings.py \
  --input peers_words.csv \
  --out-matrix peers_t5large_embeddings.npy \
  --out-order peers_word_order.csv \
  --out-csv peers_t5large_embeddings.csv \
  --out-metadata embedding_metadata.json \
  --batch-size 8 \
  --layer 12
```

The extractor also prints debugging output for the first 5 words: their
tokenization, which tokens were included in the mean, and confirmation that `</s>`
was excluded.

## Verify the outputs

```bash
python verify_embeddings.py \
  --matrix peers_t5large_embeddings.npy \
  --order peers_word_order.csv \
  --metadata embedding_metadata.json
```

The verifier confirms the shape is `(576, 1024)`, that there are 576 row mappings
with indices `0…575`, that there are no NaN/Inf values, prints the first 5 words,
prints vector-norm statistics (min/max/mean), and prints a success message if all
checks pass.

## References

- T5 model docs: https://huggingface.co/docs/transformers/model_doc/t5
- `T5EncoderModel`: https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel
