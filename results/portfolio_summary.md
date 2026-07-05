# AI Word-Embedding Fidelity and Memory

**50-char description:** `EEG→LLM embedding decoding vs memory (null result)`

## Overview

A complete audited research pipeline asking a clean neuroscience question: when
you read a word, can its EEG response be decoded into that word's large-language-
model embedding well enough to predict whether you'll later remember it? Using
the PEERS EEG dataset (OpenNeuro ds004395), I built the full chain — T5-large
word embeddings, trial-level EEG feature extraction, ridge decoding, and a
logistic mixed-effects memory model — and validated every stage. The honest
result is a **negative result**: embedding fidelity did not predict recall,
because word-specific decoding from raw EEG was at chance.

## Technical overview

- **Language model:** Google T5-large encoder (`T5EncoderModel`), middle layer
  `hidden_states[12]`, subword-mean pooled with EOS/pad excluded → 576 × 1024
  embedding matrix (per-word, reproduced from the model to float32 precision).
- **Neural data:** OpenNeuro ds004395 (PEERS), task-ltpFR2, 129-channel EGI EEG
  at 500 Hz. Sessions streamed selectively from OpenNeuro S3 (no bulk download),
  screened so every word's analysis window fits inside the recording.
- **Scale:** 4 subjects × 2 sessions × 576 words = 4608 trials.
- **Stack:** Python, PyTorch + Transformers (embeddings), MNE (EEG),
  scikit-learn (ridge), statsmodels (mixed-effects), pandas/numpy/matplotlib.

## Methods

- Free-recall labels derived from `WORD`→`REC_WORD` events by item id within
  list (recognition events never used).
- Raw 300–800 ms EEG window per word, 129 × 250 = 32,250 features, sample-based
  and bit-exact against the source EDF.
- Ridge regression (α = 10,000, SVD solver) with held-out-trial 5-fold CV and
  train-only standardization; one out-of-fold prediction per trial.
- Fidelity = cosine(predicted, true) embedding. Word-specific sanity metrics
  (centered cosine, true-word rank, retrieval percentile, top-k) plus a
  **shuffled-label control**.
- **Logistic mixed-effects memory model:**
  `recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`.

## Results

- Final model: odds ratio **0.971** per 1 SD of fidelity, 95% interval
  **[0.913, 1.033]**, p ≈ **0.354** — no effect.
- Remembered vs forgotten mean fidelity: **0.84627 vs 0.84711** (≈ identical).
- Raw cosine was inflated by a common embedding direction; **word-specific
  decoding was at chance** (percentile ≈ 0.5, real ≈ shuffled) in every session.
- **The key prediction was not supported in this analysis.**

## Limitations

- Small sample (4 subjects / 8 sessions); underpowered for confirmatory claims.
- Raw broadband voltage features only — no time-frequency band power, baseline
  correction, or spatial filtering yet.
- Single fixed decoding window (300–800 ms) and one regularization strength.

## Next steps

Future work should use **better EEG features** — time-frequency band power,
baseline correction, and spatial filtering — plus **more subjects and sessions**,
and only then re-test the memory model. Getting word-specific decoding above
chance is the prerequisite before any memory claim can be made.

---

*A complete audited research pipeline that reports a negative result: a
logistic mixed-effects memory model over chance-level word-specific EEG-to-LLM
decoding.*
