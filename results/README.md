# AI Word-Embedding Fidelity and Memory

A complete, audited **negative-result** research pipeline testing whether the
brain's response to a word — as decoded into a large language model's word
embedding — predicts whether that word is later remembered.

## Goal

Test one prediction: **do words whose EEG encoding response can be decoded more
faithfully into their T5 language-model embedding get remembered more often?**
Concretely, does trial-level *embedding fidelity* (how well ridge regression
maps 300–800 ms of EEG onto the word's T5 vector) predict free recall?

## Dataset

**PEERS / OpenNeuro [ds004395](https://openneuro.org/datasets/ds004395)** —
the Penn Electrophysiology of Encoding and Retrieval Study. We use the
**task-ltpFR2** experiment, whose 576-word pool matches the T5 embedding list
exactly. EEG is 128–129 channel EGI, 500 Hz, stored as EDF. Only validated
sessions were used (see `audit_report.md`).

## Pipeline summary

```
word appears  →  T5-large embedding (1024-d)  →  EEG 300–800 ms after onset
              →  ridge regression predicts the embedding  →  cosine fidelity
              →  logistic mixed-effects memory model (recalled ~ fidelity + …)
```

- **T5 target**: middle-layer (`hidden_states[12]`) encoder representation,
  averaged over subword tokens, EOS/pad excluded → 576 × 1024 matrix.
- **EEG feature**: raw 300–800 ms window (129 channels × 250 timepoints = 32,250
  features) per word, sample-based extraction, no filtering/baseline/resampling.
- **Ridge**: within each subject/session, `alpha = 10000`, held-out-trial 5-fold
  CV, standardization fit on training folds only.
- **Fidelity**: cosine similarity between predicted and true T5 embedding.
- **Memory model**: `recalled ~ embedding_fidelity + session + (1|subject) +
  (1|word)`, logistic (binary outcome).

## Final conclusion

**The key prediction was not supported in this analysis.** In the final outline
model, embedding fidelity did not predict recall (odds ratio 0.971 per 1 SD,
95% interval [0.913, 1.033], p ≈ 0.354). Remembered and forgotten words had
essentially identical mean fidelity (0.84627 vs 0.84711).

Importantly, raw cosine fidelity is **inflated by a common embedding direction**
(all T5 vectors share a dominant direction and have large norms), so a high
absolute cosine (~0.85) does **not** indicate successful word decoding. When
measured with word-specific metrics (true-word rank / retrieval percentile), and
against a shuffled-label control, **word-specific decoding was at chance**
across all sessions. A null memory effect is therefore the expected, honest
outcome.

## What this is

This is a **complete audited research pipeline that reports a negative result**.
Every stage — T5 embeddings, recall-label derivation, EEG extraction, decoding
metrics, and the mixed-effects model — was independently validated (see
`audit_report.md`). The value is a trustworthy, reproducible end-to-end method
and an honest null, not a positive finding.

See `results_index.md` for a guide to every file in this folder.
