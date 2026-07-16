# Code

This is the pipeline behind one question: when someone studies a word, is that
word more likely to be remembered if the EEG during encoding does a better job of
predicting the AI (T5) embedding of the word?

Dataset: PEERS / OpenNeuro ds004395 v2.0.0, `ltpFR2` task.

```
word appears -> EEG during encoding -> predicted T5 embedding
             -> compare to the real embedding -> does the similarity predict recall?
```

## Scripts, in order

Each `stepNN_` script is one stage. Run them in order.

| Script | Stage |
| --- | --- |
| `step01_inspect_dataset.py` | Look at the ds004395 structure, tasks, and channels |
| `step02_find_sessions.py` | Find valid `ltpFR2` sessions (576 words, full EEG coverage) |
| `step03_download_session.py` | Download one session (sidecars plus one EDF) |
| `step04_extract_t5_embeddings.py` | Compute one T5-large embedding per PEERS word |
| `step05_create_encoding_trials.py` | Build the encoding trials and recall labels |
| `step06_build_trial_targets.py` | Map each trial to its word's embedding (`Y_t5.npy`) |
| `step07_extract_eeg_features.py` | Pull the 300-800 ms window (`X_eeg.npy`) |
| `step08_run_ridge_cv.py` | Ridge regression with held-out CV, then cosine fidelity |
| `step09_scale_multi_subject.py` | Run the chain across subjects |
| `step10_scale_multi_session.py` | Run the chain across sessions per subject |
| `step11_run_memory_model.py` | The memory model (logistic mixed-effects) |
| `step12_analyze_results.py` | Aggregate and compare the 4 / 16 / 32-subject runs |
| `step13_build_results_package.py` | Build the `results/` tables and figures |

### Audits and validators

| Script | What it checks |
| --- | --- |
| `audit_t5_embeddings.py` | Recomputes the embeddings from scratch and compares them to the stored rows |
| `audit_eeg_extraction.py` | EEG window timing and sample alignment |
| `audit_recall_labels.py` | Recall labels against the raw behavioral records |
| `audit_precision.py` | End-to-end structural and numeric audit of the whole thing |
| `validate_encoding_trials.py` | Trial counts and word coverage |
| `validate_model_inputs.py` | `X_eeg` / `Y_t5` shapes and alignment |

### Shared

`common.py` holds the three things nearly every stage needed: `Tee` (print a
report to stdout and a file at the same time, tallying PASS/FAIL), `peers_word_set`,
and `load_word_to_row`. It's imported as a plain sibling (`from common import Tee`),
which works because a script's own directory is on `sys.path` when it runs.

## Method

- **Embeddings.** `google-t5/t5-large`, `T5EncoderModel` (the encoder, not the
  decoder), middle encoder layer `hidden_states[12]`. Subword tokens are averaged;
  the EOS and pad tokens are left out. One 1024-dim vector per word, so a 576 x 1024
  matrix. The row order is recorded in `results/embeddings/peers_word_order.csv`
  (`word`, `row_index`).
- **EEG.** 300-800 ms after word onset, 129 channels x 250 timepoints, raw 500 Hz,
  flattened to 32,250 features per trial.
- **Decoding.** Ridge regression (`alpha = 10000`, heavy regularization) fit per
  subject/session, 5-fold held-out cross-validation. `StandardScaler` is fit on the
  training folds only. Every trial gets exactly one out-of-fold prediction.
- **Fidelity.** `cosine(predicted embedding, true embedding)` per held-out trial.
- **Memory model.** `recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`,
  logistic mixed-effects. Fidelity is z-scored, so the odds ratio is per 1 SD.

## Running it

```bash
python -m venv venv && ./venv/bin/pip install -r code/requirements.txt

# Stage 1: embeddings (576 words -> 576 x 1024)
./venv/bin/python code/step04_extract_t5_embeddings.py

# Stage 2: scale the EEG chain across subjects/sessions
./venv/bin/python code/step10_scale_multi_session.py --subjects 32 --sessions-per-subject 2

# Stage 3: memory model plus the comparison against the 4- and 16-subject runs
./venv/bin/python code/step12_analyze_results.py \
    --input outputs/all_sessions32_fidelity_results.csv --tag 32

# Full audit
./venv/bin/python code/audit_precision.py
```

## A few things about the layout

- Scripts resolve the project root as `dirname(dirname(__file__))`, so `code/` has
  to stay flat. Move a script into a subfolder and the root silently resolves to
  `code/`, which breaks every path.
- `step02_find_sessions` and `step09_scale_multi_subject` get imported as modules
  by `step09`/`step10`, so their filenames have to stay valid Python identifiers.
  That's the reason the steps are named `stepNN_` and not `NN_`.
- `outputs/` is the working directory: regenerable, not part of the deliverable.
  The curated findings live in `results/`.
- `data/` holds the raw ds004395 download and is never committed.
