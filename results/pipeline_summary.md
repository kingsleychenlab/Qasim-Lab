# Pipeline Summary — files produced at each step

Every stage is a small, parameterized script under `scripts/`, reading and
writing files under `outputs/`. This document lists what each stage produced.

## Stage 0 — T5 word embeddings (project root)

| File | Shape / meaning |
| --- | --- |
| `peers_t5large_embeddings.npy` | **576 × 1024** float32 — T5-large `hidden_states[12]`, EOS/pad excluded, subword-averaged |
| `peers_word_order.csv` | 576 rows `row_index, word` (uppercase) |
| `embedding_metadata.json` | model, layer, dims, exclusions, package versions |

## Stage 1 — recall labels (per session)

`create_encoding_trials.py` derives the binary recall label from free-recall
events only (`WORD` presented, `REC_WORD` recalled, matched by `item_num`
within trial/list — recognition events are never used).

| File | Meaning |
| --- | --- |
| `outputs/encoding_trials.csv` | canonical session: 576 rows, one per studied word, with `recalled ∈ {0,1}` |

## Stage 2 — model inputs (per session)

| File | Shape / meaning |
| --- | --- |
| `outputs/Y_t5.npy` | **576 × 1024** — T5 targets aligned to trial order |
| `outputs/X_eeg.npy` | **576 × 32250** per session — raw 300–800 ms EEG (129 ch × 250 tp) |
| `outputs/trial_metadata.csv` | per-trial metadata + `start_sample`/`stop_sample`/`n_timepoints` |
| `outputs/model_input_validation.txt` | shape / NaN / alignment checks (all pass) |

## Stage 3 — decoding + fidelity (per session)

`ridge_corrected_metrics.py` runs held-out-trial ridge CV and computes raw
cosine plus the word-specific sanity metrics and shuffled control.

| File | Meaning |
| --- | --- |
| `outputs/subjects/<sub>_<ses>/fidelity_results_corrected.csv` | per-trial fidelity metrics |
| `outputs/subjects/<sub>_<ses>/predicted_embeddings_corrected.npy` | 576 × 1024 out-of-fold predictions |

## Stage 4 — combined fidelity table (all sessions)

| File | Shape / meaning |
| --- | --- |
| `outputs/all_sessions_fidelity_results.csv` | **4608 rows** — 4 subjects × 2 sessions × 576 trials; `embedding_fidelity == raw_cosine` plus supplementary metrics |

## Stage 5 — final memory model

| File | Meaning |
| --- | --- |
| `outputs/final_memory_model_summary.txt` | full model report |
| `outputs/final_memory_model_results.csv` | coefficients / OR / CI / p for all metrics |
| `outputs/final_memory_model_metadata.json` | machine-readable model metadata |

## Dataset dimensions (final combined table)

| Quantity | Value |
| --- | --- |
| T5 embedding matrix | **576 × 1024** |
| X_eeg per session | **576 × 32250** |
| all_sessions_fidelity_results.csv | **4608 rows** |
| subjects | **4** (LTP269, LTP293, LTP299, LTP303) |
| sessions | **8** (2 per subject) |
| trials | **4608** |
| recalled / forgotten | **2423 / 2185** (recall rate 0.526) |

Sessions used: LTP269 [12, 20] · LTP293 [5, 22] · LTP299 [2, 6] · LTP303 [10, 22].
