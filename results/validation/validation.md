# Validation

Everything in the pipeline was checked before the final model was run, and then
the whole thing was re-derived from scratch a couple of different ways. This is
the summary; the raw logs are the `.txt` and `.csv` files in this folder.

## Bottom line

- **Precision audit: PASS.** 133 of 133 programmatic checks passed. The T5
  recompute, the recall re-derivation, and the EEG re-extraction all verified.
- **Follows the original plan: yes.** T5-large encoder, middle layer
  `hidden_states[12]`, subword-averaged with EOS/pad left out, 576 × 1024 targets;
  per-subject/session ridge with held-out-trial CV; cosine fidelity; final model
  `recalled ~ embedding_fidelity + session + (1|subject) + (1|word)` with
  `embedding_fidelity = raw cosine`.
- **Does the result support the prediction: no.** Fidelity was not significantly
  tied to recall. A clean null.

## Data integrity (4-subject audit)

| Quantity | Value | Check |
| --- | --- | --- |
| subjects | 4 (LTP269, LTP293, LTP299, LTP303) | PASS |
| sessions | 8 (2 per subject: 269[12,20], 293[5,22], 299[2,6], 303[10,22]) | PASS |
| trials | 4608 (576 × 8) | PASS |
| recalled / forgotten | 2423 / 2185 | PASS |
| unique words / session | 576, all in `peers_word_order.csv` | PASS |
| missing values | none in subject/session/trial/serialpos/word/recalled/embedding_fidelity | PASS |
| NaN / Inf | none in any numeric column | PASS |
| `embedding_fidelity == raw_cosine` | true (allclose) | PASS |

## The math, recomputed

- **T5 embedding.** `e_w = (1/k) Σ h_i`, where `h_i` is `hidden_states[12]` of the
  T5-large encoder for the real subword tokens (EOS and pad excluded), `e_w ∈ R^1024`.
  Recomputing ACTOR / AIRPLANE / ZIPPER from the model reproduces the saved rows to
  float32 precision (max abs diff 3.97e-4 on elements of magnitude ~600, so ~4e-7
  relative). Verified on both MPS and CPU.
- **EEG window.** `start = sample + int(0.300·500) = sample+150`,
  `stop = sample + int(0.800·500) = sample+400` (exclusive), so 250 timepoints;
  129 channels × 250 = 32,250 features, flattened channel-major. Re-extracting from
  the EDF reproduces `X_eeg` bit-exactly once cast to the stored float32 dtype, so
  the stored features are genuine EDF samples, not approximations.
- **Ridge objective.** `min_W ‖Y − XW‖² + α‖W‖²`, `α = 10000`, solver `svd`, fit
  per subject/session, `StandardScaler` on train folds only, one out-of-fold
  prediction per trial.
- **Cosine fidelity.** `fidelity_i = (ŷ_i · y_i)/(‖ŷ_i‖‖y_i‖)`.
- **Memory model.** `recalled_i ~ embedding_fidelity_i + session_i + (1|subject_i)
  + (1|word_i)`, logistic, predictor z-scored so the OR is per 1 SD.

## Stage-by-stage audit

| Stage | Result | Evidence |
| --- | --- | --- |
| T5 embeddings | PASS | 576×1024, no NaN/Inf, norms > 0; recompute Δ ≤ 3.97e-4 (float32). `outputs/t5_embedding_audit.txt` |
| Word order | PASS | `row_index` 0–575, 576 unique words, consistent with the embeddings. `peers_word_order.csv` |
| Recall labels | PASS | 576/576 re-derived per session; free recall only, no `RECOG_*` used; matched within trial/list. `recall_label_audit.txt` |
| EEG windows | PASS | +150…+400 exclusive → 250 tp; 129×250 = 32250; in bounds in all sessions |
| EEG extraction | PASS | sampled rows bit-exact vs the EDF (worst diff 0.0 after float32). `eeg_extraction_audit.txt` |
| Ridge CV | PASS | per session, KFold shuffle, scaler on train only, α = 1e4, svd, one OOF prediction per trial |
| Fidelity | PASS | `embedding_fidelity == raw_cosine`; cosine(pred, true) |
| Metric controls | PASS | percentiles ∈ [0,1]; true-word ≈ 0.498, centered ≈ 0.525 (both chance); shuffled control present |
| Final model | PASS | logistic GLMM, session FE, (1|subject) + (1|word), z-scored; numbers match |
| Result wording | PASS | every file says the prediction wasn't supported; no overstatement |

A couple of notes on the trickier ones:

- **T5 reproduction.** The ~2e-4 absolute difference looks bigger than a naive
  "< 1e-5" target only because T5 elements are large (max magnitude ~600). The
  relative error (~4e-7) is at the float32 floor.
- **Recall derivation.** Free recall only. A word is `recalled = 1` if and only if
  a `REC_WORD` event with the same `item_num` shows up in the same trial/list.
  Intrusions (`item_num == −1`) never match, and recognition data is never touched.
- **Window-fit gate.** Several `ltpFR2` EDFs are truncated relative to the
  behavioral log. A session was only kept if all 576 words' full 300-800 ms window
  fits inside the recording (`stop_sample < n_times`), checked with MNE.

## Statistical result (4-subject, main metric = raw cosine)

| Quantity | Value |
| --- | --- |
| coefficient (per 1 SD) | −0.0291 |
| odds ratio (per 1 SD) | 0.971 |
| 95% interval | [0.913, 1.033] |
| p-value (approx) | ≈ 0.354 |
| remembered mean fidelity | 0.84627 |
| forgotten mean fidelity | 0.84711 |
| difference | −0.00084 |
| conclusion | no significant effect — prediction not supported |

Supplementary, as a sanity check only: centered_cosine OR 0.995 (p 0.863),
true_word_percentile OR 0.976 (p 0.435), centered_true_word_percentile OR 1.010
(p 0.742). All null.

## Full rerun

I re-ran the entire audited pipeline from scratch. Every number came back
identical.

| Quantity | Original | Rerun |
| --- | --- | --- |
| subjects / sessions / trials | 4 / 8 / 4608 | 4 / 8 / 4608 |
| recalled / forgotten | 2423 / 2185 | 2423 / 2185 |
| mean embedding_fidelity | 0.84667 | 0.84667 |
| remembered / forgotten mean | 0.84627 / 0.84711 | 0.84627 / 0.84711 |
| coefficient (per SD) | −0.0291 | −0.0291 |
| odds ratio | 0.9714 | 0.9714 |
| interval | [0.913, 1.033] | [0.913, 1.033] |
| p-value | 0.3544 | 0.3544 |
| conclusion | no effect | no effect |

Stage-by-stage the rerun matched too: embeddings, encoding trials, EEG windows,
X/Y inputs, the fidelity table, and the final model all PASS, with the conclusion
unchanged. None of the failure conditions I checked for came up — no different
subject/session set, no missing session, no word-order change, no recall-label
change, no invalid EEG window, no train/test leakage, no shape mismatch, and no
change to the coefficient or the final conclusion. Details in
`rerun_final_model_comparison.csv` and `rerun_fidelity_table_comparison.txt`.

## Second independent implementation

As a separate check, `independent_redo_comparison.txt` is a from-scratch
reimplementation that fits ridge per *subject* rather than per *subject/session*.
It reaches the same conclusion (odds ratio 0.9695, p 0.3236), which says the null
isn't an artifact of how the folds were organized. Per-session detail is in
`independent_redo_per_session.csv`.

## Reproducing it

Two virtual environments:

- **Project env** — the EEG and modeling side: `mne`, `mne-bids`, `numpy`,
  `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `requests`.
- **T5 env** — embeddings only: `torch`, `transformers`, `sentencepiece`, `numpy`,
  `pandas`, `tqdm`. You only need this to rebuild or audit the T5 embeddings.

```bash
# project env
python3 -m venv venv && source venv/bin/activate
pip install mne mne-bids numpy pandas scikit-learn statsmodels matplotlib requests

# T5 env (only for embeddings / the T5 audit)
python3 -m venv venv_t5 && source venv_t5/bin/activate
pip install torch transformers sentencepiece numpy pandas tqdm
```

The raw data is OpenNeuro **ds004395** (PEERS) from the public S3 mirror. It's
pulled one session at a time (never the full 8.7 TB) into `data/ds004395/`.

```bash
# 0. T5 embeddings (T5 env)        -> peers_t5large_embeddings.npy (576x1024)
python code/step04_extract_t5_embeddings.py

# 1. find and process valid ltpFR2 sessions (project env)
#    downloads + encoding trials + Y_t5 + X_eeg + ridge metrics per session
python code/step10_scale_multi_session.py --n-subjects 4 --sessions-per-subject 2
#    -> outputs/all_sessions_fidelity_results.csv  (4608 rows, the final input table)

# 2. the final memory model
python code/step11_run_memory_model.py

# 3. results package (tables + figures)
python code/step13_build_results_package.py

# 4. audits
python code/audit_recall_labels.py
python code/audit_eeg_extraction.py
python code/audit_precision.py
python code/audit_t5_embeddings.py        # T5 env
```

**Files too big for GitHub** are git-ignored and regenerable: `data/` and all
`*.edf` / `*.bdf` (raw EEG, ~0.5-0.7 GB each), `outputs/subjects/`, the big
`outputs/*.npy` arrays, `venv/`, `__pycache__/`, and `.DS_Store`. Everything else
you need is small and tracked (the scripts, the 576 × 1024 embedding matrix at
~2.3 MB, `all_sessions_fidelity_results.csv`, the model outputs, and `results/`)
or downloadable from OpenNeuro with the commands above.

**Determinism.** CV uses `KFold(shuffle=True, random_state=42)`; the shuffled
control uses seed 2024. The T5 recompute agrees with the saved matrix to float32
precision (~4e-7 relative), and the EEG features re-extract bit-exactly from the
EDF.
