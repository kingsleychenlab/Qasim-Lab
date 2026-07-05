# Final Precision Audit

A strict, end-to-end verification of the Neurolab project against the original
outline. Every number below was checked against the source CSV / model output;
every stage was independently recomputed. Source log:
`outputs/final_precision_audit.txt` (133 programmatic checks + external sections).

## A. Executive verdict

- **Status: PASS** — 133/133 programmatic checks pass; T5 recompute, recall
  re-derivation, and EEG re-extraction all verified.
- **Follows the original outline: YES** — T5-large encoder, middle layer
  `hidden_states[12]`, subword-averaged with EOS/pad excluded, 576×1024 targets;
  per-subject/session ridge (EEG → embedding) with held-out-trial CV; cosine
  fidelity; final model `recalled ~ embedding_fidelity + session + (1|subject) +
  (1|word)` with `embedding_fidelity = raw cosine`.
- **Does the result support the prediction: NO.** Embedding fidelity was **not**
  significantly associated with recall. This is a clean **null**.

## B. Data integrity

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

## C. Mathematical correctness

- **T5 embedding** — `e_w = (1/k) Σ_{i=1..k} h_i`, `h_i` = `hidden_states[12]`
  of the T5-large encoder for real subword tokens (EOS `</s>` and pad excluded),
  `e_w ∈ R^1024`. Recomputed from the model for ACTOR/AIRPLANE/ZIPPER → matches
  saved rows to float32 precision (max abs diff 3.97e-4; ~4e-7 relative).
- **EEG window** — `start = sample + int(0.300·500) = sample+150`,
  `stop = sample + int(0.800·500) = sample+400` (exclusive) → 250 timepoints;
  129 channels × 250 = 32,250 features, flattened channel-major (C-order).
  Re-extracted **bit-exactly** from the EDF for sampled rows in all 8 sessions.
- **Ridge objective** — `min_W ‖Y − XW‖² + α‖W‖²`, `α = 10000`, solver `svd`,
  fit per subject/session, `StandardScaler` on train folds only, one out-of-fold
  prediction per trial (verified in source and per-session metadata).
- **Cosine fidelity** — `fidelity_i = (ŷ_i · y_i)/(‖ŷ_i‖‖y_i‖)`.
- **Memory model** — `recalled_i ~ embedding_fidelity_i + session_i +
  (1|subject_i) + (1|word_i)`, logistic (binary), predictor z-scored (OR per 1 SD).

## D. Audit table

| Check | Status | Evidence | File |
| --- | --- | --- | --- |
| T5 embeddings | **PASS** | 576×1024, no NaN/Inf, norms>0; recompute Δ≤3.97e-4 (float32) | `peers_t5large_embeddings.npy`, `outputs/t5_embedding_audit.txt` |
| Word order | **PASS** | `row_index` 0–575, 576 unique words, consistent with embeddings | `peers_word_order.csv` |
| Recall labels | **PASS** | 576/576 re-derived match per session; no RECOG_* used; within-list only | `outputs/recall_label_audit.txt`, per-session `events.tsv` |
| EEG windows | **PASS** | +150…+400 exclusive → 250 tp; 129×250=32250; in-bounds all sessions | `scripts/extract_eeg_features.py` |
| EEG extraction | **PASS** | sampled rows bit-exact vs EDF (worst diff 0.0 after float32) | `outputs/eeg_extraction_audit.txt`, per-session `X_eeg.npy` |
| Ridge CV | **PASS** | per session, KFold shuffle, scaler on train only, α=1e4, svd, 1 OOF pred/trial | `scripts/ridge_corrected_metrics.py`, per-session `ridge_corrected_metadata.json` |
| Fidelity calculation | **PASS** | `embedding_fidelity == raw_cosine`; cosine(pred,true) | `outputs/all_sessions_fidelity_results.csv` |
| Corrected metric controls | **PASS** | percentiles ∈ [0,1]; true-word ≈0.498, centered ≈0.525 (chance); shuffled control present | `outputs/all_sessions_fidelity_results.csv` |
| Final model | **PASS** | logistic GLMM, session FE, (1|subject)+(1|word), z-scored; numbers match | `outputs/final_memory_model_*` |
| Result wording | **PASS** | all files say prediction not supported; no overstatement | `results/*.md` |

## E. Final statistical result (main metric = raw cosine = embedding_fidelity)

| Quantity | Value |
| --- | --- |
| coefficient (per 1 SD) | **−0.0291** |
| odds ratio (per 1 SD) | **0.971** |
| 95% interval | **[0.913, 1.033]** |
| p-value (approx) | **≈ 0.354** |
| remembered mean fidelity | **0.84627** |
| forgotten mean fidelity | **0.84711** |
| difference | **−0.00084** |
| conclusion | **no significant effect — prediction not supported** |

Supplementary (sanity only): centered_cosine OR 0.995 (p 0.863), true_word_percentile
OR 0.976 (p 0.435), centered_true_word_percentile OR 1.010 (p 0.742) — all null.

## F. Limitations

- Only **4 subjects / 8 sessions** — underpowered; not a confirmatory test.
- **Raw broadband EEG** features only (no time-frequency band power, baseline
  correction, or spatial filtering).
- **Word-specific decoding was at chance** (retrieval percentile ≈ 0.5; real ≈
  shuffled), so there was no genuine decoding signal for memory to build on.
- **Raw cosine is inflated** by the common embedding direction (large T5 norms),
  so its high absolute value (~0.85) does not indicate word decoding.
- This is a **rigorous null for this implementation**, not proof that the broader
  hypothesis is impossible.

## G. Exact final wording

> The project tested whether later-remembered words showed higher EEG-to-AI
> embedding fidelity than forgotten words. The final session-aware logistic
> mixed-effects model did not support this prediction: embedding fidelity was not
> significantly associated with later recall.

---

**FINAL AUDIT PASS: the project is accurate, reproducible, and honestly reported
as a null result.**
