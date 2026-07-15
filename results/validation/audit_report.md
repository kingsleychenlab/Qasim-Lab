# Audit Report

Every stage of the pipeline was independently verified before the final model
was run. The source audit logs live under `outputs/`; this is the summary.

## PASS / FAIL table

| Audit | Check | Result |
| --- | --- | --- |
| T5 embeddings | Recomputed ACTOR / AIRPLANE / ZIPPER from the model reproduce the saved rows | **PASS** — agree to float32 precision (max abs diff ≈ 2–4e-4 on elements of magnitude ~600 → ~4e-7 relative) |
| Recall labels | `recalled` re-derived from `WORD`/`REC_WORD` by `item_num` within trial/list | **PASS** — 576/576 labels match |
| Recall labels | No recognition events (`RECOG_*`, `recog_resp`, `recog_conf`) used | **PASS** — 0 recognition events used |
| Recall labels | Matching restricted within trial/list (no cross-trial matches) | **PASS** — verified (1 cross-trial-only candidate correctly excluded) |
| EEG extraction | `X_eeg` rows re-extracted from the EDF for rows 0/100/300/575 | **PASS** — bit-exact after float32 cast (residual ≈ 1e-9 = float64→float32) |
| EEG extraction | Every trial window is 129 × 250, extracted, 0 dropped | **PASS** — 576/576 valid, 0 dropped |
| Model inputs | X_eeg 576 × 32250, Y_t5 576 × 1024, aligned to trial_metadata | **PASS** |
| Model inputs | No NaN / Inf in X or Y | **PASS** |
| Combined table | `all_sessions_fidelity_results.csv` has no NaN / Inf | **PASS** |
| Combined table | `embedding_fidelity == raw_cosine` | **PASS** |
| Combined table | `recalled ∈ {0,1}`; ranks ∈ [1, 576]; percentiles ∈ [0, 1]; top-k ∈ {0,1} | **PASS** |
| Combined table | Multiple subjects, ≥2 sessions per subject | **PASS** — 4 subjects × 2 sessions |
| Session validity | task==ltpFR2, WORD==576, 576/576 peers coverage, 576/576 valid 300–800 ms windows | **PASS** — enforced per session before inclusion |

## Notes on the audits

- **T5 reproduction.** The absolute difference (~2e-4) looks larger than a naive
  "< 1e-5" target only because T5 elements are large (max magnitude ~600); the
  *relative* error (~4e-7) is at the float32 floor. Verified on both MPS and CPU.
- **Recall derivation.** Free recall only. A word is `recalled = 1` iff a
  `REC_WORD` event with the same `item_num` appears in the same trial/list.
  Intrusions (`item_num == −1`) never match. Recognition data is never touched.
- **EEG fidelity.** Re-reading the EDF and re-slicing reproduces `X_eeg` exactly
  once cast to the stored float32 dtype — the stored features are genuine EDF
  samples, not approximations.
- **Window-fit gate.** Several ltpFR2 EDFs are truncated relative to the
  behavioral log; sessions were only included if all 576 words' full 300–800 ms
  window fits inside the recording (`stop_sample < n_times`), checked with MNE.

Source logs: `outputs/recall_label_audit.txt`, `outputs/eeg_extraction_audit.txt`,
`outputs/model_input_validation.txt`, `outputs/all_sessions_summary.txt`,
`outputs/ridge_corrected_summary.txt`. (The T5 reproduction audit was run via
`scripts/audit_t5_embeddings.py`.)
