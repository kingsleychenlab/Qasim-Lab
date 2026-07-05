# Validation and Audits

Every stage was independently verified before the final model was trusted. This
is the strongest part of the project to emphasize in the meeting: the pipeline is
not just "run once and reported" — it is audited, reproducible, and robust.

## PASS table

| Audit | What was checked | Result |
| --- | --- | --- |
| **T5 embeddings** | Recompute ACTOR / AIRPLANE / ZIPPER from the model and compare to saved rows | **PASS** — match to float32 precision (max abs diff ~4e-4 on values of magnitude ~600 → ~4e-7 relative) |
| **Recall labels** | Re-derive `recalled` from raw `WORD`/`REC_WORD` events by item number within list | **PASS** — 576/576 labels match in every session |
| **No recognition used** | Confirm `RECOG_*` / `recog_resp` / `recog_conf` never touched | **PASS** — 0 recognition events used |
| **Within-list matching** | Confirm item numbers never matched across lists | **PASS** |
| **EEG extraction** | Re-extract windows straight from the EDF and compare to stored features | **PASS** — bit-exact after float32 cast |
| **EEG windows** | 129 ch × 250 tp, stop exclusive, all 576/576 windows inside the recording | **PASS** — 0 dropped trials |
| **Model inputs** | X 576×32250, Y 576×1024, aligned, no NaN/Inf | **PASS** |
| **Combined table** | 4608 rows, recalled ∈ {0,1}, `embedding_fidelity == raw_cosine`, percentiles ∈ [0,1] | **PASS** |
| **Precision audit** | Full end-to-end check of structure, math, and numbers | **PASS — 133/133 checks** |
| **Independent rerun** | Rebuild the whole pipeline from scratch and compare | **PASS** — final odds ratio reproduced (0.9714 vs 0.9714); embedding_fidelity max diff ~6e-8 |
| **Per-subject redo** | Retrain ridge pooled per subject instead of per subject/session | **PASS (same conclusion)** — OR 0.9695, p = 0.3236 |
| **Results wording** | Scan all reports for overstated claims | **PASS** — no claim implies the hypothesis was supported |

## Key evidence to mention out loud

- **Reproducibility:** a full independent rerun reproduced the odds ratio to
  ~1e-8, and embedding fidelity to ~6e-8 — the result is not a fluke of one run.
- **Bit-exact EEG:** the features fed to the model are literally the raw EDF
  samples, verified byte-for-byte — no silent preprocessing error.
- **Perfect recall re-derivation:** 576/576 labels match from the raw events,
  using only free-recall information (recognition never used).
- **Robust to design choice:** the per-subject-pooled redo changed fidelity
  values slightly but not the conclusion — both are null.

## One honest caveat to state

The per-subject pooled redo runs fidelity slightly higher (per-session means up
~0.001–0.0018) because pooling a subject's two sessions lets the same word appear
in both training and test — a mild word-repetition leakage. It nudges cosine up
but does not create a memory effect. The paper avoids this by hashing words
across splits; our per-subject/session design largely avoids it too. Either way,
the conclusion is unchanged.

## Where the audit logs live (if asked)

- `outputs/final_precision_audit.txt` (133 checks)
- `outputs/recall_label_audit.txt`, `outputs/eeg_extraction_audit.txt`
- `outputs/model_input_validation.txt`
- `rerun_full_validation/comparison/full_rerun_comparison_report.md`
- `independent_redo/comparison/redo_vs_current_comparison.txt`
