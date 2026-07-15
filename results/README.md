# Results — AI Word-Embedding Fidelity and Memory

**Question.** When a subject studies a word, is that word more likely to be
remembered if the EEG pattern during encoding more accurately predicts the AI
(T5-large) embedding of that word?

**Answer.** No. Across 4, 16, and 32 subjects, embedding fidelity was **not**
significantly associated with later recall. The null held at every scale.

Dataset: PEERS / OpenNeuro [ds004395](https://openneuro.org/datasets/ds004395)
v2.0.0, `task-ltpFR2` (its 576-word pool matches the T5 embedding list exactly).
EEG is 128–129 channel EGI, 500 Hz. Code: [`../code/`](../code/).

```
word appears  →  EEG during encoding  →  predicted T5 embedding
              →  compare to true embedding  →  does similarity predict recall?
```

## Headline result — 32 subjects, 64 sessions, 36,864 trials

Model: `recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`
(logistic mixed-effects; fidelity z-scored, so the odds ratio is per 1 SD).

| Quantity | Value |
| --- | --- |
| odds ratio (per 1 SD) | **0.986** |
| 95% interval | **[0.963, 1.009]** — includes 1.0 |
| p-value | 0.218 |
| remembered mean fidelity | 0.84501 |
| forgotten mean fidelity | 0.84553 |
| difference (rem − forg) | −0.00052 |

## The null is robust to scale

Only the number of subjects changed between runs; the pipeline was identical.

| | 4 subj | 16 subj | 32 subj |
| --- | --- | --- | --- |
| sessions | 8 | 32 | 64 |
| trials | 4,608 | 18,432 | 36,864 |
| odds ratio | 0.971 | 0.979 | 0.986 |
| 95% interval | [0.913, 1.033] | [0.947, 1.013] | [0.963, 1.009] |
| p-value | 0.354 | 0.220 | 0.218 |
| conclusion | no effect | no effect | no effect |

As subjects increase, the odds ratio drifts **toward 1.0** and the interval
narrows while staying centred on 1.0. More data sharpened the estimate onto
"no effect" rather than uncovering one.

## The important caveat

**Word-specific decoding was at chance** in all three runs:

| Check | Real | Shuffled-label control | Chance |
| --- | --- | --- | --- |
| true-word percentile | 0.4975 | 0.5011 | 0.50 |
| top-5 retrieval | 0.0074 | 0.0088 | ~0.0087 |
| top-10 retrieval | 0.0164 | 0.0180 | ~0.0174 |

Real performance matches the shuffled-label control, and the high raw cosine
(~0.845) is an artifact of a **common embedding direction** shared by all T5
vectors — not evidence of word decoding.

So the bottleneck is the **decoding stage, not the sample size**. Raw broadband
300–800 ms EEG with linear ridge cannot decode word identity above chance, so
there is no reliable fidelity signal for a memory effect to build on. Adding
subjects cannot rescue a signal that is not there. Strengthening the decoder —
the reference paper's EEG preprocessing and longer (~3 s) window, richer
time-frequency features, or a neural decoder — must come **before** re-testing
the memory question.

A null is therefore the expected, honest outcome here, and it is reported as
such: the value of this project is a trustworthy, reproducible end-to-end
method, not a positive finding.

## What's here

| Path | Contents |
| --- | --- |
| `summary_4_vs_16_vs_32_subjects.txt` | **Start here** — canonical write-up of all three runs |
| `final_results_4subjects.md` | Detailed write-up of the original 4-subject run |
| `methods_and_math.md` | Methods and the math behind each stage |
| `embeddings/` | The T5 embedding deliverables (below) |
| `figures/` | 6 figures, one per question a reader asks (below) |
| `tables/` | Model outputs and the trial-level fidelity tables |
| `summaries/` | Per-run model summaries (4 / 16 / 32 subjects) |
| `validation/` | Audits and independent reproducibility reruns |

### `embeddings/` — the T5 stage deliverable

| File | Contents |
| --- | --- |
| `peers_t5large_embeddings.npy` | The **576 × 1024** matrix, one row per PEERS word |
| `peers_word_order.csv` | `word`, `row_index` — the row order of the matrix |
| `peers_t5large_embeddings.csv` | Same matrix, human-readable |
| `peers_words.csv` | The 576-word PEERS pool |
| `embedding_metadata.json` | Model, layer, and token-handling provenance |

Built with `google-t5/t5-large`, `T5EncoderModel` (encoder, **not** decoder),
middle encoder layer `hidden_states[12]`, subword tokens averaged, EOS and pad
excluded.

### `figures/` — one figure per question

| Figure | Question it answers |
| --- | --- |
| `pipeline_flow.png` | What is the method? |
| `remembered_vs_forgotten_fidelity.png` | Do remembered and forgotten words differ? (no — near-identical) |
| `decoding_chance_check.png` | Is the decoder doing anything? (no — retrieval sits at chance) |
| `raw_vs_centered_metrics.png` | Why is raw cosine high? (common-direction artifact) |
| `final_model_odds_ratios.png` | Does the model back that up? (all intervals span OR = 1) |
| `scaling_progression.png` | Does more data change it? (OR → 1.0 as intervals tighten) |

Figures and the `*_32subjects` tables are rendered from the **32-subject**
headline run by `code/step13_build_results_package.py`, which reads only the
committed tables in `tables/` (re-runnable on a fresh clone).

### `validation/`

The result was independently re-derived. `rerun_full_comparison_report.md`
records a full rerun that reproduced the 4-subject numbers **exactly** (odds
ratio 0.9714, p 0.3544, verdict PASS). `independent_redo_comparison.txt` is a
second implementation that trains ridge per *subject* rather than per
*subject/session* and reaches the same conclusion. `precision_audit.md`,
`audit_report.md`, and `reproducibility_checklist.md` cover the end-to-end
structural and numeric audits.
