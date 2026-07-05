# Results Index

Everything in `results/` and what it means. All files here are derived
read-only from `outputs/` and the project root; no pipeline output was modified.

## Reports (Markdown)

| File | What it is |
| --- | --- |
| `README.md` | Project overview: goal, dataset, pipeline, final conclusion |
| `methods_and_math.md` | The math with equations (T5 embedding, EEG window, ridge, fidelity, memory model, sanity checks) |
| `pipeline_summary.md` | Step-by-step list of files produced + shapes/dimensions |
| `audit_report.md` | All validation/audit checks with a PASS/FAIL table |
| `final_results.md` | The final statistical model, numbers, and conclusion |
| `portfolio_summary.md` | Polished, honest project summary for portfolio/GitHub |
| `results_index.md` | This file |

## Summary tables (CSV)

| File | Columns |
| --- | --- |
| `summary_subjects_sessions.csv` | subject, session, n_trials, recalled, forgotten, recall_rate, mean_embedding_fidelity, mean_centered_percentile |
| `remembered_vs_forgotten_summary.csv` | group, n, mean/sd embedding_fidelity, mean/sd centered_true_word_percentile |
| `final_model_table.csv` | metric, coefficient, odds_ratio, ci_lower, ci_upper, p_value, conclusion (raw_cosine main + centered supplementary) |

## Figures (`figures/`)

| File | What it shows |
| --- | --- |
| `pipeline_flow.png` | Visual pipeline: PEERS words → T5 → EEG 300–800 ms → ridge → cosine fidelity → memory model |
| `embedding_fidelity_histogram.png` | Distribution of `embedding_fidelity` (raw cosine) across 4608 trials |
| `remembered_vs_forgotten_fidelity.png` | Boxplot: remembered vs forgotten fidelity — nearly identical |
| `subject_session_fidelity.png` | Mean fidelity per subject/session |
| `recall_rate_by_subject_session.png` | Recall rate per subject/session |
| `raw_vs_centered_metrics.png` | Raw cosine (high, ~0.85) vs centered word-specific percentile (chance ~0.5) |
| `decoding_chance_check.png` | Retrieval percentiles vs the chance line at 0.5 |
| `final_model_odds_ratios.png` | Forest plot of odds ratios with the OR = 1 reference line |
| `recalled_counts.png` | Recalled vs forgotten trial counts |
| `t5_embedding_norms.png` | Histogram of T5 embedding L2 norms (large norms drive raw-cosine inflation) |

## Headline numbers

- Subjects **4** · Sessions **8** · Trials **4608** · Recalled/Forgotten **2423 / 2185**
- Final model (raw cosine): OR **0.971** per 1 SD, 95% CI **[0.913, 1.033]**, p ≈ **0.354**
- Remembered vs forgotten mean fidelity: **0.84627 vs 0.84711** (diff **−0.00084**)
- **Conclusion: the key prediction was not supported in this analysis.**
