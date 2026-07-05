# Final Results

## The model

```
recalled ~ embedding_fidelity + session + (1 | subject) + (1 | word)
```

- Logistic mixed-effects (binary `recalled`), `BinomialBayesMixedGLM`.
- **`embedding_fidelity = raw cosine`** (the metric named in the project
  outline), z-scored → odds ratio is per 1 SD increase.
- `session` is a fixed effect; random intercepts for `subject` and `word`.
- Data: 4 subjects, 8 sessions, 4608 trials (2423 recalled / 2185 forgotten).

## Main result (embedding_fidelity = raw cosine)

| Quantity | Value |
| --- | --- |
| coefficient (per 1 SD) | **−0.0291** |
| odds ratio (per 1 SD) | **0.971** |
| 95% interval | **[0.913, 1.033]** |
| p-value (approx) | **≈ 0.354** |
| remembered mean fidelity | **0.84627** |
| forgotten mean fidelity | **0.84711** |
| difference (rem − forg) | **−0.00084** |

The 95% interval comfortably includes an odds ratio of 1.0, and the two group
means are essentially identical.

## Supplementary metrics (sanity checks, not the main result)

| Metric | Odds ratio | 95% interval | p | Conclusion |
| --- | --- | --- | --- | --- |
| raw_cosine (MAIN) | 0.971 | [0.913, 1.033] | 0.354 | no significant effect |
| centered_cosine | 0.995 | [0.936, 1.058] | 0.863 | no significant effect |
| true_word_percentile | 0.976 | [0.918, 1.038] | 0.435 | no significant effect |
| centered_true_word_percentile | 1.010 | [0.950, 1.074] | 0.742 | no significant effect |

All four metrics — and both the mixed-effects and cluster-robust fits — agree:
no significant relationship between embedding fidelity and recall.

## Conclusion

**The key prediction was not supported in this analysis.** Remembered words did
**not** show higher EEG-to-AI embedding fidelity than forgotten words.

This null is the expected outcome given the decoding checks: raw cosine was
inflated by the common embedding direction, and **word-specific decoding was at
chance** (retrieval percentile ≈ 0.5, real ≈ shuffled) across all sessions. With
no genuine word-level decoding signal, there is nothing for a memory effect to
build on.

The pipeline was completed and audited end-to-end (see `audit_report.md`); the
deliverable is a trustworthy negative result, not a positive finding.

## Caveats

- Only **4 subjects / 8 sessions** — underpowered for a confirmatory test.
- Features are **raw broadband voltage**; no time-frequency, baseline, or spatial
  filtering was applied in this first pass.
- A non-null effect here would warrant strong skepticism, not celebration.
