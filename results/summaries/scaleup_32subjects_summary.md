# 32-Subject Scale-Up — Results

```
================================================================================
32-SUBJECT SCALE-UP  (vs 16- and 4-subject)
================================================================================
Pipeline IDENTICAL across all runs (PEERS ltpFR2, T5-large middle-layer
embeddings, 300-800 ms window, per subject/session ridge alpha=10000, held-out
CV, cosine fidelity, recalled ~ embedding_fidelity + session + (1|subject) +
(1|word)). Only the number of subjects changed.

PROGRESSION (main model: raw cosine, OR per 1 SD)
  metric                 4-subj      16-subj      32-subj
  subjects                    4           16           32
  sessions                    8           32           64
  trials                   4608        18432        36864
  recalled                 2423         9980        17040
  forgotten                2185         8452        19824
  odds ratio             0.9714       0.9792       0.9856
  coefficient           -0.0291      -0.0211      -0.0145
  ci_low                 0.9134       0.9468       0.9630
  ci_high                1.0330       1.0127       1.0086
  p-value                0.3544       0.2199       0.2178
  remembered            0.84627      0.84337      0.84501
  forgotten             0.84711      0.84367      0.84553
  diff                 -0.00084     -0.00030     -0.00052
  conclusion (32-subj): no different

WORD-SPECIFIC DECODING SANITY (32-subject, averaged over sessions)
  chance: percentile 0.5, top5 ~0.0087, top10 ~0.0174
  true_word_rank                  : 289.9623
  true_word_percentile            : 0.4975
  top1_correct                    : 0.0011
  top5_correct                    : 0.0074
  top10_correct                   : 0.0164
  centered_true_word_percentile   : 0.5183
  centered_top5_correct           : 0.0025
  centered_top10_correct          : 0.0093

REAL vs SHUFFLED-LABEL CONTROL (32-subject)
  metric                               real  shuffled
  true_word_percentile               0.4975    0.5011
  centered_true_word_percentile      0.5183    0.5046
  top5_correct                       0.0074    0.0088
  top10_correct                      0.0164    0.0180

VERDICT
  Memory effect: STAYS NULL — no significant fidelity->recall effect.
  Word-specific decoding: at chance (percentile ~0.5, real ~= shuffled).

FINAL CONCLUSION
  The project tested whether later-remembered words showed higher EEG-to-AI
  embedding fidelity than forgotten words. With 32 subjects, the
  session-aware logistic mixed-effects model still did not support this prediction: embedding fidelity was not significantly associated with later recall.
================================================================================
```
