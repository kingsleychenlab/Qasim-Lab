# 16-Subject Scale-Up — Results Summary

```
==============================================================================
16-SUBJECT SCALE-UP RESULTS  (vs 4-subject)
==============================================================================
Pipeline IDENTICAL to the 4-subject run (PEERS ltpFR2, T5-large middle-layer
embeddings, 300-800 ms window, per subject/session ridge alpha=10000, held-out
CV, cosine fidelity, model recalled ~ embedding_fidelity + session +
(1|subject) + (1|word)). Only the number of subjects changed.

DATA
  subjects : 16   (4-subject: 4)
  sessions : 32   (4-subject: 8)
  trials   : 18432   (4-subject: 4608)
  recalled/forgotten : 9980 / 8452   (4-subject: 2423 / 2185)

MAIN MEMORY MODEL  (embedding_fidelity = raw cosine, z-scored; OR per 1 SD)
                           16-subject    4-subject
  coefficient                 -0.0211      -0.0291
  odds ratio                   0.9792       0.9714
  95% interval           [0.947,1.013] [0.913,1.033]
  p-value                      0.2199       0.3544
  remembered mean             0.84337      0.84627
  forgotten mean              0.84367      0.84711
  difference                 -0.00030     -0.00084
  conclusion (16-subj): no different

WORD-SPECIFIC DECODING SANITY CHECKS (16-subject, averaged over all sessions)
  chance references: percentile 0.5, top5 ~0.0087, top10 ~0.0174
  true_word_rank                  : 289.8213
  true_word_percentile            : 0.4977
  top1_correct                    : 0.0009
  top5_correct                    : 0.0071
  top10_correct                   : 0.0162
  centered_true_word_percentile   : 0.5186
  centered_top5_correct           : 0.0022
  centered_top10_correct          : 0.0087

REAL vs SHUFFLED-LABEL CONTROL (16-subject, averaged over sessions)
  metric                               real  shuffled
  true_word_percentile               0.4977    0.5011
  centered_true_word_percentile      0.5186    0.5060
  top5_correct                       0.0071    0.0088
  top10_correct                      0.0162    0.0184

VERDICT
  Memory effect: STAYS NULL — no significant fidelity->recall effect, same as 4-subject.
  Word-specific decoding: at chance (percentile ~0.5, real ~= shuffled) — same as 4-subject.

FINAL CONCLUSION
  The project tested whether later-remembered words showed higher EEG-to-AI
  embedding fidelity than forgotten words. With 16 subjects, the session-aware
  logistic mixed-effects model still did not support this prediction: embedding fidelity was not significantly associated with later recall.
==============================================================================
```
