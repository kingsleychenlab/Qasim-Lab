# One-Page Summary — AI Word-Embedding Fidelity and Memory

## Goal
Test whether a studied word is more likely to be remembered when the EEG response
during encoding more accurately predicts that word's AI (T5-large) embedding. We
call that accuracy **EEG-to-AI embedding fidelity**.

## Method (pipeline)
`word appears → EEG 300–800 ms after onset → ridge regression predicts the T5
embedding → cosine(predicted, true) = embedding_fidelity → test whether fidelity
predicts later recall.`

- **Embeddings:** 576 PEERS words → T5-large, encoder only, middle layer
  `hidden_states[12]`, subword-averaged, EOS/pad excluded → **576 × 1024** matrix.
- **EEG:** 300–800 ms window (`start = sample+150`, `stop = sample+400`,
  exclusive) → **129 ch × 250 tp = 32,250** features per trial, raw 500 Hz.
- **Decoding:** per subject/session ridge (**α = 10,000**), held-out-trial CV,
  scaler fit on train only, one out-of-fold prediction per trial.

## Data
PEERS / OpenNeuro **ds004395**, ltpFR2 task. **4 subjects · 8 sessions · 4,608
trials** (2,423 recalled, 2,185 forgotten). Sessions: LTP269 [12,20],
LTP293 [5,22], LTP299 [2,6], LTP303 [10,22].

## Final model
```
recalled ~ embedding_fidelity + session + (1|subject) + (1|word)
```
Logistic mixed-effects; `embedding_fidelity` z-scored (odds ratio per 1 SD).

## Result
| | |
| --- | --- |
| coefficient | −0.0291 |
| odds ratio | 0.971 (95% CI [0.913, 1.033]) |
| p | ≈ 0.354 |
| remembered mean fidelity | 0.84627 |
| forgotten mean fidelity | 0.84711 |
| difference | −0.00084 |

Raw cosine looked high (~0.847) but is inflated by a common embedding direction;
word-specific / retrieval metrics were **at chance** (real ≈ shuffled control).
Fully audited (133/133 checks), independently reproduced, and robust to a
per-subject-pooled redo (OR 0.9695, p = 0.3236).

## Conclusion
> The project tested whether later-remembered words showed higher EEG-to-AI
> embedding fidelity than forgotten words. The final session-aware logistic
> mixed-effects model did not support this prediction: embedding fidelity was
> not significantly associated with later recall.

## Next steps
Adopt the reference paper's EEG preprocessing and longer (~3 s) window; use
balanced top-k retrieval as the primary decoding metric; move to richer features
(time-frequency band power, spatial filtering) or a small neural decoder; add more
subjects/sessions — then re-test the memory model once decoding clears chance.
