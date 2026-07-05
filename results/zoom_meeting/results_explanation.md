# Results Explanation (in depth)

## The final model

```
recalled ~ embedding_fidelity + session + (1|subject) + (1|word)
```

Logistic mixed-effects; `embedding_fidelity` (raw cosine) z-scored so the odds
ratio is per 1 standard deviation of fidelity.

| Quantity | Value | Plain meaning |
| --- | --- | --- |
| coefficient (per 1 SD) | **−0.0291** | Direction/size of the fidelity→recall effect on the log-odds scale. Essentially zero, and slightly negative. |
| odds ratio | **0.971** | For each 1-SD increase in fidelity, the odds of recall multiply by 0.971 — i.e. **~3% lower**, but not distinguishable from 1.0 (no effect). |
| 95% interval | **[0.913, 1.033]** | The plausible range for the odds ratio. It **contains 1.0**, so we cannot rule out "no effect." |
| p-value | **≈ 0.354** | ~35% chance of seeing an effect this size (or larger) if the true effect were zero. Far from significant (which would be p < 0.05). |
| remembered mean fidelity | **0.84627** | Average fidelity for later-recalled words. |
| forgotten mean fidelity | **0.84711** | Average fidelity for later-forgotten words. |
| difference | **−0.00084** | Remembered minus forgotten. Tiny, and slightly negative. |

## What each number means

- **Coefficient (−0.0291).** In a logistic model, the coefficient is the change
  in log-odds of recall per 1-SD increase in fidelity. −0.029 is near zero; its
  sign is slightly negative, meaning if anything higher fidelity went with
  *slightly less* recall — but this is noise, not a real effect.
- **Odds ratio (0.971).** This is just `exp(coefficient)`, an easier scale.
  1.0 = no effect. 0.971 is a hair below 1.0. To call an effect real, we'd want
  the whole interval on one side of 1.0; here it straddles 1.0.
- **p ≈ 0.354.** Well above the 0.05 threshold. There is no statistical evidence
  that fidelity predicts recall.
- **Remembered vs forgotten (0.84627 vs 0.84711).** The two groups' average
  fidelity differ by less than a thousandth. Practically identical — remembered
  words were **not** decoded better than forgotten words.

## Why the result is null

Two connected reasons:

1. **Raw cosine is inflated and not informative.** T5 embeddings share a strong
   common direction and have large norms, so nearly any prediction scores ~0.85.
   A high absolute cosine here does **not** mean the model identified the word.
2. **Word-specific decoding was at chance.** When we asked the stricter question
   — can the predicted vector pick the *correct* word out of all 576? — retrieval
   sat at chance (percentile ≈ 0.5), and a shuffled-label control performed the
   same as the real model. So there was no reliable word-identity signal for a
   memory effect to ride on. With no decoding signal, a null memory effect is
   exactly what you'd expect.

## Why null does NOT mean the project failed

- The **method is sound and fully audited**: embeddings reproduce to float32,
  recall labels match 576/576, EEG features are bit-exact vs the raw files, a
  133-check precision audit passed, and a full independent rerun reproduced the
  numbers.
- A **rigorous, reproducible negative result is a real scientific contribution**:
  it says *this specific lightweight approach* (raw 300–800 ms broadband EEG +
  ridge, small sample) does not surface a memory effect — and it tells us why
  (decoding didn't clear chance).
- It also **de-risks the next step**: we now know the bottleneck is the decoding
  stage, so effort should go there before re-testing memory.
- The result is **robust**: an independent redo that pooled ridge per subject
  (instead of per subject/session) gave the same conclusion (odds ratio 0.9695,
  p = 0.3236, vs current 0.9714, p = 0.3544).

## Why the Nature paper supports the motivation but does not contradict the null

The 2025 Nature Communications paper ("Towards decoding individual words from
non-invasive brain recordings") shows EEG/MEG → T5-embedding decoding *can* work
— which is exactly why the question is worth asking. But it does **not** conflict
with our null, because their decoding regime is very different from ours:

- They use a **deep CNN + 16-layer Transformer**, not linear ridge.
- They use a **~3-second** window (we used 300–800 ms), with band-pass filtering,
  baseline correction, resampling to 50 Hz, and robust scaling (we used raw
  500 Hz).
- Their **primary metric is retrieval accuracy** (top-k / rank), which is exactly
  the word-specific metric that sat at chance for us — not raw cosine.
- They train on **723 participants and ~5 million words**, with test-time
  averaging; we had 4 subjects and 4,608 words.
- They **do not study memory** at all — that outcome is our novel addition.

So the paper says "decoding is possible with a heavy pipeline and huge data,"
while our result says "a light pipeline on a small sample doesn't reach that
regime, so the memory question stays open." Both are true and consistent.

## The one-sentence conclusion (use verbatim)

> The project tested whether later-remembered words showed higher EEG-to-AI
> embedding fidelity than forgotten words. The final session-aware logistic
> mixed-effects model did not support this prediction: embedding fidelity was
> not significantly associated with later recall.
