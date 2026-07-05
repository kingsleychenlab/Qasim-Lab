# Meeting Agenda — AI Word-Embedding Fidelity and Memory (30 min)

**Project:** AI Word-Embedding Fidelity and Memory
**One-line:** Does the brain's EEG response to a studied word predict that word's
AI (T5) embedding well enough to also predict whether the word is later recalled?

| Time | Segment | What to cover |
| --- | --- | --- |
| **0–3 min** | **Motivation** | The question, why it's interesting, the link to the Nature 2025 word-decoding paper, and the novel add-on: a memory outcome. |
| **3–8 min** | **Dataset & embeddings** | PEERS / ds004395, ltpFR2 task. 4 subjects, 8 sessions, 4608 trials. T5-large embeddings: 576 words → 576×1024 matrix (encoder, middle layer, subword-averaged, EOS/pad excluded). |
| **8–15 min** | **EEG pipeline & ridge decoding** | 300–800 ms window (129 ch × 250 tp = 32,250 features). Per subject/session ridge (α=10,000) with held-out-trial CV, scaler on train only. Fidelity = cosine(predicted, true). |
| **15–22 min** | **Validation / audits** | T5 reproduced to float32; recall labels 576/576; EEG bit-exact vs EDF; 133/133 precision-audit checks; full independent rerun; per-subject pooled redo. |
| **22–27 min** | **Final model & results** | `recalled ~ embedding_fidelity + session + (1\|subject) + (1\|word)`. OR 0.971, 95% CI [0.913, 1.033], p ≈ 0.354. Remembered 0.84627 vs forgotten 0.84711. |
| **27–30 min** | **Limitations & next steps** | Small N, raw broadband EEG, ridge, chance-level word-specific decoding. Next: paper-style preprocessing, retrieval metric, richer features, more data. |

**Final conclusion to land the meeting on (say it verbatim):**
> The project tested whether later-remembered words showed higher EEG-to-AI
> embedding fidelity than forgotten words. The final session-aware logistic
> mixed-effects model did not support this prediction: embedding fidelity was
> not significantly associated with later recall.

**Tone:** confident about the *method and its rigor*; honest and calm about the
*null result*. The deliverable is a complete, audited, reproducible pipeline and
an honest negative finding — not a positive claim.
