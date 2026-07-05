# Slide-by-Slide Outline (10 slides)

A talking outline, not a PowerPoint. One idea per slide; the sub-bullets are what
you say / show.

---

### Slide 1 — Title
- **AI Word-Embedding Fidelity and Memory**
- Does EEG-to-AI decoding of a studied word predict whether it's later recalled?
- Complete, audited pipeline · honest null result

### Slide 2 — Research question
- When a subject studies a word, is it more likely to be remembered if the EEG
  pattern more accurately predicts the AI embedding of that word?
- Builds on 2025 Nature Comms word-decoding paper; **adds a memory outcome**.
- Metric of interest: **EEG-to-AI embedding fidelity**.

### Slide 3 — Dataset
- PEERS / OpenNeuro **ds004395**, **ltpFR2** task; 129-ch EEG @ 500 Hz.
- **4 subjects · 8 sessions · 4,608 trials** (2,423 recalled / 2,185 forgotten).
- Sessions: LTP269 [12,20], LTP293 [5,22], LTP299 [2,6], LTP303 [10,22].
- Each session validated (576 words, full coverage, windows fit the recording).

### Slide 4 — Embedding pipeline
- 576 PEERS words → **T5-large**, encoder only, middle layer `hidden_states[12]`.
- Average subword tokens; exclude EOS + pad.
- Result: **576 × 1024** "answer key" matrix; word→row saved.
- Equation: `e_w = (1/k) Σ h_j`.

### Slide 5 — EEG extraction
- Window **300–800 ms** after word onset (preregistered).
- `start = sample + 150`, `stop = sample + 400` (exclusive) → **250 timepoints**.
- **129 channels × 250 = 32,250** features per trial; raw 500 Hz, no filtering.

### Slide 6 — Ridge decoding
- Per **subject/session**: ridge `min ‖Y − XW‖² + α‖W‖²`, **α = 10,000**.
- Held-out-trial 5-fold CV; scaler fit on **train only**; one OOF prediction/trial.
- Fidelity = **cosine(predicted, true)** embedding.

### Slide 7 — Audits (the strength)
- T5 reproduced to **float32**; recall labels **576/576**; EEG **bit-exact** vs EDF.
- **133/133** precision-audit checks; full **independent rerun** reproduced results.
- **Per-subject redo** → same conclusion. Everything reproducible.

### Slide 8 — Final model
- `recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`.
- Logistic mixed-effects; fidelity z-scored → **odds ratio per 1 SD**.

### Slide 9 — Results
- OR **0.971**, 95% CI **[0.913, 1.033]**, p **≈ 0.354**.
- Remembered **0.84627** vs forgotten **0.84711** (diff **−0.00084**).
- Raw cosine high (~0.847) but **inflated**; word-specific decoding **at chance**.

### Slide 10 — Interpretation & next steps
- **Conclusion:** "The project tested whether later-remembered words showed higher
  EEG-to-AI embedding fidelity than forgotten words. The final session-aware
  logistic mixed-effects model did not support this prediction: embedding fidelity
  was not significantly associated with later recall."
- Null is expected & honest for raw broadband EEG + ridge + small N.
- **Next:** paper-style preprocessing + ~3 s window; retrieval (top-k) metric;
  richer features / neural decoder; more subjects — then re-test memory.
