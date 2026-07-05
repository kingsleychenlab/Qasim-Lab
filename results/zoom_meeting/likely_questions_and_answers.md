# Likely Questions & Strong Answers

Prepared responses. Keep them honest — the strength here is rigor, not a positive
result.

---

**Q: Why T5-large?**
A: To mimic the reference approach as closely as possible. The 2025 Nature
Communications word-decoding paper used T5-large embeddings, so using the same
model — encoder only, middle layer, subword-averaged — keeps our "answer key"
comparable to a method that is known to work in a stronger setup. It's also a
strong, general-purpose semantic representation with a convenient 1024-dim output.

**Q: Why the 300–800 ms window?**
A: It's a **preregistered encoding window** tied to the word-processing response
in EEG — the N400/late-positivity range where semantic processing shows up. Fixing
it in advance avoids cherry-picking a window that happens to look good. (Note: the
Nature paper used a much longer ~3 s window; adopting that is a natural next step.)

**Q: Why ridge regression?**
A: It's the right tool for this shape of problem — ~32,000 EEG features but only a
few hundred training trials per fold. Ridge's strong L2 regularization (α =
10,000) keeps a wide, underdetermined linear map stable, it's fast, and it's
transparent. It was also the method specified in the project plan. It is a
deliberately simple first pass, not the deep network the paper used.

**Q: Why cosine similarity?**
A: Embeddings encode meaning as a **direction** in space more than a magnitude, so
cosine — the angle between predicted and true vectors — is the natural similarity.
It's also what the reference paper uses to match predictions to words.

**Q: Why was raw cosine so high (~0.85)?**
A: That's an artifact, and we flagged it. T5 embeddings share a strong **common
direction** and have large norms, so almost any reasonable prediction lands near
0.85 regardless of whether it identified the specific word. High raw cosine here
is *not* evidence of good decoding — which is exactly why we added word-specific
control metrics.

**Q: Why was decoding at chance?**
A: When we removed the common direction and asked the strict question — can the
prediction pick the *correct* word out of 576? — retrieval percentile was ~0.5
(chance), and a **shuffled-label control** matched the real model. Raw broadband
500 Hz EEG over a 500 ms window, with a linear model and a small sample, simply
doesn't carry reliable word-identity information at the single-trial level.

**Q: Does this contradict the Nature paper?**
A: No. Their pipeline is far heavier: a deep CNN + Transformer, a ~3 s window,
band-pass filtering, baseline correction, resampling, and **723 participants /
~5M words** with test-time averaging — and their headline metric is retrieval
accuracy, the very metric that was at chance for us. They show decoding is
*possible* with that machinery; we show a *light* approach on a *small* sample
doesn't reach it. Both are consistent. They also don't study memory — that's our
addition.

**Q: Why is a null result still useful?**
A: Three reasons. First, it's a **complete, audited, reproducible pipeline** — a
reusable asset. Second, a rigorous negative tells the field that *this specific
lightweight method* doesn't surface a memory effect, and **why** (decoding didn't
clear chance). Third, it **de-risks and directs** the next phase: the bottleneck
is the decoding stage, so that's where to invest before re-testing memory.

**Q: What would you do next?**
A: In order: (1) adopt the paper's EEG preprocessing and a longer (~3 s) window;
(2) make **balanced top-k retrieval accuracy** the primary decoding metric — we
already compute it; (3) move from ridge to richer features (time-frequency band
power, spatial filtering) or a small neural decoder; (4) add many more
subjects/sessions. Only once decoding clears chance does re-testing the memory
model become meaningful.

**Q: What's the biggest limitation?**
A: The decoding stage never cleared chance, so the memory test was essentially
asking "does a signal we couldn't detect predict memory?" Combined with a small
sample (4 subjects, 8 sessions) and raw broadband features, that's the core
limitation. The memory model itself is sound; the input signal was too weak.

**Q: How do you know the recall labels and EEG extraction are correct?**
A: We audited both independently. Recall labels were **re-derived from the raw
event files** and matched 576/576 in every session, using only free-recall events
(recognition never used) and matching within-list only. The EEG features were
**re-extracted straight from the EDF** and are **bit-for-bit identical** to what
the model used. Plus a 133-check precision audit and a full independent rerun both
passed.

**Q: Could the null be a power problem — too few subjects?**
A: Partly, yes — 4 subjects is small. But the more fundamental issue is that
word-specific decoding was at chance, so there was no signal to be underpowered
*about*. More subjects help only after the decoding stage produces real
above-chance fidelity. That's why "more data" is step 4, not step 1, in the plan.

**Q: Is raw cosine really the right primary metric?**
A: It's the metric named in the project plan, so we report it as primary — but we
were transparent that it's inflated by the common direction, and we ran the
word-specific metrics as controls. If anything, the word-specific/retrieval
metrics (the paper's choice) are the more meaningful read, and they were at
chance. Either way the memory conclusion is the same: null.
