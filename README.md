# AI Word-Embedding Fidelity and Memory

**Does the brain's response to a studied word predict that word's AI embedding
well enough to also predict whether the word is later remembered?**

A complete, audited, reproducible EEG pipeline — and an honest **negative
result**.

- **Question.** When a subject studies a word, is it more likely to be recalled
  later if the EEG pattern during encoding more accurately predicts the AI
  (T5-large) embedding of that word?
- **Answer.** **No.** Across 4, 16, and 32 subjects, embedding fidelity was
  **not** significantly associated with later recall. The null held at every scale.

Dataset: PEERS / OpenNeuro [ds004395](https://openneuro.org/datasets/ds004395)
v2.0.0, `task-ltpFR2` (its 576-word pool matches the T5 embedding list exactly).
EEG: 128–129 channel EGI, 500 Hz, EDF.

```
word appears  →  EEG during encoding (300–800 ms)  →  ridge predicts T5 embedding
              →  cosine(predicted, true) = "embedding fidelity"
              →  does fidelity predict later recall?
```

This mirrors the decoding logic of *[Towards decoding individual words from
non-invasive brain recordings](https://www.nature.com/articles/s41467-025-65499-0)*
(Nature Communications, 2025) and **adds a memory outcome**, which that paper
does not study.

---

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

Only the number of subjects changed; the pipeline was identical.

| | 4 subj | 16 subj | 32 subj |
| --- | --- | --- | --- |
| sessions | 8 | 32 | 64 |
| trials | 4,608 | 18,432 | 36,864 |
| odds ratio | 0.971 | 0.979 | 0.986 |
| 95% interval | [0.913, 1.033] | [0.947, 1.013] | [0.963, 1.009] |
| p-value | 0.354 | 0.220 | 0.218 |
| conclusion | no effect | no effect | no effect |

As subjects increase the odds ratio drifts **toward 1.0** while the interval
narrows around it (`results/figures/scaling_progression.png`). More data
sharpened the estimate onto *no effect* rather than uncovering one.

## The caveat that explains the null

**Word-specific decoding was at chance** — the decoder never identified the
correct word above guessing (32-subject run, vs a shuffled-label control):

| Check | Real | Shuffled | Chance |
| --- | --- | --- | --- |
| true-word percentile | 0.4975 | 0.5011 | 0.50 |
| top-5 retrieval | 0.0074 | 0.0088 | ~0.0087 |
| top-10 retrieval | 0.0164 | 0.0180 | ~0.0174 |

Raw cosine looks high (~0.845), but that is an artifact of a **common embedding
direction** shared by all T5 vectors (their norms average ~2,165) — not evidence
of decoding. So the bottleneck is the **decoding stage, not the sample size**:
raw broadband 300–800 ms EEG with linear ridge cannot decode word identity, so
there is no fidelity signal for a memory effect to build on. Adding subjects
cannot rescue a signal that isn't there.

A null is therefore the **expected, honest** outcome. The value of this project
is a trustworthy end-to-end method and a reproducible negative result — not a
positive finding.

---

## Method

| Stage | Choice |
| --- | --- |
| **Embeddings** | `google-t5/t5-large`, `T5EncoderModel` (encoder, **not** decoder), middle encoder layer `hidden_states[12]`, subword tokens averaged, EOS + pad excluded → one 1024-d vector per word → **576 × 1024** matrix |
| **Recall labels** | Free recall only: a studied `WORD` is `recalled=1` iff a `REC_WORD` with the same `item_num` occurs in the **same list**. Recognition events never used. |
| **EEG window** | 300–800 ms after onset, integer sample indexing: `start = sample + int(0.300×500) = +150`, `stop = sample + int(0.800×500) = +400` (**exclusive**) → 250 timepoints × 129 channels = **32,250** features. Raw 500 Hz — no filtering/baseline/resampling. |
| **Decoding** | Ridge `min ‖Y − XW‖² + α‖W‖²`, **α = 10,000**, SVD solver, per subject/session, 5-fold held-out-trial CV, `StandardScaler` fit on **train folds only**, one out-of-fold prediction per trial |
| **Fidelity** | `cos(ŷ, y) = (ŷ·y)/(‖ŷ‖‖y‖)` |
| **Memory model** | Logistic mixed-effects, session fixed effect, crossed `(1\|subject)` + `(1\|word)` random intercepts |

Full derivations: [`results/methods_and_math.md`](results/methods_and_math.md).

## Repository layout

| Path | Contents |
| --- | --- |
| [`code/`](code/) | The pipeline, one `stepNN_` script per outline stage, plus audits ([`code/README.md`](code/README.md)) |
| [`results/`](results/) | All findings: summaries, figures, tables, embeddings, validation ([`results/README.md`](results/README.md)) |
| `outputs/` | Working directory — regenerable, git-ignored |

Start with [`results/summary_4_vs_16_vs_32_subjects.txt`](results/summary_4_vs_16_vs_32_subjects.txt).

The 576 × 1024 embedding matrix and its word order ship in
[`results/embeddings/`](results/embeddings/), so the T5 stage does not need to be
re-run to inspect it.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
```

Raw EEG is **not** in the repo (ds004395 is ~8.7 TB upstream; each session is
~600 MB). The scripts stream one session at a time from OpenNeuro's public S3.

```bash
# 1. T5 embeddings (already shipped in results/embeddings/ — only to rebuild)
python code/step04_extract_t5_embeddings.py

# 2. Find valid ltpFR2 sessions, then run the chain across subjects/sessions
#    (downloads EEG, builds trials + X/Y, runs ridge CV -> fidelity)
python code/step10_scale_multi_session.py --n-subjects 32 --sessions-per-subject 2 \
    --combined outputs/all_sessions32_fidelity_results.csv

# 3. The memory model
python code/step11_run_memory_model.py --input outputs/all_sessions32_fidelity_results.csv

# 4. Render figures + tables from the committed results tables
python code/step13_build_results_package.py --tag 32
```

A session is only used if it is `ltpFR2`, has 576 studied words, 576/576 word
coverage against the embedding list, and a recording long enough that **every**
word's 300–800 ms window fits inside it. Many ltpFR2 EDFs are truncated and are
rejected by this gate.

## Validation

Every stage was independently verified — see
[`results/validation/`](results/validation/):

| Check | Result |
| --- | --- |
| T5 embeddings recomputed from the model | reproduce saved rows to float32 precision |
| Recall labels re-derived from raw events | **576/576 match** per session; no recognition used |
| EEG features re-extracted from the EDF | **bit-exact** |
| Precision audit | **133/133 checks passed** |
| Full independent rerun | reproduced the 4-subject numbers exactly (OR 0.9714, p 0.3544) |
| Independent redo (ridge per *subject*) | same conclusion (OR 0.9695, p 0.3236) |

## Limitations & next steps

- 32 subjects / 64 sessions is still modest, and **word-specific decoding never
  cleared chance** — the memory test was asking whether an undetectable signal
  predicts memory.
- Features are **raw broadband voltage**: no time-frequency band power, baseline
  correction, or spatial filtering.
- The reference paper reaches real decoding with a **deep CNN + Transformer**, a
  **~3 s** window, filtered/baselined/resampled EEG, and **723 participants /
  ~5M words** — and scores **retrieval accuracy**, not raw cosine. Our lighter
  linear setup does not reach that regime, which is consistent with (not
  contradicted by) their findings.
- **Next:** adopt that preprocessing and longer window, make top-k retrieval the
  primary decoding metric, use richer features or a neural decoder, add more
  data — and only re-test the memory question once decoding clears chance.

## Conclusion

> The project tested whether later-remembered words showed higher EEG-to-AI
> embedding fidelity than forgotten words. The final session-aware logistic
> mixed-effects model did not support this prediction: embedding fidelity was
> not significantly associated with later recall.

Data: PEERS / OpenNeuro ds004395. See `LICENSE` for terms.
