# Results

**Question.** When someone studies a word, is that word more likely to be
remembered if their EEG during encoding does a better job of predicting the AI
(T5-large) embedding of that word?

**Answer.** No. Across 4, 16, and 32 subjects, fidelity was not significantly
tied to later recall. The null held every time.

Dataset: PEERS / OpenNeuro [ds004395](https://openneuro.org/datasets/ds004395)
v2.0.0, `task-ltpFR2` (its 576-word pool matches the T5 embedding list exactly).
The EEG is 128-129 channel EGI at 500 Hz. Code is in [`../code/`](../code/).

```
word appears  ->  EEG during encoding  ->  predicted T5 embedding
              ->  compare to the real embedding  ->  does the similarity predict recall?
```

## The headline number: 32 subjects, 64 sessions, 36,864 trials

Model: `recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`,
logistic mixed-effects, fidelity z-scored so the odds ratio is per 1 SD.

| Quantity | Value |
| --- | --- |
| odds ratio (per 1 SD) | 0.986 |
| 95% interval | [0.963, 1.009] — includes 1.0 |
| p-value | 0.218 |
| remembered mean fidelity | 0.84501 |
| forgotten mean fidelity | 0.84553 |
| difference (rem − forg) | −0.00052 |

## The null doesn't budge with scale

The only thing that changed between runs was the number of subjects. The pipeline
was identical.

| | 4 subj | 16 subj | 32 subj |
| --- | --- | --- | --- |
| sessions | 8 | 32 | 64 |
| trials | 4,608 | 18,432 | 36,864 |
| odds ratio | 0.971 | 0.979 | 0.986 |
| 95% interval | [0.913, 1.033] | [0.947, 1.013] | [0.963, 1.009] |
| p-value | 0.354 | 0.220 | 0.218 |
| conclusion | no effect | no effect | no effect |

As subjects go up, the odds ratio drifts toward 1.0 and the interval tightens
while staying centered on 1.0. More data sharpened the estimate onto "no effect"
rather than uncovering one.

### Run by run

**4 subjects** (LTP269, LTP293, LTP299, LTP303), 8 sessions, 4,608 trials, 2,423
recalled / 2,185 forgotten.

- odds ratio 0.971, interval [0.913, 1.033], p 0.354
- coefficient per SD −0.0291
- remembered 0.84627, forgotten 0.84711, difference −0.00084

A small sample, so nothing confirmatory, but honest. For the original 4-subject
run I also fit the supplementary metrics as a cross-check, and they all agreed:

| Metric | Odds ratio | 95% interval | p | Verdict |
| --- | --- | --- | --- | --- |
| raw_cosine (the main one) | 0.971 | [0.913, 1.033] | 0.354 | no effect |
| centered_cosine | 0.995 | [0.936, 1.058] | 0.863 | no effect |
| true_word_percentile | 0.976 | [0.918, 1.038] | 0.435 | no effect |
| centered_true_word_percentile | 1.010 | [0.950, 1.074] | 0.742 | no effect |

**16 subjects** (the original 4 plus 12 more), 32 sessions, 18,432 trials, 9,980
recalled / 8,452 forgotten.

- odds ratio 0.979, interval [0.947, 1.013], p 0.220
- coefficient per SD −0.0211
- remembered 0.84337, forgotten 0.84367, difference −0.00030

Four times the data tightened the interval, but the odds ratio stayed around 0.98
and the interval still contained 1.0.

**32 subjects** (the 16 plus 16 more), 64 sessions, 36,864 trials, 17,040 recalled
/ 19,824 forgotten.

- odds ratio 0.986, interval [0.963, 1.009], p 0.218
- coefficient per SD −0.0145
- remembered 0.84501, forgotten 0.84553, difference −0.00052

The estimate is now homing in on exactly "no effect."

## The caveat that explains all of it

Word-specific decoding was at chance in all three runs.

| Check | Real | Shuffled-label control | Chance |
| --- | --- | --- | --- |
| true-word percentile | 0.4975 | 0.5011 | 0.50 |
| top-5 retrieval | 0.0074 | 0.0088 | ~0.0087 |
| top-10 retrieval | 0.0164 | 0.0180 | ~0.0174 |

(The numbers above are the 32-subject run; the 16-subject run was the same story:
percentile 0.4977 real vs 0.5011 shuffled, top-5 0.0071 vs 0.0088, top-10 0.0162
vs 0.0184.)

The real model matches the shuffled-label control, and the high raw cosine (~0.845)
comes from a common direction that all T5 vectors share, not from real word
decoding.

So the bottleneck is the decoding stage, not the sample size. Raw broadband
300-800 ms EEG with a linear ridge can't decode word identity above chance, so
there's no reliable fidelity signal for a memory effect to build on. Adding
subjects can't rescue a signal that isn't there. Any next step has to strengthen
the decoder first — the reference paper's EEG preprocessing and longer (~3 s)
window, richer time-frequency features, or a neural decoder — before the memory
question is worth re-testing.

A null is the honest outcome here. The value of the project is a trustworthy,
reproducible end-to-end method, not a positive finding.

## What's in this folder

| Path | Contents |
| --- | --- |
| `summary_4_vs_16_vs_32_subjects.txt` | Start here — the plain-text writeup of all three runs |
| `methods_and_math.md` | The methods and the math behind each stage |
| `embeddings/` | The T5 embedding deliverables (below) |
| `figures/` | 6 figures, one per question a reader tends to ask (below) |
| `tables/` | Model outputs and the trial-level fidelity tables |
| `summaries/` | The raw per-run model summaries and metadata (4 / 16 / 32 subjects) |
| `validation/` | The audits and the independent reproducibility reruns |

### `embeddings/`

| File | Contents |
| --- | --- |
| `peers_t5large_embeddings.npy` | The 576 × 1024 matrix, one row per PEERS word |
| `peers_word_order.csv` | `word`, `row_index` — the row order of the matrix |
| `peers_t5large_embeddings.csv` | The same matrix, human-readable |
| `peers_words.csv` | The 576-word PEERS pool |
| `embedding_metadata.json` | Model, layer, and token-handling provenance |

Built with `google-t5/t5-large`, `T5EncoderModel` (the encoder, not the decoder),
middle encoder layer `hidden_states[12]`, subword tokens averaged, EOS and pad
left out.

### `figures/`

| Figure | The question it answers |
| --- | --- |
| `pipeline_flow.png` | What's the method? |
| `remembered_vs_forgotten_fidelity.png` | Do remembered and forgotten words differ? (no, near-identical) |
| `decoding_chance_check.png` | Is the decoder doing anything? (no, retrieval is at chance) |
| `raw_vs_centered_metrics.png` | Why is raw cosine high? (the common-direction artifact) |
| `final_model_odds_ratios.png` | Does the model back that up? (every interval spans OR = 1) |
| `scaling_progression.png` | Does more data change it? (OR heads to 1.0 as the intervals tighten) |

The figures and the `*_32subjects` tables are rendered from the 32-subject run by
`code/step13_build_results_package.py`, which reads only the committed tables in
`tables/`, so it re-runs cleanly on a fresh clone.

### `validation/`

The result was re-derived independently. The full writeup is in
[`validation/validation.md`](validation/validation.md): a precision audit
(133/133 checks), stage-by-stage recomputes of the embeddings, recall labels, and
EEG windows, a complete rerun that reproduced the 4-subject numbers exactly (odds
ratio 0.9714, p 0.3544), and a second implementation that fits ridge per subject
instead of per session and lands on the same conclusion. The raw logs it draws
from are the `.txt` and `.csv` files in the same folder.
