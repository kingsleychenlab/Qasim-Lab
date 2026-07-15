# AI Word-Embedding Fidelity and Memory

Do you remember a word better if your brain's response to it looks more like the
AI's representation of that word?

Short answer: no. We tested it on 4, 16, and then 32 subjects. The effect was
absent every time.

Dataset: PEERS / OpenNeuro [ds004395](https://openneuro.org/datasets/ds004395)
v2.0.0, the `ltpFR2` task. Subjects read word lists while wearing a 129-channel
EEG cap, then tried to recall them. We use `ltpFR2` because its 576-word pool is
exactly the word list we built embeddings for.

## The idea

When you read a word, your brain makes a distinctive electrical response. A
language model like T5 also turns that word into a list of numbers (an
"embedding") that captures its meaning.

A [2025 Nature Communications paper](https://www.nature.com/articles/s41467-025-65499-0)
showed you can partly work out which word someone is reading by mapping their
EEG onto that word's embedding. We reused that idea and added a memory question
the paper does not ask: if the mapping works *better* on a given word, is that
word remembered more often?

The pipeline:

```
word appears  ->  EEG from 300-800 ms after it appears
              ->  ridge regression predicts the word's T5 embedding
              ->  cosine similarity between predicted and true embedding
              ->  we call that "embedding fidelity"
              ->  does higher fidelity mean better recall?
```

## What we found

32 subjects, 64 sessions, 36,864 words studied.

Model: `recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`.
Fidelity is z-scored, so the odds ratio is the change in recall odds per 1
standard deviation of fidelity.

| Result | Value | What it means |
| --- | --- | --- |
| odds ratio | 0.986 | 1.0 would mean no effect. This is 1.0 for practical purposes. |
| 95% interval | [0.963, 1.009] | Contains 1.0, so we cannot rule out "no effect". |
| p-value | 0.218 | Well above 0.05. Not significant. |
| remembered fidelity | 0.84501 | Average for words later recalled. |
| forgotten fidelity | 0.84553 | Average for words later forgotten. Same number. |
| difference | -0.00052 | Essentially zero, and slightly the wrong way. |

Adding subjects did not change it:

| | 4 subjects | 16 subjects | 32 subjects |
| --- | --- | --- | --- |
| sessions | 8 | 32 | 64 |
| words studied | 4,608 | 18,432 | 36,864 |
| odds ratio | 0.971 | 0.979 | 0.986 |
| 95% interval | [0.913, 1.033] | [0.947, 1.013] | [0.963, 1.009] |
| p-value | 0.354 | 0.220 | 0.218 |

With eight times the data the odds ratio moved *toward* 1.0 and the interval
tightened around it. More data made the "no effect" answer sharper instead of
turning up a hidden effect. See `results/figures/scaling_progression.png`.

## Why the answer is no

The fidelity score looks high, around 0.845. That is misleading.

All T5 word vectors point in a broadly similar direction and have large norms
(about 2,165 on average). So almost any prediction scores about 0.85, whether or
not it identified the word. A high cosine here is not evidence of decoding.

The real test is whether the prediction can pick the correct word out of all 576.
It cannot:

| Check | Real model | Shuffled labels | Chance |
| --- | --- | --- | --- |
| true-word percentile | 0.4975 | 0.5011 | 0.50 |
| top-5 retrieval | 0.0074 | 0.0088 | 0.0087 |
| top-10 retrieval | 0.0164 | 0.0180 | 0.0174 |

The real model matches a control trained on shuffled labels. Word decoding sits
at chance.

That explains the null. There was no working decoder, so there was no fidelity
signal for memory to track. Adding subjects cannot recover a signal that was
never there. The limit is the decoding method, not the sample size.

So a null is what you would expect here. What the project delivers is a working,
checked, reproducible method and a result you can trust, not a discovery.

## How it works

| Stage | What we did |
| --- | --- |
| Embeddings | `google-t5/t5-large`, encoder only (`T5EncoderModel`, no decoder). Take the middle encoder layer, `hidden_states[12]`. If a word splits into pieces, average them. Drop the end-of-sequence and padding tokens. Result: one 1024-number vector per word, a 576 x 1024 matrix. |
| Recall labels | Free recall only. A studied word counts as recalled if the same item number comes back during the recall phase of the same list. Recognition data is never used. |
| EEG window | 300-800 ms after the word appears. At 500 Hz that is `sample+150` to `sample+400`, stop exclusive, so 250 timepoints. Across 129 channels that is 32,250 numbers per word. Raw signal, no filtering or baseline correction. |
| Decoding | Ridge regression, `alpha = 10000`, fit separately for each subject and session. 5-fold cross-validation, always tested on words the model did not train on. The scaler is fit on training folds only. |
| Fidelity | `cos(predicted, true)`, the angle between the predicted and true embedding. |
| Memory model | Logistic mixed-effects, since recall is yes/no. Session is a fixed effect. Subject and word get random intercepts, because some people and some words are simply easier. |

The math is written out in [`results/methods_and_math.md`](results/methods_and_math.md).

## What's in here

| Path | Contents |
| --- | --- |
| [`code/`](code/) | The pipeline. One `stepNN_` script per stage, plus the audit scripts. See [`code/README.md`](code/README.md). |
| [`results/`](results/) | Everything we found: summaries, figures, tables, embeddings, checks. See [`results/README.md`](results/README.md). |
| `outputs/` | Scratch space the scripts write to. Regenerable, so it is git-ignored. |

Best starting point:
[`results/summary_4_vs_16_vs_32_subjects.txt`](results/summary_4_vs_16_vs_32_subjects.txt).

The finished 576 x 1024 embedding matrix ships in
[`results/embeddings/`](results/embeddings/), so you can look at it without
rebuilding it.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
```

The raw EEG is not in this repo. ds004395 is about 8.7 TB and one session alone
is roughly 600 MB, so the scripts stream one session at a time from OpenNeuro's
public S3 bucket.

```bash
# Rebuild the T5 embeddings. Optional, they are already in results/embeddings/.
python code/step04_extract_t5_embeddings.py

# Download sessions, build the trials and X/Y matrices, run the ridge CV.
python code/step10_scale_multi_session.py --n-subjects 32 --sessions-per-subject 2 \
    --combined outputs/all_sessions32_fidelity_results.csv

# Fit the memory model.
python code/step11_run_memory_model.py --input outputs/all_sessions32_fidelity_results.csv

# Redraw the figures and tables from the committed results.
python code/step13_build_results_package.py --tag 32
```

A session is only used if it is `ltpFR2`, has all 576 words, matches our word
list completely, and the recording is long enough that every word's 300-800 ms
window fits inside it. Plenty of `ltpFR2` recordings are cut short and get
rejected here.

## Checks we ran

Each stage was verified on its own. Details in
[`results/validation/`](results/validation/).

| Check | Result |
| --- | --- |
| Recomputed the T5 embeddings from the model | Match the saved ones to float32 precision |
| Rebuilt the recall labels from the raw event files | 576/576 match in every session |
| Re-extracted the EEG windows from the source EDF | Identical, bit for bit |
| Full precision audit | 133 of 133 checks passed |
| Independent rerun of the whole pipeline | Reproduced the 4-subject numbers exactly (odds ratio 0.9714, p 0.3544) |
| Second version that fits ridge per subject instead of per session | Same conclusion (odds ratio 0.9695, p 0.3236) |

## Limits, and what to try next

The decoder never beat chance, so the memory test was asking whether a signal we
could not detect predicts recall. That is the main limit. The sample (32
subjects) is still modest, and the features are raw broadband voltage with no
time-frequency analysis, baseline correction, or spatial filtering.

The Nature paper gets real decoding, but with much heavier machinery: a deep
CNN plus Transformer, a 3-second window, filtered and resampled EEG, and 723
participants reading about 5 million words. They also score retrieval accuracy
rather than raw cosine. Our lighter linear setup never reaches that regime, so
their result and ours do not conflict.

Worth trying, in order:

1. Use their preprocessing and the longer window.
2. Score decoding by top-k retrieval, not raw cosine.
3. Move to richer features, or a small neural decoder.
4. Add more subjects.

Re-ask the memory question only once decoding clears chance.

## Conclusion

> The project tested whether later-remembered words showed higher EEG-to-AI
> embedding fidelity than forgotten words. The final session-aware logistic
> mixed-effects model did not support this prediction: embedding fidelity was
> not significantly associated with later recall.

Data from PEERS / OpenNeuro ds004395. Terms in `LICENSE`.
