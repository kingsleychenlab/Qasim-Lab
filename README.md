# Word embeddings, EEG, and memory

The question here is simple to state: if your brain's response to a word looks
more like an AI model's version of that word, do you remember the word better?

The answer came back no. I ran the check three times, on 4 subjects, then 16,
then 32, and the effect never showed up.

The data is PEERS / OpenNeuro [ds004395](https://openneuro.org/datasets/ds004395)
v2.0.0, the `ltpFR2` task. People read lists of words while wearing a 129-channel
EEG cap, then tried to recall them a moment later. I picked `ltpFR2` because its
576-word pool is the exact list I built embeddings for.

## The idea

Reading a word sets off a distinctive bit of electrical activity in your brain. A
language model like T5 also turns a word into a list of numbers, an "embedding,"
that stands in for what the word means.

A [2025 Nature Communications paper](https://www.nature.com/articles/s41467-025-65499-0)
showed you can partly recover which word someone is reading by mapping their EEG
onto that word's embedding. I took that setup and added a question the paper
doesn't ask: when the mapping works better for a given word, does that word get
remembered more often?

The chain looks like this:

```
word appears  ->  take the EEG from 300-800 ms after it shows up
              ->  ridge regression predicts the word's T5 embedding
              ->  cosine similarity between the predicted and the real embedding
              ->  call that number "embedding fidelity"
              ->  ask whether higher fidelity goes with better recall
```

## What came out

32 subjects, 64 sessions, 36,864 words studied.

The model was `recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`.
Fidelity is z-scored, so the odds ratio is the change in recall odds for a
one-standard-deviation change in fidelity.

| Result | Value | What it means |
| --- | --- | --- |
| odds ratio | 0.986 | 1.0 would be no effect. For practical purposes this is 1.0. |
| 95% interval | [0.963, 1.009] | Contains 1.0, so "no effect" is not ruled out. |
| p-value | 0.218 | Well above 0.05. Not significant. |
| remembered fidelity | 0.84501 | Average for words later recalled. |
| forgotten fidelity | 0.84553 | Average for words later forgotten. Basically the same. |
| difference | -0.00052 | About zero, and if anything the wrong way. |

Adding subjects didn't change anything:

| | 4 subjects | 16 subjects | 32 subjects |
| --- | --- | --- | --- |
| sessions | 8 | 32 | 64 |
| words studied | 4,608 | 18,432 | 36,864 |
| odds ratio | 0.971 | 0.979 | 0.986 |
| 95% interval | [0.913, 1.033] | [0.947, 1.013] | [0.963, 1.009] |
| p-value | 0.354 | 0.220 | 0.218 |

Going from 4 subjects to 32 is eight times the data, and all it did was pull the
odds ratio a little closer to 1.0 and shrink the interval around it. The extra
data didn't turn up a hidden effect. It made "no effect" a sharper answer. See
`results/figures/scaling_progression.png`.

## Why it's a no

The fidelity score sits around 0.845, which looks high until you see where it
comes from.

Every T5 word vector points in roughly the same direction and has a big norm
(about 2,165 on average). Because of that, almost any prediction lands near a
cosine of 0.85 whether or not it actually identified the word. A high cosine here
doesn't mean the decoding worked.

The real test is whether a prediction can pick its own word out of the full set
of 576. It can't:

| Check | Real model | Shuffled labels | Chance |
| --- | --- | --- | --- |
| true-word percentile | 0.4975 | 0.5011 | 0.50 |
| top-5 retrieval | 0.0074 | 0.0088 | 0.0087 |
| top-10 retrieval | 0.0164 | 0.0180 | 0.0174 |

The real model does no better than a control trained on shuffled labels. Word
decoding is at chance.

That is the whole story behind the null. There was no working decoder, so there
was no fidelity signal for memory to track. You can't recover a signal by adding
subjects when the signal was never there in the first place. The limit is the
decoding method, not the number of people I ran.

So a null is what I'd expect here. What the project actually gives you is a
pipeline that works and checks out end to end, not a discovery.

## How it works

| Stage | What I did |
| --- | --- |
| Embeddings | `google-t5/t5-large`, encoder only (`T5EncoderModel`, no decoder). Take the middle encoder layer, `hidden_states[12]`. When a word splits into pieces, average them. Drop the end-of-sequence and padding tokens. That leaves one 1024-number vector per word, a 576 x 1024 matrix. |
| Recall labels | Free recall only. A studied word counts as recalled if the same item number comes back during the recall phase of the same list. Recognition data is never used. |
| EEG window | 300-800 ms after the word appears. At 500 Hz that runs from `sample+150` to `sample+400`, stop exclusive, so 250 timepoints. Across 129 channels that's 32,250 numbers per word. Raw signal, no filtering or baseline correction. |
| Decoding | Ridge regression, `alpha = 10000`, fit separately for each subject and session. 5-fold cross-validation, always tested on words the model didn't train on. The scaler is fit on the training folds only. |
| Fidelity | `cos(predicted, true)`, the angle between the predicted and the real embedding. |
| Memory model | Logistic mixed-effects, because recall is yes/no. Session is a fixed effect. Subject and word get random intercepts, since some people and some words are just easier. |

The math is written out in [`results/methods_and_math.md`](results/methods_and_math.md).

## What's in here

| Path | Contents |
| --- | --- |
| [`code/`](code/) | The pipeline. One `stepNN_` script per stage, plus the audit scripts. See [`code/README.md`](code/README.md). |
| [`results/`](results/) | Everything I found: the writeup, figures, tables, embeddings, and checks. See [`results/README.md`](results/README.md). |
| `outputs/` | Scratch space the scripts write to. Regenerable, so it's git-ignored. |

If you only read one thing, read
[`results/summary_4_vs_16_vs_32_subjects.txt`](results/summary_4_vs_16_vs_32_subjects.txt).

The finished 576 x 1024 embedding matrix is committed in
[`results/embeddings/`](results/embeddings/), so you can look at it without
rebuilding it.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
```

The raw EEG isn't in this repo. ds004395 is about 8.7 TB, and a single session is
roughly 600 MB, so the scripts stream one session at a time from OpenNeuro's
public S3 bucket.

```bash
# Rebuild the T5 embeddings. Optional, they're already in results/embeddings/.
python code/step04_extract_t5_embeddings.py

# Download sessions, build the trials and X/Y matrices, run the ridge CV.
python code/step10_scale_multi_session.py --n-subjects 32 --sessions-per-subject 2 \
    --combined outputs/all_sessions32_fidelity_results.csv

# Fit the memory model.
python code/step11_run_memory_model.py --input outputs/all_sessions32_fidelity_results.csv

# Redraw the figures and tables from the committed results.
python code/step13_build_results_package.py --tag 32
```

A session is only used if it's `ltpFR2`, has all 576 words, matches the word list
completely, and the recording is long enough that every word's 300-800 ms window
fits inside it. A lot of `ltpFR2` recordings are cut short and get dropped at this
gate.

## Checks I ran

Each stage was verified on its own. The details are in
[`results/validation/`](results/validation/).

| Check | Result |
| --- | --- |
| Recomputed the T5 embeddings from the model | Match the saved ones to float32 precision |
| Rebuilt the recall labels from the raw event files | 576/576 match in every session |
| Re-extracted the EEG windows from the source EDF | Identical, bit for bit |
| Full precision audit | 133 of 133 checks passed |
| Independent rerun of the whole pipeline | Reproduced the 4-subject numbers exactly (odds ratio 0.9714, p 0.3544) |
| Second version that fits ridge per subject instead of per session | Same conclusion (odds ratio 0.9695, p 0.3236) |

## Limits, and what I'd try next

The decoder never beat chance, which means the memory test was asking whether a
signal I couldn't detect predicts recall. That's the main limit. The sample (32
subjects) is still on the small side, and the features are raw broadband voltage
with no time-frequency analysis, baseline correction, or spatial filtering.

The Nature paper does get real decoding, but with much heavier machinery: a deep
CNN plus a Transformer, a 3-second window, filtered and resampled EEG, and 723
participants reading around 5 million words. They also score retrieval accuracy
instead of raw cosine. My lighter linear setup never gets near that regime, so
their result and mine don't actually conflict.

If I picked this back up, roughly in order:

1. Use their preprocessing and the longer window.
2. Score decoding by top-k retrieval, not raw cosine.
3. Move to richer features, or a small neural decoder.
4. Add more subjects.

The memory question is only worth re-asking once decoding clears chance.

## Conclusion

The project tested whether words that were later remembered showed higher
EEG-to-AI embedding fidelity than words that were forgotten. The final
session-aware logistic mixed-effects model didn't back that up: fidelity was not
significantly tied to later recall.

Data from PEERS / OpenNeuro ds004395. Terms in `LICENSE`.
