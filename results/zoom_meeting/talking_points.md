# Talking Points — Speaking Script

Plain language, scientifically accurate. Read or paraphrase. Bracketed cues are
for you, not to say aloud.

---

## Opening (motivation)

"**The core idea was** this: when you read a word, your brain produces a burst
of electrical activity we can record with EEG. Modern AI language models like
T5 also represent each word as a list of numbers — an 'embedding' that captures
its meaning. A 2025 Nature Communications paper showed you can partly *decode*
which word someone is reading by mapping their brain signal onto that word's AI
embedding.

Our project borrowed that core logic but **added a memory twist**: not just
'can we decode the word,' but 'does decoding it *well* predict whether the person
later *remembers* it?' The intuition is that a cleaner, more semantically
complete encoding might lead to better memory."

## The question, precisely

"So the research question is: **when a subject studies a word, is it more likely
to be remembered if their EEG pattern more accurately predicts the AI embedding
of that word?** We call that accuracy the *EEG-to-AI embedding fidelity*."

## Data

"We used the **PEERS dataset** from OpenNeuro — people studied lists of words
while wearing a 129-channel EEG cap, then tried to recall them. For this analysis
we used **4 subjects, 8 sessions, 4,608 word trials** — about 2,400 later
recalled and 2,200 forgotten."

## Embeddings

"**The AI side worked like this.** We took the **576 PEERS words** and ran each
through **T5-large** — specifically the *encoder*, and we pulled the
representation from the **middle layer**. If a word split into pieces, we
averaged the pieces, and we dropped the end-of-sequence and padding tokens. That
gives one **1024-number vector per word** — a 576 × 1024 matrix that is the
'answer key' the brain model tries to predict."

## EEG + ridge

"**The EEG part worked like this.** For every word a subject studied, we took a
fixed slice of brain activity — **300 to 800 milliseconds after the word
appeared** — across all 129 channels. Flattened, that's about **32,000 numbers
per trial**. Then, separately for each subject and session, we trained a
**ridge regression** — a standard, well-regularized linear model — to map those
32,000 EEG numbers onto the 1,024-number T5 embedding.

We were careful to test only on **held-out trials** the model never trained on,
and we standardized using only the training data, so there's no peeking."

## Fidelity

"For each held-out word, the model predicts an embedding, and we compare it to
the true embedding using **cosine similarity** — how aligned the two vectors are.
That single number is the **embedding fidelity** for that trial."

## Memory model

"Then the memory test. We fit a **logistic mixed-effects model**:
`recalled ~ embedding_fidelity + session + (1|subject) + (1|word)`. In plain
terms: does higher fidelity make recall more likely, after accounting for
session, and for the fact that some people and some words are just easier?"

## Validation (say this — it's the strength of the project)

"**The important validation step was** that we didn't just trust the pipeline —
we audited every stage. We recomputed the T5 embeddings from the model and they
matched to floating-point precision. We re-derived the recall labels from the raw
event files and got a perfect 576-out-of-576 match per session. We re-extracted
the EEG straight from the raw files and it was **bit-for-bit identical**. A full
independent rerun reproduced the final number, and a **133-point precision audit
passed completely**."

## Result

"**The final result was** a null. Higher fidelity did **not** predict recall. The
odds ratio was **0.971** — essentially 1.0, meaning no effect — with a confidence
interval from 0.913 to 1.033 that straddles 1, and a p-value around **0.35**. The
average fidelity for remembered words, **0.846**, was basically identical to
forgotten words, **0.847**."

## The nuance (be honest here)

"Now, one subtlety worth being upfront about. The raw cosine similarity looked
*high* — about 0.85 — which might sound like great decoding. But it isn't. T5
embeddings all share a strong common direction, so almost any prediction lands
near 0.85. When we used stricter, **word-specific** metrics — can we actually
pick the *right* word out of 576? — performance was **at chance**, and a
shuffled-label control matched the real one. So the brain signal, with this
lightweight method, wasn't carrying reliable word-identity information."

## Interpretation

"**The honest interpretation is** that with raw broadband EEG and a simple linear
model on a small sample, we couldn't reach the level of decoding where the memory
question even becomes answerable. The null is the *expected and honest* outcome —
not a failed project. What we have is a complete, rigorously audited, fully
reproducible pipeline and a trustworthy negative result."

## Robustness (optional add)

"We even re-ran it a second way — pooling each subject's sessions instead of
keeping them separate — and got the **same conclusion** (odds ratio 0.97,
p ≈ 0.32). So the null is robust to that design choice."

## Close

"So to close: **The project tested whether later-remembered words showed higher
EEG-to-AI embedding fidelity than forgotten words. The final session-aware
logistic mixed-effects model did not support this prediction: embedding fidelity
was not significantly associated with later recall.** Next steps are about
strengthening the decoding side — the paper's preprocessing, a longer window,
richer features, and more data — before re-asking the memory question."
