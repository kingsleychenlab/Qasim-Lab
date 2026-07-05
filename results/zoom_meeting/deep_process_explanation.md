# Deep Process Explanation

End-to-end, with the math. Notation: a word is `w`; a trial (one word
presentation) is indexed `i`.

## 1. The dataset (PEERS / ds004395, ltpFR2)

PEERS (Penn Electrophysiology of Encoding and Retrieval Study) recorded EEG while
people studied word lists and then freely recalled them. We used the **ltpFR2**
task because its **576-word pool** matches our embedding list exactly. EEG is
129-channel EGI at **500 Hz**, stored as EDF. Final analysis:

- 4 subjects, 8 sessions (2 each), 4,608 trials
- Sessions: LTP269 [12, 20], LTP293 [5, 22], LTP299 [2, 6], LTP303 [10, 22]
- 2,423 later recalled, 2,185 forgotten

Each session was validated before use: it had to be ltpFR2, have 576 studied
words, 100% word coverage against our list, and a recording long enough that
every word's analysis window fits inside it.

## 2. Word → T5 embedding (the "answer key")

For each of the 576 words `w`:

1. Tokenize `w` with T5's tokenizer → subword tokens `t_1 … t_k`.
2. Run through **T5-large's encoder only** (`T5EncoderModel`) with
   `output_hidden_states=True`.
3. Take the **middle encoder layer**, `hidden_states[12]` (T5-large has 24
   encoder layers; index 0 is the embedding output, so 12 is the middle),
   giving a vector `h_j` per token.
4. Average the subword tokens, excluding the end-of-sequence (`</s>`) and
   padding tokens:

```
e_w = (1/k) · Σ_{j=1..k} h_j          e_w ∈ R^1024
```

Stacking all 576 words gives a **576 × 1024** matrix, with word→row saved in a
CSV so every row is traceable to its word.

## 3. Recalled labels (the memory outcome)

Labels come from the behavioral event stream, **free recall only**:

- `WORD` events = the studied (presented) words.
- `REC_WORD` events = the words the subject freely recalled.
- A studied word is `recalled = 1` if a `REC_WORD` with the **same item number**
  occurs **in the same list/trial**; otherwise `recalled = 0`.
- Recognition events (`RECOG_*`, `recog_resp`, `recog_conf`) are **never used**.
- Matching is strictly within a list — an item number is never matched across
  different lists.

## 4. EEG feature per trial

For trial `i`, with the word's onset at EDF sample index `sample_i` and
`sfreq = 500 Hz`, take the preregistered **300–800 ms** window:

```
start = sample_i + int(0.300 × 500) = sample_i + 150
stop  = sample_i + int(0.800 × 500) = sample_i + 400     (stop EXCLUSIVE)
timepoints = 400 − 150 = 250
```

Using all **129 EEG channels**, the raw voltage block is `129 × 250`, flattened
in channel-major (C) order into one feature vector:

```
x_i ∈ R^(129 × 250) = R^32250
```

No filtering, baseline correction, or resampling — a deliberately simple first
pass on raw 500 Hz data.

## 5. Ridge regression (EEG → embedding)

Separately **within each subject/session**, fit multi-output ridge from EEG
features `X` to T5 targets `Y`:

```
minimize_W   ‖Y − X W‖_F²  +  α ‖W‖_F²          α = 10000
```

- **5-fold held-out-trial cross-validation** — train and test trials are always
  disjoint; every trial gets exactly **one out-of-fold prediction** `ŷ_i`.
- `StandardScaler` is fit on the **training fold only** and applied to test.
- SVD solver (features ≫ trials: 32,250 ≫ ~460), α = 10,000 is strong
  regularization suited to this wide, low-sample regime.

## 6. Embedding fidelity (cosine)

For each held-out trial, compare the predicted embedding `ŷ_i` to the true one
`y_i`:

```
embedding_fidelity_i = cos(ŷ_i, y_i) = (ŷ_i · y_i) / (‖ŷ_i‖ · ‖y_i‖)
```

Higher = the EEG pattern more accurately predicted the word's AI representation.

## 7. The memory model (the hypothesis test)

Because `recalled` is binary, use a **logistic mixed-effects** model:

```
recalled_i ~ embedding_fidelity_i + session_i + (1|subject_i) + (1|word_i)
```

- `embedding_fidelity` is **z-scored**, so the odds ratio is **per 1 SD** of
  fidelity.
- `session` is a fixed effect; `subject` and `word` are crossed random
  intercepts (accounting for easy/hard people and easy/hard words).
- Fit as a logistic GLMM (`statsmodels BinomialBayesMixedGLM`), with a
  cluster-robust logistic model as a cross-check.

The key prediction being tested: **remembered words should have higher
embedding_fidelity than forgotten words** (odds ratio > 1).

## 8. Supplementary sanity checks (why we didn't stop at raw cosine)

Raw cosine is inflated because all T5 vectors share a dominant common direction
(their norms are large, ~2,165). So we also computed **word-specific** metrics:
centered cosine (remove the common direction), true-word rank / retrieval
percentile among all 576 candidates, top-k retrieval, and a **shuffled-label
control**. These reveal whether the model decodes the *specific* word, not just
lands near the shared center.
