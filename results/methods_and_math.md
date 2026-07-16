# Methods and math

Notation: a word is `w`, and a trial (one word presentation) is indexed `i`.

## 1. T5 word embedding

The T5 SentencePiece tokenizer splits each word `w` into subword tokens
`t_1, …, t_k`, plus an appended end-of-sequence token `</s>`. That goes through
the T5-large encoder (`T5EncoderModel`, encoder only) with
`output_hidden_states=True`. I take the middle encoder layer, `hidden_states[12]`
(T5-large has 24 encoder layers, and index 0 is the embedding output, so 12 is
the middle), which gives a hidden vector `h_j ∈ R^1024` for each token.

Dropping the EOS (`</s>`) and padding tokens, the word embedding is the mean over
the real subword tokens:

```
e_w = (1 / k) · Σ_{j=1..k} h_j          e_w ∈ R^1024
```

Stack that over the 576-word pool and you get the target matrix, 576 × 1024.

One thing to keep in mind for §4: T5 embeddings aren't unit-normalized, and they
share a strong common direction. Their L2 norms are large, mean around 2165.

## 2. EEG feature vector

For each word presentation I take a fixed window, 300-800 ms after word onset.
With sampling rate `sfreq = 500 Hz` and the integer EDF sample index `sample` of
the onset:

```
start = sample + int(0.300 × 500) = sample + 150
stop  = sample + int(0.800 × 500) = sample + 400      (stop is EXCLUSIVE)
timepoints = stop − start = 250
```

Using all 129 EEG channels, the raw voltage segment is 129 channels × 250
timepoints, flattened channel-major (C-order) into

```
x_i ∈ R^(129 × 250) = R^32250
```

No filtering, baseline correction, or resampling. This first pass keeps the raw
500 Hz data. A session's design matrix is 576 × 32250.

## 3. Ridge regression (per subject/session)

Within each subject/session I fit a multi-output ridge from the EEG features `X`
to the T5 targets `Y`:

```
minimize_W  ‖Y − X W‖_F²  +  α ‖W‖_F²          α = 10000
```

- 5-fold `KFold(shuffle=True, random_state=42)` over trials, so the train and
  test trials are always disjoint — no trial is scored by a model that trained on
  it.
- `StandardScaler` is fit on the training fold only, then applied to both.
- Solver `svd`, since features far outnumber samples (32250 ≫ ~460).
- Each trial ends up with exactly one out-of-fold prediction `ŷ_i`.

## 4. Embedding fidelity

For a held-out trial with predicted embedding `ŷ_i` and true embedding `y_i`:

```
fidelity_i = cos(ŷ_i, y_i) = (ŷ_i · y_i) / (‖ŷ_i‖ · ‖y_i‖)
```

This raw cosine is the `embedding_fidelity` the memory model uses. Because T5
vectors share a dominant common direction, the raw cosine runs high (~0.85) even
without any real word decoding — see §6.

## 5. Memory model

Recall is binary per trial, so I model it with a logistic mixed-effects model:

```
recalled_i ~ embedding_fidelity_i + session_i + (1 | subject_i) + (1 | word_i)
```

- `recalled ∈ {0, 1}`, logistic link.
- `embedding_fidelity` is z-scored, so the reported odds ratio is per 1 SD
  increase in fidelity.
- `session` is a fixed effect (dummy-coded).
- Random intercepts for `subject` and `word`, crossed.
- The primary fit is `statsmodels BinomialBayesMixedGLM` (variational Bayes). As
  a fallback I also run a logistic model with subject/session fixed effects and
  cluster-robust standard errors by subject.

## 6. Supplementary checks (sanity, not the main result)

Since the raw cosine is inflated by the common embedding direction, I also
computed word-specific decoding metrics:

- **centered cosine** — subtract the training-fold mean embedding from both the
  prediction and the target before taking the cosine, which removes the common
  direction.
- **true-word rank** — rank the correct word among all 576 candidate embeddings
  by cosine to the prediction (1 = best).
- **retrieval percentile** — `1 − (rank − 1) / 575`, where 0.5 is chance.
- **top-k retrieval** — whether the correct word lands in the top 1/5/10.
- **shuffled-label control** — rerun the whole CV with `Y` permuted across trials;
  a real model has to beat this.

Across all sessions these word-specific metrics sat at chance (percentile ≈ 0.5,
real ≈ shuffled), which confirms that the high raw cosine is the common-direction
artifact rather than genuine word-level decoding.
