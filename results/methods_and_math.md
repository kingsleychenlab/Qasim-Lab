# Methods and Math

All notation: a word is `w`; a trial (one word presentation) is indexed `i`.

## 1. T5 word embedding

Each word `w` is tokenized by the T5 SentencePiece tokenizer into subword
tokens `t_1, …, t_k` (plus an appended end-of-sequence token `</s>`). The word
is passed through the **T5-large encoder** (`T5EncoderModel`, encoder only) with
`output_hidden_states=True`. We take the **middle encoder layer**,
`hidden_states[12]` (T5-large has 24 encoder layers; index 0 is the embedding
output, so 12 is the middle), giving a hidden vector `h_j ∈ R^1024` per token.

Excluding the EOS (`</s>`) and padding tokens, the word embedding is the mean
over the real subword tokens:

```
e_w = (1 / k) · Σ_{j=1..k} h_j          e_w ∈ R^1024
```

Stacked over the 576-word pool this gives the target matrix **576 × 1024**.

> Note: T5 embeddings are NOT unit-normalized and share a strong common
> direction; their L2 norms are large (mean ≈ 2165). This matters for §4.

## 2. EEG feature vector

For each word presentation we take a fixed window **300–800 ms after word
onset**. With sampling rate `sfreq = 500 Hz` and the integer EDF sample index
`sample` of the onset:

```
start = sample + int(0.300 × 500) = sample + 150
stop  = sample + int(0.800 × 500) = sample + 400      (stop is EXCLUSIVE)
timepoints = stop − start = 250
```

Using all **129 EEG channels**, the raw voltage segment is
`(129 channels × 250 timepoints)`, flattened channel-major (C-order) into

```
x_i ∈ R^(129 × 250) = R^32250
```

No filtering, baseline correction, or resampling is applied (first pass keeps
raw 500 Hz data). A session's design matrix is **576 × 32250**.

## 3. Ridge regression (per subject/session)

Within each subject/session we fit multi-output ridge from EEG features `X` to
T5 targets `Y`:

```
minimize_W  ‖Y − X W‖_F²  +  α ‖W‖_F²          α = 10000
```

- 5-fold `KFold(shuffle=True, random_state=42)` over trials — train and test
  trials are always disjoint (no trial is tested on a model that trained on it).
- `StandardScaler` is fit on the **training** fold only and applied to both.
- Solver `svd` (features ≫ samples: 32250 ≫ ~460).
- Each trial receives exactly one **out-of-fold** prediction `ŷ_i`.

## 4. Embedding fidelity

For a held-out trial with predicted embedding `ŷ_i` and true embedding `y_i`:

```
fidelity_i = cos(ŷ_i, y_i) = (ŷ_i · y_i) / (‖ŷ_i‖ · ‖y_i‖)
```

This **raw cosine** is the `embedding_fidelity` used by the outline memory
model. Because T5 vectors share a dominant common direction, raw cosine is
inflated (~0.85) even without real word decoding — see §6.

## 5. Memory model

The binary recall outcome per trial is modeled with a **logistic
mixed-effects** model:

```
recalled_i ~ embedding_fidelity_i + session_i + (1 | subject_i) + (1 | word_i)
```

- `recalled ∈ {0, 1}` (logistic link).
- `embedding_fidelity` is **z-scored**, so the reported **odds ratio is per 1 SD
  increase** in fidelity.
- `session` is a fixed effect (dummy-coded).
- random intercepts for `subject` and `word` (crossed).
- Primary fit: `statsmodels BinomialBayesMixedGLM` (variational Bayes). A
  logistic model with subject/session fixed effects and cluster-robust standard
  errors by subject is run as a fallback.

## 6. Supplementary checks (sanity, not the main result)

Because raw cosine is inflated by the common embedding direction, we also
computed **word-specific** decoding metrics:

- **centered cosine** — subtract the training-fold mean embedding from both
  prediction and target before cosine (removes the common direction).
- **true-word rank** — rank the correct word among all 576 candidate embeddings
  by cosine to the prediction (1 = best).
- **retrieval percentile** — `1 − (rank − 1) / 575` (0.5 = chance).
- **top-k retrieval** — whether the correct word is in the top 1/5/10.
- **shuffled-label control** — repeat the whole CV with `Y` permuted across
  trials; a real model must beat this.

Across all sessions these word-specific metrics sat at **chance** (percentile
≈ 0.5; real ≈ shuffled), confirming that the high raw cosine reflects the
common-direction artifact, not genuine word-level decoding.
