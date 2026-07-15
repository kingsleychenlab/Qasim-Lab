#!/usr/bin/env python3
"""
Turn the pipeline's raw outputs into the presentable results/ package.

Reads what the pipeline already computed and renders it: the summary tables in
results/tables/ and the ten figures in results/figures/. Deliberately performs
no analysis of its own -- every number it draws was produced upstream by step08
and step11. If a figure disagrees with a summary, the bug is here, not in the
result.

Strictly read-only with respect to the pipeline: it reads outputs/ and
results/embeddings/, and writes only under results/. Safe to re-run.

The figures answer the three questions a reader asks in order:
  - is there a fidelity difference between remembered and forgotten words?
    (remembered_vs_forgotten_fidelity, embedding_fidelity_histogram)
  - is the decoder doing anything at all?
    (decoding_chance_check, raw_vs_centered_metrics, t5_embedding_norms)
  - does the model back that up?
    (final_model_odds_ratios)
plus per-subject/session breakdowns and a pipeline schematic.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.autolayout": True,
})
BLUE, ORANGE, GRAY = "#2b6cb0", "#dd6b20", "#718096"
created = []


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    created.append(os.path.relpath(p, HERE))


# ---------------------------------------------------------------- load
df = pd.read_csv(os.path.join(HERE, "outputs/all_sessions_fidelity_results.csv"))
emb = np.load(os.path.join(HERE, "results/embeddings/peers_t5large_embeddings.npy"))
fm = pd.read_csv(os.path.join(HERE, "outputs/final_memory_model_results.csv"))

df["sesid"] = df.subject.astype(str) + "/" + df.session.astype(str)
rec = df[df.recalled == 1]
forg = df[df.recalled == 0]

# ================================================================ CSVs
# 1) per subject/session summary
g = df.groupby(["subject", "session"])
sss = pd.DataFrame({
    "n_trials": g.size(),
    "recalled": g.recalled.sum().astype(int),
    "forgotten": (g.size() - g.recalled.sum()).astype(int),
    "recall_rate": g.recalled.mean().round(4),
    "mean_embedding_fidelity": g.embedding_fidelity.mean().round(5),
    "mean_centered_percentile": g.centered_true_word_percentile.mean().round(5),
}).reset_index()
sss.to_csv(os.path.join(TAB, "summary_subjects_sessions.csv"), index=False)
created.append("results/tables/summary_subjects_sessions.csv")

# 2) remembered vs forgotten summary
rvf = pd.DataFrame([
    {"group": "remembered", "n": len(rec),
     "mean_embedding_fidelity": rec.embedding_fidelity.mean(),
     "sd_embedding_fidelity": rec.embedding_fidelity.std(ddof=1),
     "mean_centered_true_word_percentile": rec.centered_true_word_percentile.mean(),
     "sd_centered_true_word_percentile": rec.centered_true_word_percentile.std(ddof=1)},
    {"group": "forgotten", "n": len(forg),
     "mean_embedding_fidelity": forg.embedding_fidelity.mean(),
     "sd_embedding_fidelity": forg.embedding_fidelity.std(ddof=1),
     "mean_centered_true_word_percentile": forg.centered_true_word_percentile.mean(),
     "sd_centered_true_word_percentile": forg.centered_true_word_percentile.std(ddof=1)},
])
for c in rvf.columns:
    if c not in ("group", "n"):
        rvf[c] = rvf[c].round(5)
rvf.to_csv(os.path.join(TAB, "remembered_vs_forgotten_summary.csv"), index=False)
created.append("results/tables/remembered_vs_forgotten_summary.csv")

# 3) final model table (GLMM rows only; raw_cosine main + centered supplementary)
glmm = fm[fm.model.str.contains("GLMM")].copy()
label = {"raw_cosine": "raw_cosine (embedding_fidelity, MAIN)",
         "centered_cosine": "centered_cosine",
         "true_word_percentile": "true_word_percentile",
         "centered_true_word_percentile": "centered_true_word_percentile"}
fmt = pd.DataFrame({
    "metric": glmm.metric.map(label).fillna(glmm.metric),
    "coefficient": glmm.coef_per_sd.round(4),
    "odds_ratio": glmm.odds_ratio.round(4),
    "ci_lower": glmm.ci_low.round(4),
    "ci_upper": glmm.ci_high.round(4),
    "p_value": glmm.p_approx.round(4),
    "conclusion": glmm.direction.map(
        {"no different": "no significant effect", "higher": "higher fidelity -> more recall",
         "lower": "lower fidelity -> more recall"}).fillna(glmm.direction),
})
# order: raw_cosine first
order = {"raw_cosine": 0, "centered_cosine": 1, "true_word_percentile": 2,
         "centered_true_word_percentile": 3}
fmt = fmt.assign(_o=glmm.metric.map(order)).sort_values("_o").drop(columns="_o")
fmt.to_csv(os.path.join(TAB, "final_model_table.csv"), index=False)
created.append("results/tables/final_model_table.csv")

# ================================================================ figures
# 1) pipeline flow
fig, ax = plt.subplots(figsize=(13, 2.6))
ax.axis("off")
steps = ["PEERS words\n(576)", "T5-large\nembeddings\n576x1024",
         "EEG 300-800 ms\n129x250 = 32250", "Ridge regression\n(alpha=1e4, CV)",
         "cosine fidelity\ncos(y_hat, y)", "Logistic\nmixed-effects\nmemory model"]
n = len(steps)
w, h, gap = 1.9, 1.5, 0.35
x = 0.1
centers = []
for i, s in enumerate(steps):
    box = FancyBboxPatch((x, 0.4), w, h, boxstyle="round,pad=0.08",
                         linewidth=1.5, edgecolor=BLUE,
                         facecolor="#ebf4ff" if i % 2 == 0 else "#fffaf0")
    ax.add_patch(box)
    ax.text(x + w / 2, 0.4 + h / 2, s, ha="center", va="center", fontsize=9.5)
    centers.append(x + w)
    x += w + gap
for i in range(n - 1):
    ax.add_patch(FancyArrowPatch((centers[i], 1.15), (centers[i] + gap, 1.15),
                                 arrowstyle="-|>", mutation_scale=16, color=GRAY, lw=1.5))
ax.set_xlim(0, x)
ax.set_ylim(0, 2.4)
ax.set_title("Neurolab pipeline: word -> T5 embedding -> EEG window -> ridge -> fidelity -> memory model",
             fontsize=11)
save(fig, "pipeline_flow.png")

# 2) embedding_fidelity histogram
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df.embedding_fidelity, bins=40, color=BLUE, edgecolor="white")
ax.axvline(df.embedding_fidelity.mean(), color=ORANGE, lw=2,
           label=f"mean = {df.embedding_fidelity.mean():.3f}")
ax.set_xlabel("embedding_fidelity (raw cosine)")
ax.set_ylabel("trials")
ax.set_title("Distribution of embedding fidelity (raw cosine), all 4608 trials")
ax.legend()
save(fig, "embedding_fidelity_histogram.png")

# 3) remembered vs forgotten fidelity (box)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
data = [forg.embedding_fidelity.values, rec.embedding_fidelity.values]
bp = ax.boxplot(data, tick_labels=["forgotten", "remembered"], patch_artist=True,
                widths=0.55, showmeans=True, meanline=True)
for patch, c in zip(bp["boxes"], [GRAY, BLUE]):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
ax.set_ylabel("embedding_fidelity (raw cosine)")
ax.set_title("Remembered vs forgotten fidelity — distributions nearly identical\n"
             f"means: forgotten {forg.embedding_fidelity.mean():.4f}  |  "
             f"remembered {rec.embedding_fidelity.mean():.4f}")
save(fig, "remembered_vs_forgotten_fidelity.png")

# 4) subject/session mean fidelity
fig, ax = plt.subplots(figsize=(9, 4.5))
order_ss = sss.assign(k=sss.subject + "/" + sss.session.astype(str)).sort_values("k")
ax.bar(order_ss.k, order_ss.mean_embedding_fidelity, color=BLUE, edgecolor="white")
ax.set_ylim(0.83, 0.855)
ax.set_ylabel("mean embedding_fidelity")
ax.set_xlabel("subject / session")
ax.set_title("Mean embedding fidelity by subject/session (note compressed y-axis)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
save(fig, "subject_session_fidelity.png")

# 5) recall rate by subject/session
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(order_ss.k, order_ss.recall_rate, color=ORANGE, edgecolor="white")
ax.axhline(0.5, color=GRAY, ls="--", lw=1, label="0.5")
ax.set_ylabel("recall rate")
ax.set_xlabel("subject / session")
ax.set_title("Recall rate by subject/session")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend()
save(fig, "recall_rate_by_subject_session.png")

# 6) raw vs centered metrics
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(df.raw_cosine, bins=40, color=BLUE, edgecolor="white")
axes[0].axvline(df.raw_cosine.mean(), color=ORANGE, lw=2)
axes[0].set_title(f"raw_cosine (inflated)\nmean = {df.raw_cosine.mean():.3f}")
axes[0].set_xlabel("raw cosine")
axes[0].set_xlim(0, 1)
axes[1].hist(df.centered_true_word_percentile, bins=40, color=BLUE, edgecolor="white")
axes[1].axvline(0.5, color=GRAY, ls="--", lw=2, label="chance = 0.5")
axes[1].axvline(df.centered_true_word_percentile.mean(), color=ORANGE, lw=2,
                label=f"mean = {df.centered_true_word_percentile.mean():.3f}")
axes[1].set_title("centered_true_word_percentile (at chance)")
axes[1].set_xlabel("centered true-word percentile")
axes[1].set_xlim(0, 1)
axes[1].legend()
fig.suptitle("Raw cosine is high (common-direction artifact); word-specific metric is at chance",
             fontsize=11)
save(fig, "raw_vs_centered_metrics.png")

# 7) decoding chance check
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.hist(df.true_word_percentile, bins=40, alpha=0.55, color=BLUE,
        label=f"true_word_percentile (mean {df.true_word_percentile.mean():.3f})")
ax.hist(df.centered_true_word_percentile, bins=40, alpha=0.55, color=ORANGE,
        label=f"centered (mean {df.centered_true_word_percentile.mean():.3f})")
ax.axvline(0.5, color="black", ls="--", lw=2, label="chance = 0.5")
ax.set_xlabel("word-retrieval percentile (higher = better)")
ax.set_ylabel("trials")
ax.set_title("Word-specific decoding sits at chance (0.5)")
ax.legend()
save(fig, "decoding_chance_check.png")

# 8) forest plot of odds ratios
fig, ax = plt.subplots(figsize=(8.5, 4))
fp = glmm.assign(_o=glmm.metric.map(order)).sort_values("_o", ascending=False)
ylabs = fp.metric.map(label).fillna(fp.metric)
y = np.arange(len(fp))
ax.errorbar(fp.odds_ratio, y,
            xerr=[fp.odds_ratio - fp.ci_low, fp.ci_high - fp.odds_ratio],
            fmt="o", color=BLUE, ecolor=GRAY, elinewidth=2, capsize=4, ms=8)
ax.axvline(1.0, color="black", ls="--", lw=1.5, label="OR = 1 (no effect)")
ax.set_yticks(y)
ax.set_yticklabels(ylabs)
ax.set_xlabel("odds ratio per 1 SD (95% interval)")
ax.set_title("Memory-model odds ratios — all intervals span OR = 1 (no effect)")
ax.legend(loc="lower right")
save(fig, "final_model_odds_ratios.png")

# 9) recalled counts
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.bar(["recalled", "forgotten"], [len(rec), len(forg)], color=[BLUE, GRAY],
       edgecolor="white")
for i, v in enumerate([len(rec), len(forg)]):
    ax.text(i, v + 20, str(v), ha="center", fontsize=12)
ax.set_ylabel("trials")
ax.set_title(f"Recalled vs forgotten (n = {len(df)} trials)")
save(fig, "recalled_counts.png")

# 10) T5 embedding norms
norms = np.linalg.norm(emb, axis=1)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(norms, bins=40, color=BLUE, edgecolor="white")
ax.axvline(norms.mean(), color=ORANGE, lw=2, label=f"mean = {norms.mean():.0f}")
ax.set_xlabel("L2 norm of T5 embedding vector")
ax.set_ylabel("words")
ax.set_title("T5-large embedding norms (576 words) — large norms drive raw-cosine inflation")
ax.legend()
save(fig, "t5_embedding_norms.png")

# ================================================================ done
print("Created files:")
for c in created:
    print("  " + c)
print(f"\nTotal: {len(created)} files")
