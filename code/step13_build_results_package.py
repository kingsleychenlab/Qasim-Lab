#!/usr/bin/env python3
"""
Render the presentable results/ package: summary tables + figures.

Reads only the committed, curated tables in results/tables/ (not the
regenerable outputs/ working directory), so this script runs on a fresh clone:

    results/tables/trial_fidelity_<NN>subjects.csv      (per-trial fidelity)
    results/tables/memory_model_<NN>subjects_results.csv (fitted model)
    results/embeddings/peers_t5large_embeddings.npy      (for the norms check)

It runs no analysis of its own. Every number it draws was produced upstream by
step08 (fidelity) and step11 (memory model), so if a figure disagrees with a
summary the bug is in this script, not in the result.

There are only a few figures, one per question a reader tends to ask. The
method overview is pipeline_flow. Whether remembered and forgotten words differ
is remembered_vs_forgotten_fidelity. Whether the decoder does anything is
decoding_chance_check and raw_vs_centered_metrics. Whether the model agrees is
final_model_odds_ratios, and whether more data changes the answer is
scaling_progression.

Usage:
    python code/step13_build_results_package.py            # headline: 32 subjects
    python code/step13_build_results_package.py --tag 04   # rebuild for 4 subjects
"""

import argparse
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


def glmm_row(path, metric="raw_cosine"):
    """The fitted mixed-effects row for one metric, from step11's output."""
    fm = pd.read_csv(path)
    return fm[(fm.metric == metric) & (fm.model.str.contains("GLMM"))].iloc[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="32", help="headline run: 04, 16 or 32")
    args = ap.parse_args()
    tag = args.tag

    trials_p = os.path.join(TAB, f"trial_fidelity_{tag}subjects.csv")
    model_p = os.path.join(TAB, f"memory_model_{tag}subjects_results.csv")
    for p in (trials_p, model_p):
        if not os.path.isfile(p):
            raise SystemExit(f"ERROR: required input missing: {p}")

    df = pd.read_csv(trials_p)
    emb = np.load(os.path.join(RES, "embeddings/peers_t5large_embeddings.npy"))
    n_sub = df.subject.nunique()
    n_ses = df.groupby(["subject", "session"]).ngroups
    rec = df[df.recalled == 1]
    forg = df[df.recalled == 0]
    print(f"headline run: {n_sub} subjects, {n_ses} sessions, {len(df)} trials")

    # ================================================================ tables
    g = df.groupby(["subject", "session"])
    pd.DataFrame({
        "n_trials": g.size(),
        "recalled": g.recalled.sum().astype(int),
        "forgotten": (g.size() - g.recalled.sum()).astype(int),
        "recall_rate": g.recalled.mean().round(4),
        "mean_embedding_fidelity": g.embedding_fidelity.mean().round(5),
        "mean_centered_percentile": g.centered_true_word_percentile.mean().round(5),
    }).reset_index().to_csv(
        os.path.join(TAB, f"summary_subjects_sessions_{tag}subjects.csv"), index=False)
    created.append(f"results/tables/summary_subjects_sessions_{tag}subjects.csv")

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
    rvf.to_csv(os.path.join(TAB, f"remembered_vs_forgotten_{tag}subjects.csv"), index=False)
    created.append(f"results/tables/remembered_vs_forgotten_{tag}subjects.csv")

    # final model table (all metrics, headline run)
    fm = pd.read_csv(model_p)
    glmm = fm[fm.model.str.contains("GLMM")].copy()
    label = {"raw_cosine": "raw_cosine (embedding_fidelity, MAIN)",
             "centered_cosine": "centered_cosine",
             "true_word_percentile": "true_word_percentile",
             "centered_true_word_percentile": "centered_true_word_percentile"}
    order = {"raw_cosine": 0, "centered_cosine": 1, "true_word_percentile": 2,
             "centered_true_word_percentile": 3}
    pd.DataFrame({
        "metric": glmm.metric.map(label).fillna(glmm.metric),
        "coefficient": glmm.coef_per_sd.round(4),
        "odds_ratio": glmm.odds_ratio.round(4),
        "ci_lower": glmm.ci_low.round(4),
        "ci_upper": glmm.ci_high.round(4),
        "p_value": glmm.p_approx.round(4),
        "conclusion": glmm.direction.map({"no different": "no significant effect"}
                                         ).fillna(glmm.direction),
        "_o": glmm.metric.map(order),
    }).sort_values("_o").drop(columns="_o").to_csv(
        os.path.join(TAB, f"final_model_table_{tag}subjects.csv"), index=False)
    created.append(f"results/tables/final_model_table_{tag}subjects.csv")

    # ================================================================ figures
    # 1) pipeline schematic (method overview; data-independent)
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.axis("off")
    steps = ["PEERS words\n(576)", "T5-large\nembeddings\n576x1024",
             "EEG 300-800 ms\n129x250 = 32250", "Ridge regression\n(alpha=1e4, CV)",
             "cosine fidelity\ncos(y_hat, y)", "Logistic\nmixed-effects\nmemory model"]
    w, h, gap, x = 1.9, 1.5, 0.35, 0.1
    centers = []
    for i, s in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x, 0.4), w, h, boxstyle="round,pad=0.08",
                                    linewidth=1.5, edgecolor=BLUE,
                                    facecolor="#ebf4ff" if i % 2 == 0 else "#fffaf0"))
        ax.text(x + w / 2, 0.4 + h / 2, s, ha="center", va="center", fontsize=9.5)
        centers.append(x + w)
        x += w + gap
    for i in range(len(steps) - 1):
        ax.add_patch(FancyArrowPatch((centers[i], 1.15), (centers[i] + gap, 1.15),
                                     arrowstyle="-|>", mutation_scale=16, color=GRAY, lw=1.5))
    ax.set_xlim(0, x); ax.set_ylim(0, 2.4)
    ax.set_title("Pipeline: word -> T5 embedding -> EEG window -> ridge -> fidelity -> memory model",
                 fontsize=11)
    save(fig, "pipeline_flow.png")

    # 2) the core comparison: remembered vs forgotten
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bp = ax.boxplot([forg.embedding_fidelity.values, rec.embedding_fidelity.values],
                    tick_labels=["forgotten", "remembered"], patch_artist=True,
                    widths=0.55, showmeans=True, meanline=True)
    for patch, c in zip(bp["boxes"], [GRAY, BLUE]):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    ax.set_ylabel("embedding_fidelity (raw cosine)")
    ax.set_title("Remembered vs forgotten fidelity — distributions nearly identical\n"
                 f"{n_sub} subjects | forgotten {forg.embedding_fidelity.mean():.4f}  "
                 f"remembered {rec.embedding_fidelity.mean():.4f}")
    save(fig, "remembered_vs_forgotten_fidelity.png")

    # 3) is the decoder doing anything? retrieval vs chance
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(df.true_word_percentile, bins=40, alpha=0.55, color=BLUE,
            label=f"true_word_percentile (mean {df.true_word_percentile.mean():.3f})")
    ax.hist(df.centered_true_word_percentile, bins=40, alpha=0.55, color=ORANGE,
            label=f"centered (mean {df.centered_true_word_percentile.mean():.3f})")
    ax.axvline(0.5, color="black", ls="--", lw=2, label="chance = 0.5")
    ax.set_xlabel("word-retrieval percentile (higher = better)")
    ax.set_ylabel("trials")
    ax.set_title(f"Word-specific decoding sits at chance ({n_sub} subjects)")
    ax.legend()
    save(fig, "decoding_chance_check.png")

    # 4) why raw cosine is misleading
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(df.raw_cosine, bins=40, color=BLUE, edgecolor="white")
    axes[0].axvline(df.raw_cosine.mean(), color=ORANGE, lw=2)
    axes[0].set_title(f"raw_cosine (inflated)\nmean = {df.raw_cosine.mean():.3f}")
    axes[0].set_xlabel("raw cosine"); axes[0].set_xlim(0, 1)
    axes[1].hist(df.centered_true_word_percentile, bins=40, color=BLUE, edgecolor="white")
    axes[1].axvline(0.5, color=GRAY, ls="--", lw=2, label="chance = 0.5")
    axes[1].axvline(df.centered_true_word_percentile.mean(), color=ORANGE, lw=2,
                    label=f"mean = {df.centered_true_word_percentile.mean():.3f}")
    axes[1].set_title("centered_true_word_percentile (at chance)")
    axes[1].set_xlabel("centered true-word percentile"); axes[1].set_xlim(0, 1)
    axes[1].legend()
    norms = np.linalg.norm(emb, axis=1)
    fig.suptitle("High raw cosine is a common-direction artifact "
                 f"(T5 norms mean {norms.mean():.0f}), not word decoding", fontsize=11)
    save(fig, "raw_vs_centered_metrics.png")

    # 5) the model: odds ratios with OR=1 reference
    fig, ax = plt.subplots(figsize=(8.5, 4))
    fp = glmm.assign(_o=glmm.metric.map(order)).sort_values("_o", ascending=False)
    y = np.arange(len(fp))
    ax.errorbar(fp.odds_ratio, y,
                xerr=[fp.odds_ratio - fp.ci_low, fp.ci_high - fp.odds_ratio],
                fmt="o", color=BLUE, ecolor=GRAY, elinewidth=2, capsize=4, ms=8)
    ax.axvline(1.0, color="black", ls="--", lw=1.5, label="OR = 1 (no effect)")
    ax.set_yticks(y); ax.set_yticklabels(fp.metric.map(label).fillna(fp.metric))
    ax.set_xlabel("odds ratio per 1 SD (95% interval)")
    ax.set_title(f"Memory-model odds ratios ({n_sub} subjects) — all intervals span OR = 1")
    ax.legend(loc="lower right")
    save(fig, "final_model_odds_ratios.png")

    # 6) does more data change the answer? 4 -> 16 -> 32
    prog = []
    for t in ("04", "16", "32"):
        p = os.path.join(TAB, f"memory_model_{t}subjects_results.csv")
        if os.path.isfile(p):
            r = glmm_row(p)
            prog.append((int(t), float(r.odds_ratio), float(r.ci_low), float(r.ci_high)))
    if len(prog) >= 2:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        xs = np.arange(len(prog))
        ors = [p[1] for p in prog]
        lo = [p[1] - p[2] for p in prog]
        hi = [p[3] - p[1] for p in prog]
        ax.errorbar(xs, ors, yerr=[lo, hi], fmt="o", color=BLUE, ecolor=GRAY,
                    elinewidth=2, capsize=5, ms=9)
        ax.axhline(1.0, color="black", ls="--", lw=1.5, label="OR = 1 (no effect)")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{p[0]} subjects" for p in prog])
        ax.set_ylabel("odds ratio per 1 SD (95% interval)")
        ax.set_title("More data does not reveal an effect:\n"
                     "the odds ratio converges on 1.0 as the interval tightens")
        ax.legend()
        save(fig, "scaling_progression.png")

    print("\nCreated / refreshed:")
    for c in created:
        print("  " + c)
    print(f"\nTotal: {len(created)} files")


if __name__ == "__main__":
    main()
