#!/usr/bin/env python3
"""
Run the memory model on a scaled fidelity table and line it up against the
earlier runs.

This is where the scale-up gets checked. The pipeline stays fixed and only the
number of subjects changes, so what it answers is whether the null survives more
data or whether the first run was just underpowered. It prints the odds ratio
alongside the 4- and 16-subject results, and re-aggregates the word-retrieval
sanity checks and the shuffled-label control over every session in the run.

Works for any N. It calls step11 instead of reimplementing the model, so the
scaled runs and the original 4-subject run go through the exact same code.

Usage:
  python code/step12_analyze_results.py --input outputs/all_sessions32_fidelity_results.csv --tag 32
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SUBJ = os.path.join(HERE, "outputs/subjects")

# Results from the earlier runs, hardcoded so a new run can be compared without
# refitting them (the 4- and 16-subject fidelity tables are large, and the 16 one
# isn't kept anymore). These only feed the comparison table, not the model. If
# either run ever gets refit, update these to match.
REF = {
    4:  {"subjects": 4,  "sessions": 8,  "trials": 4608,  "recalled": 2423, "forgotten": 2185,
         "rem": 0.84627, "forg": 0.84711, "diff": -0.00084, "OR": 0.9714, "coef": -0.0291,
         "ci": [0.9134, 1.0330], "p": 0.3544, "twp": 0.4981},
    16: {"subjects": 16, "sessions": 32, "trials": 18432, "recalled": 9980, "forgotten": 8452,
         "rem": 0.84337, "forg": 0.84367, "diff": -0.00030, "OR": 0.9792, "coef": -0.0211,
         "ci": [0.9468, 1.0127], "p": 0.2199, "twp": 0.4977},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tag", required=True, help="short label, e.g. 32")
    a = ap.parse_args()
    comb = pd.read_csv(a.input)
    n_sub = comb.subject.nunique()
    n_ses = comb.groupby(["subject", "session"]).ngroups
    n = len(comb)
    n_rec = int((comb.recalled == 1).sum()); n_forg = int((comb.recalled == 0).sum())
    rem = comb.loc[comb.recalled == 1, "embedding_fidelity"].mean()
    forg = comb.loc[comb.recalled == 0, "embedding_fidelity"].mean()

    # Fit via step11 as a subprocess. Every output is tagged with N, so a scaled
    # run never overwrites the canonical 4-subject files.
    fm_csv = os.path.join(HERE, f"outputs/final_memory_model{a.tag}_results.csv")
    subprocess.run([PY, os.path.join(HERE, "code/step11_run_memory_model.py"),
                    "--input", a.input,
                    "--out-summary", os.path.join(HERE, f"outputs/final_memory_model{a.tag}_summary.txt"),
                    "--out-csv", fm_csv,
                    "--out-meta", os.path.join(HERE, f"outputs/final_memory_model{a.tag}_metadata.json")],
                   check=True, capture_output=True, text=True)
    fmr = pd.read_csv(fm_csv)
    # The headline number is the GLMM on raw_cosine. step11 also writes the
    # fallback fit and three supplementary metrics to the same CSV, so it takes
    # both filters to pull out the one row the plan asks about.
    mr = fmr[(fmr.metric == "raw_cosine") & (fmr.model.str.contains("GLMM"))].iloc[0]
    OR, coef, p = float(mr.odds_ratio), float(mr.coef_per_sd), float(mr.p_approx)
    ci = [float(mr.ci_low), float(mr.ci_high)]; concl = str(mr.direction)

    # sanity + shuffled aggregation across sessions
    rank_cols = ["true_word_rank", "true_word_percentile", "top1_correct", "top5_correct",
                 "top10_correct", "centered_true_word_percentile", "centered_top5_correct",
                 "centered_top10_correct"]
    frames, rvs_list = [], []
    for (sub, ses), _ in comb.groupby(["subject", "session"]):
        d = os.path.join(SUBJ, f"sub-{sub}_ses-{ses}")
        fp = os.path.join(d, "fidelity_results_corrected.csv")
        if os.path.isfile(fp):
            frames.append(pd.read_csv(fp))
        mj = os.path.join(d, "ridge_corrected_metadata.json")
        if os.path.isfile(mj):
            rvs_list.append(json.load(open(mj)).get("real_vs_shuffled", {}))
    allrows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sanity = {c: float(allrows[c].mean()) for c in rank_cols if c in allrows.columns}

    def rvs(m, w):
        v = [r[m][w] for r in rvs_list if m in r]
        return float(np.mean(v)) if v else float("nan")
    shuf = {m: {"real": rvs(m, "real"), "shuffled": rvs(m, "shuffled")}
            for m in ["true_word_percentile", "centered_true_word_percentile",
                      "top5_correct", "top10_correct"]}

    verdict = ("STAYS NULL — no significant fidelity->recall effect."
               if concl == "no different" else
               "CHANGED — a significant effect appeared; investigate.")

    L = [
        "=" * 80,
        f"{n_sub}-SUBJECT SCALE-UP  (vs 16- and 4-subject)",
        "=" * 80,
        "Pipeline IDENTICAL across all runs (PEERS ltpFR2, T5-large middle-layer",
        "embeddings, 300-800 ms window, per subject/session ridge alpha=10000, held-out",
        "CV, cosine fidelity, recalled ~ embedding_fidelity + session + (1|subject) +",
        "(1|word)). Only the number of subjects changed.",
        "",
        "PROGRESSION (main model: raw cosine, OR per 1 SD)",
        f"  {'metric':16} {'4-subj':>12} {'16-subj':>12} {f'{n_sub}-subj':>12}",
        f"  {'subjects':16} {REF[4]['subjects']:>12} {REF[16]['subjects']:>12} {n_sub:>12}",
        f"  {'sessions':16} {REF[4]['sessions']:>12} {REF[16]['sessions']:>12} {n_ses:>12}",
        f"  {'trials':16} {REF[4]['trials']:>12} {REF[16]['trials']:>12} {n:>12}",
        f"  {'recalled':16} {REF[4]['recalled']:>12} {REF[16]['recalled']:>12} {n_rec:>12}",
        f"  {'forgotten':16} {REF[4]['forgotten']:>12} {REF[16]['forgotten']:>12} {n_forg:>12}",
        f"  {'odds ratio':16} {REF[4]['OR']:>12.4f} {REF[16]['OR']:>12.4f} {OR:>12.4f}",
        f"  {'coefficient':16} {REF[4]['coef']:>12.4f} {REF[16]['coef']:>12.4f} {coef:>12.4f}",
        f"  {'ci_low':16} {REF[4]['ci'][0]:>12.4f} {REF[16]['ci'][0]:>12.4f} {ci[0]:>12.4f}",
        f"  {'ci_high':16} {REF[4]['ci'][1]:>12.4f} {REF[16]['ci'][1]:>12.4f} {ci[1]:>12.4f}",
        f"  {'p-value':16} {REF[4]['p']:>12.4f} {REF[16]['p']:>12.4f} {p:>12.4f}",
        f"  {'remembered':16} {REF[4]['rem']:>12.5f} {REF[16]['rem']:>12.5f} {rem:>12.5f}",
        f"  {'forgotten':16} {REF[4]['forg']:>12.5f} {REF[16]['forg']:>12.5f} {forg:>12.5f}",
        f"  {'diff':16} {REF[4]['diff']:>+12.5f} {REF[16]['diff']:>+12.5f} {rem-forg:>+12.5f}",
        f"  conclusion ({n_sub}-subj): {concl}",
        "",
        f"WORD-SPECIFIC DECODING SANITY ({n_sub}-subject, averaged over sessions)",
        "  chance: percentile 0.5, top5 ~0.0087, top10 ~0.0174",
    ]
    for k, v in sanity.items():
        L.append(f"  {k:32}: {v:.4f}")
    L += ["", f"REAL vs SHUFFLED-LABEL CONTROL ({n_sub}-subject)",
          f"  {'metric':32} {'real':>8} {'shuffled':>9}"]
    for m, d in shuf.items():
        L.append(f"  {m:32} {d['real']:>8.4f} {d['shuffled']:>9.4f}")
    L += ["", "VERDICT",
          f"  Memory effect: {verdict}",
          "  Word-specific decoding: " + (
              "at chance (percentile ~0.5, real ~= shuffled)."
              if abs(sanity.get("true_word_percentile", 0.5) - 0.5) < 0.03 else
              f"ABOVE chance (percentile {sanity.get('true_word_percentile'):.3f}) — investigate."),
          "",
          "FINAL CONCLUSION",
          "  The project tested whether later-remembered words showed higher EEG-to-AI",
          f"  embedding fidelity than forgotten words. With {n_sub} subjects, the",
          "  session-aware logistic mixed-effects model " + (
              "still did not support this prediction: embedding fidelity was not "
              "significantly associated with later recall." if concl == "no different"
              else "CHANGED — see table above; interpret carefully."),
          "=" * 80]
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, f"outputs/{n_sub}subject_results_summary.txt"), "w").write(txt)
    open(os.path.join(HERE, f"outputs/{n_sub}subject_results_summary.md"), "w").write(
        f"# {n_sub}-Subject Scale-Up — Results\n\n```\n" + txt + "```\n")
    print(txt)


if __name__ == "__main__":
    main()
