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
    args = ap.parse_args()
    combined = pd.read_csv(args.input)
    n_sub = combined.subject.nunique()
    n_ses = combined.groupby(["subject", "session"]).ngroups
    n_trials = len(combined)
    n_rec = int((combined.recalled == 1).sum()); n_forg = int((combined.recalled == 0).sum())
    rem = combined.loc[combined.recalled == 1, "embedding_fidelity"].mean()
    forg = combined.loc[combined.recalled == 0, "embedding_fidelity"].mean()

    # Fit via step11 as a subprocess. Every output is tagged with N, so a scaled
    # run never overwrites the canonical 4-subject files.
    model_csv = os.path.join(HERE, f"outputs/final_memory_model{args.tag}_results.csv")
    subprocess.run([PY, os.path.join(HERE, "code/step11_run_memory_model.py"),
                    "--input", args.input,
                    "--out-summary", os.path.join(HERE, f"outputs/final_memory_model{args.tag}_summary.txt"),
                    "--out-csv", model_csv,
                    "--out-meta", os.path.join(HERE, f"outputs/final_memory_model{args.tag}_metadata.json")],
                   check=True, capture_output=True, text=True)
    final_model_rows = pd.read_csv(model_csv)
    # The headline number is the GLMM on raw_cosine; step11 also writes the fallback
    # fit and supplementary metrics to the same CSV, so both filters are needed.
    main_row = final_model_rows[(final_model_rows.metric == "raw_cosine")
                                & (final_model_rows.model.str.contains("GLMM"))].iloc[0]
    OR, coef, p = float(main_row.odds_ratio), float(main_row.coef_per_sd), float(main_row.p_approx)
    ci = [float(main_row.ci_low), float(main_row.ci_high)]; conclusion = str(main_row.direction)

    # Aggregate the per-session sanity metrics and shuffled-label control.
    rank_cols = ["true_word_rank", "true_word_percentile", "top1_correct", "top5_correct",
                 "top10_correct", "centered_true_word_percentile", "centered_top5_correct",
                 "centered_top10_correct"]
    per_session_frames, real_vs_shuffled_records = [], []
    for (sub, ses), _ in combined.groupby(["subject", "session"]):
        session_dir = os.path.join(SUBJ, f"sub-{sub}_ses-{ses}")
        fidelity_path = os.path.join(session_dir, "fidelity_results_corrected.csv")
        if os.path.isfile(fidelity_path):
            per_session_frames.append(pd.read_csv(fidelity_path))
        metadata_path = os.path.join(session_dir, "ridge_corrected_metadata.json")
        if os.path.isfile(metadata_path):
            real_vs_shuffled_records.append(json.load(open(metadata_path)).get("real_vs_shuffled", {}))
    all_fidelity = pd.concat(per_session_frames, ignore_index=True) if per_session_frames else pd.DataFrame()
    sanity_means = {c: float(all_fidelity[c].mean()) for c in rank_cols if c in all_fidelity.columns}

    def real_vs_shuffled_mean(metric, which):
        values = [record[metric][which] for record in real_vs_shuffled_records if metric in record]
        return float(np.mean(values)) if values else float("nan")
    shuffled_control = {metric: {"real": real_vs_shuffled_mean(metric, "real"),
                                 "shuffled": real_vs_shuffled_mean(metric, "shuffled")}
                        for metric in ["true_word_percentile", "centered_true_word_percentile",
                                       "top5_correct", "top10_correct"]}

    verdict = ("STAYS NULL — no significant fidelity->recall effect."
               if conclusion == "no different" else
               "CHANGED — a significant effect appeared; investigate.")

    report_lines = [
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
        f"  {'trials':16} {REF[4]['trials']:>12} {REF[16]['trials']:>12} {n_trials:>12}",
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
        f"  conclusion ({n_sub}-subj): {conclusion}",
        "",
        f"WORD-SPECIFIC DECODING SANITY ({n_sub}-subject, averaged over sessions)",
        "  chance: percentile 0.5, top5 ~0.0087, top10 ~0.0174",
    ]
    for name, value in sanity_means.items():
        report_lines.append(f"  {name:32}: {value:.4f}")
    report_lines += ["", f"REAL vs SHUFFLED-LABEL CONTROL ({n_sub}-subject)",
                     f"  {'metric':32} {'real':>8} {'shuffled':>9}"]
    for metric, pair in shuffled_control.items():
        report_lines.append(f"  {metric:32} {pair['real']:>8.4f} {pair['shuffled']:>9.4f}")
    report_lines += ["", "VERDICT",
          f"  Memory effect: {verdict}",
          "  Word-specific decoding: " + (
              "at chance (percentile ~0.5, real ~= shuffled)."
              if abs(sanity_means.get("true_word_percentile", 0.5) - 0.5) < 0.03 else
              f"ABOVE chance (percentile {sanity_means.get('true_word_percentile'):.3f}) — investigate."),
          "",
          "FINAL CONCLUSION",
          "  The project tested whether later-remembered words showed higher EEG-to-AI",
          f"  embedding fidelity than forgotten words. With {n_sub} subjects, the",
          "  session-aware logistic mixed-effects model " + (
              "still did not support this prediction: embedding fidelity was not "
              "significantly associated with later recall." if conclusion == "no different"
              else "CHANGED — see table above; interpret carefully."),
          "=" * 80]
    report_text = "\n".join(report_lines) + "\n"
    open(os.path.join(HERE, f"outputs/{n_sub}subject_results_summary.txt"), "w").write(report_text)
    open(os.path.join(HERE, f"outputs/{n_sub}subject_results_summary.md"), "w").write(
        f"# {n_sub}-Subject Scale-Up — Results\n\n```\n" + report_text + "```\n")
    print(report_text)


if __name__ == "__main__":
    main()
