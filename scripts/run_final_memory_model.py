#!/usr/bin/env python3
"""
FINAL OUTLINE MEMORY MODEL (multi-subject, multi-session).

    recalled ~ embedding_fidelity + session + (1|subject) + (1|word)

This is THE model from the original project outline:
  - recalled is binary -> logistic mixed-effects.
  - embedding_fidelity == raw_cosine (the metric named in the outline).
  - embedding_fidelity is z-scored -> odds ratio is per 1 SD increase.
  - session is a fixed effect (dummy-coded, C(session)).
  - random intercepts for subject and word.

Primary fit: logistic GLMM (statsmodels BinomialBayesMixedGLM, variational
Bayes, crossed subject+word random intercepts). A logistic + subject/session
fixed-effects model with cluster-robust SEs (by subject) is also run as a
fallback (with only 4 subject-clusters, those SEs are very unreliable).

Supplementary SANITY models (NOT the main result) repeat the fit with
centered_cosine, true_word_percentile, centered_true_word_percentile.

CAUTION: the corrected decoding metrics were at CHANCE (real ~= shuffled word
retrieval), so any memory effect below must be interpreted cautiously.

Outputs:
  outputs/final_memory_model_summary.txt
  outputs/final_memory_model_results.csv
  outputs/final_memory_model_metadata.json
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_METRIC = "raw_cosine"            # == embedding_fidelity per the outline
SUPP_METRICS = ["centered_cosine", "true_word_percentile", "centered_true_word_percentile"]
FORMULA = "recalled ~ embedding_fidelity + session + (1|subject) + (1|word)"


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def fit_glmm(df, metric):
    """Logistic GLMM: recalled ~ zfid + C(session), RE 1|subject + 1|word."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from scipy.stats import norm
    d = df.copy()
    d["zfid"] = zscore(d[metric])
    d["session"] = d["session"].astype("category")
    d["subject"] = d["subject"].astype("category")
    d["word"] = d["word"].astype("category")
    try:
        vc = {"subject": "0 + C(subject)", "word": "0 + C(word)"}
        res = BinomialBayesMixedGLM.from_formula(
            "recalled ~ zfid + C(session)", vc, d).fit_vb(verbose=False)
        i = list(res.model.exog_names).index("zfid")
        mean, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
        z = mean / sd if sd > 0 else np.nan
        p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        return {"model": "logistic GLMM (BinomialBayesMixedGLM VB; +C(session); 1|subject+1|word)",
                "coef": mean, "se": sd, "odds_ratio": float(np.exp(mean)),
                "ci_low": float(np.exp(mean - 1.96 * sd)),
                "ci_high": float(np.exp(mean + 1.96 * sd)),
                "z_approx": z, "p_approx": p,
                "interval_kind": "95% credible interval (VB posterior mean +/- 1.96 SD)"}
    except Exception as e:  # noqa: BLE001
        return {"model": "logistic GLMM", "error": f"{type(e).__name__}: {e}"}


def fit_fallback(df, metric):
    """Logistic + session & subject fixed effects, cluster-robust SE by subject."""
    import statsmodels.formula.api as smf
    d = df.copy()
    d["zfid"] = zscore(d[metric])
    try:
        m = smf.logit("recalled ~ zfid + C(session) + C(subject)", data=d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": d["subject"]})
        coef, se, p = float(m.params["zfid"]), float(m.bse["zfid"]), float(m.pvalues["zfid"])
        ci = m.conf_int().loc["zfid"]
        return {"model": "logistic + C(session)+C(subject) FE + cluster-robust SE (by subject)",
                "coef": coef, "se": se, "odds_ratio": float(np.exp(coef)),
                "ci_low": float(np.exp(ci[0])), "ci_high": float(np.exp(ci[1])),
                "z_approx": coef / se if se > 0 else np.nan, "p_approx": p,
                "interval_kind": "95% CI (cluster-robust; ONLY 4 clusters -> very unreliable)"}
    except Exception as e:  # noqa: BLE001
        return {"model": "logistic + cluster-robust", "error": f"{type(e).__name__}: {e}"}


def verdict(res):
    if "error" in res:
        return "inconclusive (fit failed)"
    p, lo, hi = res.get("p_approx"), res.get("ci_low"), res.get("ci_high")
    if p is not None and np.isfinite(p) and p < 0.05 and not (lo <= 1.0 <= hi):
        return "higher" if res["coef"] > 0 else "lower"
    return "no different"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=os.path.join(HERE, "outputs/all_sessions_fidelity_results.csv"))
    ap.add_argument("--out-summary", default=os.path.join(HERE, "outputs/final_memory_model_summary.txt"))
    ap.add_argument("--out-csv", default=os.path.join(HERE, "outputs/final_memory_model_results.csv"))
    ap.add_argument("--out-meta", default=os.path.join(HERE, "outputs/final_memory_model_metadata.json"))
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: input not found: {args.input}")
    df = pd.read_csv(args.input)
    # embedding_fidelity is the outline metric == raw_cosine; sanity-check.
    if "embedding_fidelity" in df.columns and not np.allclose(
            df["embedding_fidelity"], df["raw_cosine"]):
        sys.exit("ERROR: embedding_fidelity != raw_cosine in input.")

    n_sub = df.subject.nunique()
    n_ses = df.groupby(["subject", "session"]).ngroups
    n_trials = len(df)
    n_rec = int((df.recalled == 1).sum())
    n_forg = int((df.recalled == 0).sum())

    fh = open(args.out_summary, "w")

    def log(*p):
        line = " ".join(str(x) for x in p)
        print(line)
        fh.write(line + "\n")

    log("=" * 80)
    log("FINAL OUTLINE MEMORY MODEL  (multi-subject, multi-session)")
    log("=" * 80)
    log(f"input             : {os.path.relpath(args.input, HERE)}")
    log(f"1. number of subjects : {n_sub}")
    log(f"2. number of sessions : {n_ses}")
    log(f"3. total trials       : {n_trials}")
    log(f"4. recalled/forgotten : {n_rec} / {n_forg}  (recall rate {n_rec/n_trials:.3f})")
    log(f"5. model formula      : {FORMULA}")
    log(f"6. embedding_fidelity metric: {MAIN_METRIC}  (== embedding_fidelity, per outline)")
    log("   predictor z-scored -> odds ratio is PER 1 SD; session = fixed effect C(session).")
    log("\n!!! CAUTION: corrected decoding was at CHANCE (real ~= shuffled word "
        "retrieval). Interpret any effect below cautiously. !!!")

    rows = []
    meta_out = {}
    all_metrics = [(MAIN_METRIC, "MAIN")] + [(m, "SUPPLEMENTARY") for m in SUPP_METRICS]
    for metric, role in all_metrics:
        rem = float(df.loc[df.recalled == 1, metric].mean())
        forg = float(df.loc[df.recalled == 0, metric].mean())
        log("\n" + "-" * 80)
        log(f"METRIC: {metric}   [{role}"
            + ("  == embedding_fidelity == raw_cosine]" if role == "MAIN" else "]"))
        log("-" * 80)
        log(f"11. remembered mean fidelity: {rem:.5f}")
        log(f"    forgotten  mean fidelity: {forg:.5f}   (rem - forg = {rem-forg:+.5f})")

        glmm = fit_glmm(df, metric)
        fb = fit_fallback(df, metric)
        for res in (glmm, fb):
            log(f"\n[{res['model']}]")
            if "error" in res:
                log(f"  FAILED: {res['error']}")
                rows.append({"metric": metric, "role": role, "model": res["model"],
                             "error": res["error"], "remembered_mean": rem, "forgotten_mean": forg})
                continue
            log(f"  7. coefficient (per 1 SD): {res['coef']:+.4f}")
            log(f"  8. odds ratio per 1 SD  : {res['odds_ratio']:.4f}")
            log(f"  9. {res['interval_kind']}: [{res['ci_low']:.4f}, {res['ci_high']:.4f}]")
            log(f"  10. approx p-value       : {res['p_approx']:.4f}"
                if np.isfinite(res.get('p_approx', np.nan)) else "  10. approx p-value: n/a")
            log(f"      direction vs forgotten: {verdict(res)}")
            rows.append({"metric": metric, "role": role, "model": res["model"],
                         "coef_per_sd": res["coef"], "odds_ratio": res["odds_ratio"],
                         "se": res["se"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                         "p_approx": res["p_approx"], "direction": verdict(res),
                         "remembered_mean": rem, "forgotten_mean": forg})
        meta_out[metric] = {"role": role, "remembered_mean": rem, "forgotten_mean": forg,
                            "glmm": glmm, "fallback": fb}

    # ---- FINAL CONCLUSION (main model = raw_cosine GLMM) ----
    main = meta_out[MAIN_METRIC]["glmm"]
    rem = meta_out[MAIN_METRIC]["remembered_mean"]
    forg = meta_out[MAIN_METRIC]["forgotten_mean"]
    direction = verdict(main) if "error" not in main else verdict(meta_out[MAIN_METRIC]["fallback"])
    concl_map = {
        "higher": "Remembered words had HIGHER EEG-to-AI embedding fidelity than forgotten words.",
        "lower": "Remembered words had LOWER EEG-to-AI embedding fidelity than forgotten words.",
        "no different": "Remembered and forgotten words showed NO DIFFERENT EEG-to-AI embedding "
                        "fidelity (no significant effect).",
    }
    log("\n" + "=" * 80)
    log("FINAL OUTLINE MODEL — CONCLUSION")
    log("=" * 80)
    log(f"model : {FORMULA}   (embedding_fidelity = raw_cosine, z-scored)")
    if "error" not in main:
        log(f"embedding_fidelity coefficient (per 1 SD): {main['coef']:+.4f}")
        log(f"odds ratio per 1 SD: {main['odds_ratio']:.4f}  "
            f"95% CrI [{main['ci_low']:.4f}, {main['ci_high']:.4f}]  "
            f"approx p={main['p_approx']:.4f}")
    log(f"remembered mean = {rem:.5f}   forgotten mean = {forg:.5f}   "
        f"diff = {rem-forg:+.5f}")
    log(f"\n12. CONCLUSION: {concl_map[direction]}")
    log("\nInterpretation caveat: word-specific decoding was at chance across all "
        "sessions (real ~= shuffled), and this dataset has only 4 subjects / 8 "
        "sessions. A null here is the expected, honest outcome; any non-null would "
        "warrant strong skepticism. This is the outline model, NOT a powered "
        "confirmatory test.")

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    json.dump({
        "kind": "FINAL outline memory model (multi-subject multi-session)",
        "formula": FORMULA,
        "embedding_fidelity_metric": MAIN_METRIC,
        "predictor_scaling": "z-scored (OR per 1 SD)",
        "session": "fixed effect, dummy-coded C(session)",
        "random_effects": "1|subject + 1|word",
        "supplementary_metrics": SUPP_METRICS,
        "n_subjects": n_sub, "n_sessions": n_ses, "n_trials": n_trials,
        "n_recalled": n_rec, "n_forgotten": n_forg,
        "decoding_was_at_chance": True,
        "final_conclusion": concl_map[direction],
        "results": meta_out,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, open(args.out_meta, "w"), indent=2, default=str)

    log(f"\nwrote {os.path.relpath(args.out_summary, HERE)}")
    log(f"wrote {os.path.relpath(args.out_csv, HERE)}")
    log(f"wrote {os.path.relpath(args.out_meta, HERE)}")
    log("STATUS: OK (final outline model complete)")
    fh.close()


if __name__ == "__main__":
    main()
