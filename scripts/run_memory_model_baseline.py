#!/usr/bin/env python3
"""
Baseline multi-subject memory model (5-subject BASELINE — NOT final inference).

Question: does trial-level EEG->T5 decoding fidelity predict whether a word was
later freely recalled?

    recalled ~ embedding_fidelity + (1|subject) + (1|word)

recalled is binary, so the primary fit is a LOGISTIC mixed-effects model
(statsmodels BinomialBayesMixedGLM, variational Bayes, crossed random
intercepts for subject and word). A fallback logistic regression with subject
fixed effects and cluster-robust SEs by subject is ALSO always run (note: with
only 5 clusters those SEs are unreliable and reported with caution).

The embedding_fidelity predictor is z-scored within the analysis, so the odds
ratio is "per 1 SD of the metric" and is comparable across metrics.

Metrics:
  PRIMARY  : centered_true_word_percentile
  SECONDARY: centered_cosine, true_word_percentile
  ARTIFACT : raw_cosine   (sanity check only — known inflated/common-direction)

IMPORTANT: the corrected decoding metrics were at chance (real ~= shuffled), so
any memory effect here must be interpreted with strong caution.
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

PRIMARY = "centered_true_word_percentile"
SECONDARY = ["centered_cosine", "true_word_percentile"]
ARTIFACT = ["raw_cosine"]
ALL_METRICS = [PRIMARY] + SECONDARY + ARTIFACT
FORMULA = "recalled ~ embedding_fidelity + (1|subject) + (1|word)"


def fit_glmm(df, metric):
    """Logistic GLMM (Bayesian VB) with crossed subject+word random intercepts.
    Returns dict or None if it fails."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = df.copy()
    d["zfid"] = (d[metric] - d[metric].mean()) / d[metric].std(ddof=0)
    d["subject"] = d["subject"].astype("category")
    d["word"] = d["word"].astype("category")
    try:
        vc = {"subject": "0 + C(subject)", "word": "0 + C(word)"}
        model = BinomialBayesMixedGLM.from_formula(
            "recalled ~ zfid", vc, d)
        res = model.fit_vb(verbose=False)
        names = list(res.model.exog_names)
        i = names.index("zfid")
        mean = float(res.fe_mean[i])
        sd = float(res.fe_sd[i])
        z = mean / sd if sd > 0 else np.nan
        # normal-approx two-sided p from the VB posterior (approximate)
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        return {
            "model": "logistic GLMM (BinomialBayesMixedGLM, VB; 1|subject + 1|word)",
            "coef": mean, "se": sd, "odds_ratio": float(np.exp(mean)),
            "ci_low": float(np.exp(mean - 1.96 * sd)),
            "ci_high": float(np.exp(mean + 1.96 * sd)),
            "z_approx": float(z), "p_approx": p,
            "interval_kind": "95% credible interval (VB posterior mean +/- 1.96 SD)",
        }
    except Exception as e:  # noqa: BLE001
        return {"model": "logistic GLMM", "error": f"{type(e).__name__}: {e}"}


def fit_fallback(df, metric):
    """Logistic regression with subject fixed effects + cluster-robust SE by subject."""
    import statsmodels.formula.api as smf
    d = df.copy()
    d["zfid"] = (d[metric] - d[metric].mean()) / d[metric].std(ddof=0)
    try:
        m = smf.logit("recalled ~ zfid + C(subject)", data=d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": d["subject"]})
        coef = float(m.params["zfid"])
        se = float(m.bse["zfid"])
        p = float(m.pvalues["zfid"])
        ci = m.conf_int().loc["zfid"]
        return {
            "model": "logistic + subject FE + cluster-robust SE (by subject)",
            "coef": coef, "se": se, "odds_ratio": float(np.exp(coef)),
            "ci_low": float(np.exp(ci[0])), "ci_high": float(np.exp(ci[1])),
            "z_approx": coef / se if se > 0 else np.nan, "p_approx": p,
            "interval_kind": "95% CI (cluster-robust; ONLY 5 clusters -> unreliable)",
        }
    except Exception as e:  # noqa: BLE001
        return {"model": "logistic + cluster-robust", "error": f"{type(e).__name__}: {e}"}


def verdict(res):
    if "error" in res:
        return "inconclusive (fit failed)"
    p = res.get("p_approx")
    lo, hi = res.get("ci_low"), res.get("ci_high")
    if p is not None and np.isfinite(p):
        if p < 0.05 and not (lo <= 1.0 <= hi):
            return "POSITIVE (higher fidelity -> more recall)" if res["coef"] > 0 \
                else "NEGATIVE (higher fidelity -> less recall)"
        return "no significant effect (inconclusive)"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=os.path.join(HERE, "outputs/all_subjects_fidelity_results.csv"))
    ap.add_argument("--out-summary", default=os.path.join(HERE, "outputs/memory_model_baseline_summary.txt"))
    ap.add_argument("--out-csv", default=os.path.join(HERE, "outputs/memory_model_baseline_results.csv"))
    ap.add_argument("--out-meta", default=os.path.join(HERE, "outputs/memory_model_baseline_metadata.json"))
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: input not found: {args.input}")
    df = pd.read_csv(args.input)

    n_subjects = df.subject.nunique()
    n_trials = len(df)
    n_rec = int((df.recalled == 1).sum())
    n_forg = int((df.recalled == 0).sum())

    fh = open(args.out_summary, "w")

    def log(*p):
        line = " ".join(str(x) for x in p)
        print(line)
        fh.write(line + "\n")

    log("=" * 78)
    log("BASELINE MULTI-SUBJECT MEMORY MODEL — 5-SUBJECT BASELINE (NOT final inference)")
    log("=" * 78)
    log(f"input            : {os.path.relpath(args.input, HERE)}")
    log(f"number of subjects: {n_subjects}")
    log(f"number of trials  : {n_trials}")
    log(f"recalled/forgotten: {n_rec} / {n_forg}  (recall rate {n_rec/n_trials:.3f})")
    log(f"model formula     : {FORMULA}")
    log("predictor is z-scored -> odds ratio is PER 1 SD of the metric.")
    log("session NOT included (one session per subject).")
    log("\n!!! CAVEAT: corrected decoding metrics were AT CHANCE (real ~= shuffled). "
        "Any memory effect below must be interpreted with strong caution. !!!")

    rows = []
    meta_out = {}
    for metric in ALL_METRICS:
        role = ("PRIMARY" if metric == PRIMARY
                else "ARTIFACT/sanity" if metric in ARTIFACT else "SECONDARY")
        rem = float(df.loc[df.recalled == 1, metric].mean())
        forg = float(df.loc[df.recalled == 0, metric].mean())

        log("\n" + "-" * 78)
        log(f"METRIC: {metric}   [{role}]")
        log("-" * 78)
        log(f"remembered mean fidelity: {rem:.5f}")
        log(f"forgotten  mean fidelity: {forg:.5f}")
        log(f"remembered - forgotten  : {rem - forg:+.5f}")

        glmm = fit_glmm(df, metric)
        fb = fit_fallback(df, metric)

        for res in (glmm, fb):
            log(f"\n[{res['model']}]")
            if "error" in res:
                log(f"  FAILED: {res['error']}")
                rows.append({"metric": metric, "role": role, "model": res["model"],
                             "error": res["error"], "remembered_mean": rem,
                             "forgotten_mean": forg})
                continue
            log(f"  coefficient (per 1 SD): {res['coef']:+.4f}")
            log(f"  odds ratio            : {res['odds_ratio']:.4f}")
            log(f"  {res['interval_kind']}: [{res['ci_low']:.4f}, {res['ci_high']:.4f}]")
            log(f"  approx p-value        : {res['p_approx']:.4f}"
                if np.isfinite(res.get("p_approx", np.nan)) else "  approx p-value: n/a")
            log(f"  verdict               : {verdict(res)}")
            rows.append({"metric": metric, "role": role, "model": res["model"],
                         "coef_per_sd": res["coef"], "odds_ratio": res["odds_ratio"],
                         "se": res["se"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                         "p_approx": res["p_approx"], "verdict": verdict(res),
                         "remembered_mean": rem, "forgotten_mean": forg})
        meta_out[metric] = {"role": role, "remembered_mean": rem, "forgotten_mean": forg,
                            "glmm": glmm, "fallback": fb}

    # ---- headline (primary metric, GLMM) ----
    log("\n" + "=" * 78)
    log("HEADLINE — PRIMARY metric (centered_true_word_percentile), logistic GLMM")
    log("=" * 78)
    pg = meta_out[PRIMARY]["glmm"]
    if "error" not in pg:
        log(f"embedding_fidelity coefficient (per 1 SD): {pg['coef']:+.4f}")
        log(f"odds ratio: {pg['odds_ratio']:.4f}  "
            f"95% CrI [{pg['ci_low']:.4f}, {pg['ci_high']:.4f}]  "
            f"approx p={pg['p_approx']:.4f}")
        log(f"RESULT: {verdict(pg).upper()}")
    else:
        log(f"GLMM failed ({pg['error']}); rely on fallback.")
        pf = meta_out[PRIMARY]["fallback"]
        log(f"fallback OR={pf.get('odds_ratio')}, p={pf.get('p_approx')}, "
            f"verdict {verdict(pf)}")

    log("\nINTERPRETATION NOTE: decoding was at chance across all 5 subjects "
        "(real ~= shuffled), so a null memory effect here is the EXPECTED and "
        "honest outcome. A non-null effect would more likely reflect residual "
        "confounds than genuine memory-related decoding, and would NOT be "
        "trustworthy at n=5 subjects. This is a BASELINE, not final inference.")

    # ---- save ----
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    json.dump({
        "kind": "5-subject BASELINE memory model (NOT final inference)",
        "formula": FORMULA,
        "predictor_scaling": "z-scored within analysis (OR per 1 SD)",
        "primary_metric": PRIMARY, "secondary_metrics": SECONDARY,
        "artifact_metrics": ARTIFACT,
        "n_subjects": n_subjects, "n_trials": n_trials,
        "n_recalled": n_rec, "n_forgotten": n_forg,
        "session_included": False,
        "decoding_was_at_chance": True,
        "results": meta_out,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }, open(args.out_meta, "w"), indent=2, default=str)

    log(f"\nwrote {os.path.relpath(args.out_summary, HERE)}")
    log(f"wrote {os.path.relpath(args.out_csv, HERE)}")
    log(f"wrote {os.path.relpath(args.out_meta, HERE)}")
    log("STATUS: OK (baseline analysis complete; NOT final inference)")
    fh.close()


if __name__ == "__main__":
    main()
