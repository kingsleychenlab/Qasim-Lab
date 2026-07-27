#!/usr/bin/env python3
"""
The memory model, the last stage of the project.

    recalled ~ embedding_fidelity + session + (1|subject) + (1|word)

This is the question the whole project is built around. Ridge already turned
each trial's EEG into a predicted T5 embedding, and fidelity measures how close
that prediction landed to the real one. Here I ask whether higher fidelity makes
a word more likely to be recalled later.

A few modeling choices, most of them fixed by the plan. recalled is yes/no, so
the model is logistic rather than linear. embedding_fidelity is raw_cosine, the
metric the plan names. The predictor is z-scored, so the odds ratio reads as
"per 1 SD of fidelity" and stays comparable across metrics. session is a fixed
effect because there are only two sessions per subject, too few to estimate a
variance from. subject and word are crossed random intercepts (see fit_glmm).

Each metric gets two fits. fit_glmm is the one I report; fit_fallback makes
different assumptions and should agree with it. Three supplementary metrics run
the same model to check the conclusion doesn't hinge on using raw cosine.

One caveat that the script prints: word-specific decoding came out at chance
(real about equal to the shuffled-label control), so there's no fidelity signal
for a memory effect to sit on top of. A null is what you'd expect here, and a
non-null would be more reason for suspicion than excitement.

Usage:
    python code/step11_run_memory_model.py \
        --input outputs/all_sessions32_fidelity_results.csv \
        --out-summary outputs/final_memory_model32_summary.txt

Outputs:
  outputs/final_memory_model_summary.txt    human-readable report
  outputs/final_memory_model_results.csv    one row per (metric, model)
  outputs/final_memory_model_metadata.json  full provenance
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import Tee

# Silence only the library deprecation chatter. Convergence warnings stay on: a
# VB fit that didn't converge would hand back a confident-looking odds ratio with
# nothing behind it, and the whole result depends on these fits converging.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# raw_cosine is the plan's embedding_fidelity, so it stays the main metric even
# though we know it's inflated by the common embedding direction (the centered_*
# supplementaries strip that direction out). Keeping the preregistered metric as
# primary keeps the analysis honest, and the supplementaries show the conclusion
# doesn't depend on that choice.
MAIN_METRIC = "raw_cosine"
SUPP_METRICS = ["centered_cosine", "true_word_percentile", "centered_true_word_percentile"]
FORMULA = "recalled ~ embedding_fidelity + session + (1|subject) + (1|word)"


def zscore(s):
    # ddof=0: this rescales the observed sample, it isn't an estimate of a
    # population SD, so the coefficient means "per 1 SD of the fidelity we saw".
    return (s - s.mean()) / s.std(ddof=0)


def fit_glmm(df, metric):
    """The main model. Logistic GLMM with crossed random intercepts.

    recalled ~ zfid + C(session), with (1|subject) + (1|word).

    The random effects are crossed, not nested: every subject sees essentially
    every word, so subject and word are two independent grouping factors. The
    word intercept is what keeps a handful of intrinsically memorable words from
    passing themselves off as a fidelity effect.

    Fit by variational Bayes. statsmodels has no Laplace or MCMC path for crossed
    binomial GLMMs at this scale (36k trials by ~600 word levels), and VB is the
    only estimator here that converges in a reasonable time. The catch is that VB
    tends to understate posterior variance, so if anything the interval below is
    too narrow, which only makes a null conclusion more conservative.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from scipy.stats import norm
    d = df.copy()
    d["zfid"] = zscore(d[metric])
    d["session"] = d["session"].astype("category")
    d["subject"] = d["subject"].astype("category")
    d["word"] = d["word"].astype("category")
    try:
        # "0 + C(...)" gives one variance component per level with no extra
        # intercept, which is just a plain random intercept per grouping factor.
        vc = {"subject": "0 + C(subject)", "word": "0 + C(word)"}
        res = BinomialBayesMixedGLM.from_formula(
            "recalled ~ zfid + C(session)", vc, d).fit_vb(verbose=False)
        i = list(res.model.exog_names).index("zfid")
        mean, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
        # VB yields a posterior, not a sampling distribution, so there is no
        # exact p-value. Treat the posterior as approximately normal and read a
        # Wald-style two-sided tail off it. Reported as "approx" throughout.
        z = mean / sd if sd > 0 else np.nan
        p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        return {"model": "logistic GLMM (BinomialBayesMixedGLM VB; +C(session); 1|subject+1|word)",
                "coef": mean, "se": sd, "odds_ratio": float(np.exp(mean)),
                "ci_low": float(np.exp(mean - 1.96 * sd)),
                "ci_high": float(np.exp(mean + 1.96 * sd)),
                "z_approx": z, "p_approx": p,
                "interval_kind": "95% credible interval (VB posterior mean +/- 1.96 SD)"}
    except Exception as e:  # noqa: BLE001
        # Report the failure in the summary rather than aborting: a failed
        # supplementary fit should not lose the other metrics' results.
        return {"model": "logistic GLMM", "error": f"{type(e).__name__}: {e}"}


def fit_fallback(df, metric):
    """A cross-check on the GLMM, not a result in its own right.

    This swaps the random intercepts for subject/session fixed effects and uses
    cluster-robust SEs to soak up within-subject correlation. It's worth running
    because it assumes something completely different from VB, so if the two
    disagree badly the GLMM is suspect.

    The SEs are only trustworthy when there are a lot of clusters, so the caller
    reports the actual count and lets the reader judge. This model can't absorb
    word-level variance at all, which is the main reason it's the secondary one.
    """
    import statsmodels.formula.api as smf
    d = df.copy()
    d["zfid"] = zscore(d[metric])
    n_clusters = int(d["subject"].nunique())
    # Cluster-robust asymptotics lean on the number of clusters, not the number
    # of rows. Rule of thumb is ~40; below that the SEs are biased downward.
    reliability = ("very unreliable" if n_clusters < 15
                   else "treat with caution" if n_clusters < 40
                   else "adequate")
    try:
        m = smf.logit("recalled ~ zfid + C(session) + C(subject)", data=d).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": d["subject"]})
        coef, se, p = float(m.params["zfid"]), float(m.bse["zfid"]), float(m.pvalues["zfid"])
        ci = m.conf_int().loc["zfid"]
        return {"model": "logistic + C(session)+C(subject) FE + cluster-robust SE (by subject)",
                "coef": coef, "se": se, "odds_ratio": float(np.exp(coef)),
                "ci_low": float(np.exp(ci[0])), "ci_high": float(np.exp(ci[1])),
                "z_approx": coef / se if se > 0 else np.nan, "p_approx": p,
                "n_clusters": n_clusters,
                "interval_kind": f"95% CI (cluster-robust; {n_clusters} clusters "
                                 f"-> {reliability})"}
    except Exception as e:  # noqa: BLE001
        return {"model": "logistic + cluster-robust", "error": f"{type(e).__name__}: {e}"}


def verdict(res):
    """Only call an effect when the p-value and the interval agree.

    This is stricter than p < 0.05 on its own, since the interval also has to
    exclude an odds ratio of 1.0. With a VB posterior the two can disagree near
    the margin, and on a preregistered null a borderline p isn't worth claiming.
    """
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
    # The upstream table stores fidelity twice, once under the plan's name and
    # once under the metric it actually is. If those two ever diverge, the column
    # names stop meaning what the summary says they mean, so bail out instead of
    # quietly modeling the wrong column.
    if "embedding_fidelity" in df.columns and not np.allclose(
            df["embedding_fidelity"], df["raw_cosine"]):
        sys.exit("ERROR: embedding_fidelity != raw_cosine in input.")

    n_sub = df.subject.nunique()
    n_ses = df.groupby(["subject", "session"]).ngroups
    n_trials = len(df)
    n_rec = int((df.recalled == 1).sum())
    n_forg = int((df.recalled == 0).sum())

    fh = open(args.out_summary, "w")
    log = Tee(fh)

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
    log("\nCaution: corrected decoding came out at chance (real about equal to "
        "shuffled word retrieval), so read any effect below with that in mind.")

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

    # FINAL CONCLUSION (main model = raw_cosine GLMM)
    main = meta_out[MAIN_METRIC]["glmm"]
    rem = meta_out[MAIN_METRIC]["remembered_mean"]
    forg = meta_out[MAIN_METRIC]["forgotten_mean"]
    direction = verdict(main) if "error" not in main else verdict(meta_out[MAIN_METRIC]["fallback"])
    concl_map = {
        "higher": "Remembered words had higher EEG-to-AI embedding fidelity than forgotten words.",
        "lower": "Remembered words had lower EEG-to-AI embedding fidelity than forgotten words.",
        "no different": "Remembered and forgotten words showed no difference in EEG-to-AI "
                        "embedding fidelity (no significant effect).",
    }
    conclusion = concl_map.get(direction, "Model fit was inconclusive (both fits failed).")
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
    log(f"\n12. CONCLUSION: {conclusion}")
    # Match the caveat to the run being analyzed. People read this straight out of
    # the summary file, so a hardcoded sample size here would contradict the header
    # above it.
    log(f"\nOne caveat: word-specific decoding sat at chance across all sessions "
        f"(real about equal to shuffled), and this run covers {n_sub} subjects / "
        f"{n_ses} sessions. A null is the honest outcome to expect here; a non-null "
        f"would be worth real skepticism. This is the plan's model, not a powered "
        f"confirmatory test.")

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
        "final_conclusion": conclusion,
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
