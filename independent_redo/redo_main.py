#!/usr/bin/env python3
"""
Clean-room independent redo of the Neurolab project, following ONLY the outline.

Key literal reading of the outline: "For each subject separately: train ridge
regression" -> ONE ridge model PER SUBJECT, pooling that subject's trials across
sessions, with held-out-trial cross-validation. (The current project instead
trained per subject/session; that is the main methodological difference tested
here.)

Self-contained: uses only the redo T5 embeddings (from peers_words.csv), the
source events.tsv/EDF for the same 4 subjects x 2 sessions as the current final
analysis (needed for an apples-to-apples comparison; the outline does not name
subjects), numpy/pandas/mne/sklearn/statsmodels. Does not import the existing
pipeline. Writes only into independent_redo/.

Steps (outline): recall labels from WORD/REC_WORD -> EEG 300-800 ms window
-> per-subject ridge (alpha=10000) held-out CV -> cosine fidelity -> logistic
mixed model recalled ~ embedding_fidelity + session + (1|subject) + (1|word).
"""
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(HERE, "independent_redo")
ROUT = os.path.join(RD, "outputs")
RCMP = os.path.join(RD, "comparison")
EMB = os.path.join(ROUT, "peers_t5large_embeddings.npy")
ORDER = os.path.join(ROUT, "peers_word_order.csv")
ALPHA, FOLDS, SEED = 10000.0, 5, 42
S0, S1 = 150, 400   # int(0.300*500), int(0.800*500); stop exclusive -> 250 tp

# same 4 subjects x 2 sessions as the current final analysis (for comparison)
SESSIONS = [("LTP269", 12), ("LTP269", 20), ("LTP293", 5), ("LTP293", 22),
            ("LTP299", 2), ("LTP299", 6), ("LTP303", 10), ("LTP303", 22)]

log_lines = []
def log(*p):
    s = " ".join(str(x) for x in p); print(s, flush=True); log_lines.append(s)


def build_session_trials(sub, ses, word2row):
    """Recall labels + EEG window per WORD trial for one session (outline logic)."""
    base = os.path.join(HERE, "data/ds004395", f"sub-{sub}", f"ses-{ses}", "eeg",
                        f"sub-{sub}_ses-{ses}_task-ltpFR2")
    ev = pd.read_csv(base + "_events.tsv", sep="\t", na_values=["n/a", ""])
    w = ev[ev.trial_type == "WORD"].sort_values(["trial", "onset"]).copy()
    w["serialpos"] = w.groupby("trial").cumcount() + 1
    rec = ev[ev.trial_type == "REC_WORD"].dropna(subset=["trial", "item_num"])
    rec = rec[rec.item_num != -1]
    keys = set(zip(rec.trial.astype(int), rec.item_num.astype(int)))
    rows = []
    for _, r in w.iterrows():
        t, inum = int(r.trial), int(r.item_num)
        rows.append({"subject": sub, "session": ses, "trial": t,
                     "serialpos": int(r.serialpos), "word": str(r.item_name).upper(),
                     "item_num": inum, "sample": int(r["sample"]),
                     "recalled": 1 if (t, inum) in keys else 0})
    df = pd.DataFrame(rows)
    assert len(df) == 576, f"{sub}/{ses}: {len(df)} WORD trials != 576"
    assert df.word.isin(word2row).all(), f"{sub}/{ses}: words missing from peers list"
    assert (df["sample"] >= 0).all() and (df.item_num >= 0).all()
    return df


def extract_X(sub, ses, samples):
    import mne
    edf = os.path.join(HERE, "data/ds004395", f"sub-{sub}", f"ses-{ses}", "eeg",
                       f"sub-{sub}_ses-{ses}_task-ltpFR2_eeg.edf")
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks)
    nt = data.shape[1]
    assert sfreq == 500.0 and len(picks) == 129
    X = np.zeros((len(samples), 129 * 250), dtype=np.float32)
    for i, smp in enumerate(samples):
        a, b = smp + S0, smp + S1
        assert 0 <= a and b <= nt, f"{sub}/{ses} window out of bounds"
        X[i] = data[:, a:b].reshape(-1).astype(np.float32)
    return X


def per_subject_ridge(X, Y):
    """5-fold held-out-trial CV, StandardScaler on train only, Ridge svd."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    n = X.shape[0]
    pred = np.zeros_like(Y)
    for tr, te in KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=ALPHA, solver="svd").fit(sc.transform(X[tr]), Y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


def row_cos(A, B):
    an = np.linalg.norm(A, axis=1); bn = np.linalg.norm(B, axis=1)
    return np.sum(A * B, axis=1) / (an * bn)


def fit_glmm(df):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from scipy.stats import norm
    d = df.copy()
    d["zfid"] = (d.embedding_fidelity - d.embedding_fidelity.mean()) / d.embedding_fidelity.std(ddof=0)
    d["session"] = d.session.astype("category")
    res = BinomialBayesMixedGLM.from_formula(
        "recalled ~ zfid + C(session)",
        {"subject": "0 + C(subject)", "word": "0 + C(word)"}, d).fit_vb(verbose=False)
    i = list(res.model.exog_names).index("zfid")
    mean, sd = float(res.fe_mean[i]), float(res.fe_sd[i])
    z = mean / sd
    return {"coef": mean, "odds_ratio": float(np.exp(mean)),
            "ci_low": float(np.exp(mean - 1.96 * sd)), "ci_high": float(np.exp(mean + 1.96 * sd)),
            "p": float(2 * (1 - norm.cdf(abs(z))))}


def main():
    order = pd.read_csv(ORDER)
    emb = np.load(EMB).astype(np.float64)
    word2row = dict(zip(order.word.str.upper(), order.row_index.astype(int)))
    log("=" * 78)
    log("INDEPENDENT REDO (literal outline; PER-SUBJECT ridge)")
    log("=" * 78)
    log(f"embeddings {emb.shape}; {len(SESSIONS)} sessions, "
        f"{len(set(s for s,_ in SESSIONS))} subjects")

    # build trials for all sessions
    trials = {s: build_session_trials(*s, word2row) for s in SESSIONS}
    log("recall labels + windows built for all sessions (576 WORD each).")

    # per-subject pooled ridge
    frames = []
    for sub in sorted(set(s for s, _ in SESSIONS)):
        subs_ses = [(a, b) for (a, b) in SESSIONS if a == sub]
        parts = []
        for a, b in subs_ses:
            df = trials[(a, b)]
            X = extract_X(a, b, df["sample"].tolist())
            Y = emb[[word2row[w] for w in df.word]]
            parts.append((df.reset_index(drop=True), X, Y))
        meta = pd.concat([p[0] for p in parts], ignore_index=True)
        X = np.vstack([p[1] for p in parts]).astype(np.float64)
        Y = np.vstack([p[2] for p in parts])
        pred = per_subject_ridge(X, Y)
        meta["embedding_fidelity"] = row_cos(pred, Y)
        frames.append(meta)
        log(f"  {sub}: pooled {len(meta)} trials ({len(subs_ses)} sessions) -> "
            f"mean fidelity {meta.embedding_fidelity.mean():.5f}")

    comb = pd.concat(frames, ignore_index=True)
    comb = comb[["subject", "session", "trial", "serialpos", "word", "recalled",
                 "embedding_fidelity"]]
    comb.to_csv(os.path.join(ROUT, "all_sessions_fidelity_results.csv"), index=False)
    log(f"\ncombined fidelity table: {len(comb)} rows, "
        f"recalled {int((comb.recalled==1).sum())} / forgotten {int((comb.recalled==0).sum())}")

    # final model
    fm = fit_glmm(comb)
    rem = comb.loc[comb.recalled == 1, "embedding_fidelity"].mean()
    forg = comb.loc[comb.recalled == 0, "embedding_fidelity"].mean()
    concl = "no different" if (fm["ci_low"] <= 1.0 <= fm["ci_high"]) else \
            ("higher" if fm["coef"] > 0 else "lower")
    log("\n--- REDO FINAL MODEL (recalled ~ embedding_fidelity + session + (1|subject)+(1|word)) ---")
    log(f"coefficient (per SD): {fm['coef']:+.4f}   odds ratio: {fm['odds_ratio']:.4f}")
    log(f"95% CrI: [{fm['ci_low']:.4f}, {fm['ci_high']:.4f}]   p~{fm['p']:.4f}")
    log(f"remembered mean {rem:.5f}   forgotten mean {forg:.5f}   diff {rem-forg:+.5f}")
    log(f"conclusion: {concl}")
    json.dump({"design": "per-SUBJECT ridge (pooled sessions), literal outline",
               "alpha": ALPHA, "folds": FOLDS, "seed": SEED,
               "coefficient": fm["coef"], "odds_ratio": fm["odds_ratio"],
               "ci": [fm["ci_low"], fm["ci_high"]], "p": fm["p"],
               "remembered_mean": rem, "forgotten_mean": forg, "difference": rem - forg,
               "conclusion": concl, "n_trials": len(comb),
               "timestamp_utc": datetime.now(timezone.utc).isoformat()},
              open(os.path.join(ROUT, "final_memory_model_metadata.json"), "w"), indent=2)

    # ================= comparison vs current =================
    cur = pd.read_csv(os.path.join(HERE, "outputs/all_sessions_fidelity_results.csv"))
    curfm = pd.read_csv(os.path.join(HERE, "outputs/final_memory_model_results.csv"))
    cm = curfm[(curfm.metric == "raw_cosine") & (curfm.model.str.contains("GLMM"))].iloc[0]
    key = ["subject", "session", "trial", "serialpos"]
    m = comb.merge(cur[key + ["word", "recalled", "embedding_fidelity"]], on=key,
                   suffixes=("_redo", "_cur"))
    word_ok = (m.word_redo.str.upper() == m.word_cur.str.upper()).all()
    rec_ok = (m.recalled_redo == m.recalled_cur).all()
    d = np.abs(m.embedding_fidelity_redo - m.embedding_fidelity_cur)
    ps_redo = comb.groupby(["subject", "session"]).embedding_fidelity.mean()
    ps_cur = cur.groupby(["subject", "session"]).embedding_fidelity.mean()

    clines = ["INDEPENDENT REDO vs CURRENT PROJECT", "=" * 60,
              "DESIGN DIFFERENCE: redo trains ridge PER SUBJECT (pooled sessions);",
              "current trains PER SUBJECT/SESSION. Everything else follows the outline.",
              "",
              f"rows aligned            : {len(m)} (redo {len(comb)}, current {len(cur)})",
              f"word column matches     : {word_ok}",
              f"recalled labels match   : {rec_ok}",
              "",
              "embedding_fidelity (redo per-subject) vs current (per-session):",
              f"  max abs diff  : {d.max():.4e}",
              f"  mean abs diff : {d.mean():.4e}",
              f"  rows > 0.01   : {int((d>0.01).sum())} / {len(d)}",
              "",
              "per-session mean embedding_fidelity (redo vs current):"]
    for k in ps_redo.index:
        clines.append(f"  {k[0]}/{k[1]}: redo {ps_redo[k]:.5f}  current {ps_cur[k]:.5f}  "
                      f"diff {ps_redo[k]-ps_cur[k]:+.2e}")
    clines += ["",
               "remembered vs forgotten mean fidelity:",
               f"  redo    : remembered {rem:.5f}  forgotten {forg:.5f}  diff {rem-forg:+.5f}",
               f"  current : remembered {cur.loc[cur.recalled==1,'embedding_fidelity'].mean():.5f}  "
               f"forgotten {cur.loc[cur.recalled==0,'embedding_fidelity'].mean():.5f}  "
               f"diff {cur.loc[cur.recalled==1,'embedding_fidelity'].mean()-cur.loc[cur.recalled==0,'embedding_fidelity'].mean():+.5f}",
               "",
               "FINAL MODEL (main metric = raw cosine / embedding_fidelity):",
               f"  redo    : coef {fm['coef']:+.4f}  OR {fm['odds_ratio']:.4f}  "
               f"CrI [{fm['ci_low']:.4f},{fm['ci_high']:.4f}]  p {fm['p']:.4f}  -> {concl}",
               f"  current : coef {float(cm.coef_per_sd):+.4f}  OR {float(cm.odds_ratio):.4f}  "
               f"CI [{float(cm.ci_low):.4f},{float(cm.ci_high):.4f}]  p {float(cm.p_approx):.4f}  "
               f"-> {cm.direction}",
               ""]
    or_diff = abs(fm["odds_ratio"] - float(cm.odds_ratio))
    concl_same = (concl == str(cm.direction))
    major = []
    if not word_ok:
        major.append("word labels differ")
    if not rec_ok:
        major.append("recall labels differ")
    if not concl_same:
        major.append("FINAL CONCLUSION CHANGED")
    if or_diff > 0.05:
        major.append(f"odds ratio changed by {or_diff:.3f} (>0.05)")
    clines += ["MAJOR DIFFERENCES: " + (", ".join(major) if major else
               "none that change the conclusion"),
               f"conclusion redo == current: {concl_same}",
               f"odds-ratio abs diff: {or_diff:.4f}",
               "",
               "VERDICT: " + ("SAME CONCLUSION — both null (no significant fidelity->recall effect)."
                              if concl_same and not major else
                              "DIFFERENCE AFFECTS CONCLUSION — investigate.")]
    open(os.path.join(RCMP, "redo_vs_current_comparison.txt"), "w").write("\n".join(clines) + "\n")
    pd.DataFrame({"subject_session": [f"{k[0]}/{k[1]}" for k in ps_redo.index],
                  "redo_mean_fidelity": ps_redo.values,
                  "current_mean_fidelity": [ps_cur[k] for k in ps_redo.index]}).to_csv(
        os.path.join(RCMP, "redo_vs_current_per_session.csv"), index=False)
    log("\n" + "\n".join(clines[-6:]))

    open(os.path.join(RD, "logs", "redo_run.txt"), "w").write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
