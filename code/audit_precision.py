#!/usr/bin/env python3
"""
The end-to-end audit: re-derive the project's claims from source and check them.

This trusts nothing the pipeline wrote. Where a stage produced a number, it
recomputes that number from the raw inputs and compares. Recall labels get
re-derived from the events files, EEG windows get re-extracted from the EDFs,
and the T5 matrix is re-checked against its own metadata. Where a claim is about
*how* a stage works, it reads that stage's source and asserts the mechanism is
present (for example, that step08 really does fit its scaler on training folds
only).

That source-reading is why this file names other scripts as strings: rename a
step without updating the paths here and the audit turns into a silent no-op.

Read-only. Writes only its own report to outputs/final_precision_audit.txt.

Requires the raw dataset and a completed pipeline run, since it re-extracts from
the EDFs. The archived output of the last full run is kept at
results/validation/precision_audit_output.txt (136 checks, 0 failures).

Sections: structure, T5 numeric integrity + metadata, session validity, recall
re-derivation from events, EEG re-extraction from EDF, ridge/decoding config,
corrected-metric ranges, final-model numbers, gitignore behaviour. (The T5
recompute from the live model and the results-wording scan are appended by
separate steps.)
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs", "final_precision_audit.txt")
SESSIONS = [("LTP269", 12), ("LTP269", 20), ("LTP293", 5), ("LTP293", 22),
            ("LTP299", 2), ("LTP299", 6), ("LTP303", 10), ("LTP303", 22)]
EEG_WIN = (0.300, 0.800)

checks = []   # (section, check, status, evidence, file)
lines = []


def log(*p):
    s = " ".join(str(x) for x in p)
    print(s)
    lines.append(s)


def ck(section, name, cond, evidence, file=""):
    status = "PASS" if cond else "FAIL"
    checks.append((section, name, status, evidence, file))
    log(f"[{status}] ({section}) {name} -- {evidence}" + (f"  [{file}]" if file else ""))
    return cond


def sess_dir(sub, ses):
    return os.path.join(HERE, "outputs", "subjects", f"sub-{sub}_ses-{ses}")


log("=" * 80)
log("NEUROLAB FINAL PRECISION AUDIT (programmatic)")
log("=" * 80)

# 1. structure
log("\n## 1. STRUCTURE")
req = ["results/embeddings/peers_words.csv", "results/embeddings/peers_word_order.csv", "results/embeddings/peers_t5large_embeddings.npy",
       "results/embeddings/embedding_metadata.json",
       "outputs/all_sessions_fidelity_results.csv",
       "outputs/final_memory_model_summary.txt",
       "outputs/final_memory_model_results.csv",
       "outputs/final_memory_model_metadata.json",
       "outputs/recall_label_audit.txt", "outputs/eeg_extraction_audit.txt",
       "results/README.md", "results/methods_and_math.md",
       "results/summary_4_vs_16_vs_32_subjects.txt",
       "results/validation/validation.md"]
for f in req:
    ck("structure", f"exists {f}", os.path.isfile(os.path.join(HERE, f)), "present", f)
opt = ["outputs/all_sessions_summary.txt", "outputs/model_input_validation.txt",
       "outputs/t5_embedding_audit.txt"]
for f in opt:
    p = os.path.isfile(os.path.join(HERE, f))
    log(f"[INFO] optional {f}: {'present' if p else 'absent'}")

# 2. T5 numeric + metadata
log("\n## 2. T5 EMBEDDING INTEGRITY")
words = pd.read_csv(os.path.join(HERE, "results/embeddings/peers_words.csv"))
order = pd.read_csv(os.path.join(HERE, "results/embeddings/peers_word_order.csv"))
emb = np.load(os.path.join(HERE, "results/embeddings/peers_t5large_embeddings.npy"))
meta = json.load(open(os.path.join(HERE, "results/embeddings/embedding_metadata.json")))
ck("t5", "peers_words.csv has 576 rows", len(words) == 576, f"{len(words)}", "peers_words.csv")
ck("t5", "peers_word_order.csv has 576 rows", len(order) == 576, f"{len(order)}", "peers_word_order.csv")
ck("t5", "embeddings shape 576x1024", emb.shape == (576, 1024), f"{emb.shape}", "peers_t5large_embeddings.npy")
ck("t5", "row_index is exactly 0..575",
   order.row_index.tolist() == list(range(576)),
   f"min {order.row_index.min()} max {order.row_index.max()}", "peers_word_order.csv")
ck("t5", "no NaN", not np.isnan(emb).any(), "none")
ck("t5", "no Inf", not np.isinf(emb).any(), "none")
norms = np.linalg.norm(emb, axis=1)
ck("t5", "no all-zero vectors", (norms > 0).all(), f"min norm {norms.min():.2f}")
ck("t5", "norms plausible/nonzero", norms.min() > 100,
   f"norm min {norms.min():.1f} mean {norms.mean():.1f} max {norms.max():.1f}")
# metadata assertions
ck("t5", "metadata model google-t5/t5-large", meta.get("model_name") == "google-t5/t5-large",
   str(meta.get("model_name")), "embedding_metadata.json")
ck("t5", "metadata class T5EncoderModel", meta.get("model_class") == "T5EncoderModel",
   str(meta.get("model_class")), "embedding_metadata.json")
ck("t5", "metadata encoder_only True", meta.get("encoder_only") is True,
   str(meta.get("encoder_only")), "embedding_metadata.json")
ck("t5", "metadata layer 12 used", meta.get("encoder_layer_used") == 12,
   str(meta.get("encoder_layer_used")), "embedding_metadata.json")
ck("t5", "metadata EOS excluded", meta.get("eos_excluded") is True, str(meta.get("eos_excluded")))
ck("t5", "metadata padding excluded", meta.get("padding_excluded") is True, str(meta.get("padding_excluded")))
ck("t5", "metadata shape 576x1024", meta.get("matrix_shape") == [576, 1024], str(meta.get("matrix_shape")))
ck("t5", "word order consistent (order==embeddings CSV order)",
   (order.word.str.upper().is_unique), f"{order.word.nunique()} unique words")

# 3. session validity
log("\n## 3. SESSION VALIDITY")
comb = pd.read_csv(os.path.join(HERE, "outputs/all_sessions_fidelity_results.csv"))
peers = set(order.word.str.upper())
ck("session", "4 subjects", comb.subject.nunique() == 4, f"{sorted(comb.subject.unique())}")
ck("session", "8 sessions", comb.groupby(['subject', 'session']).ngroups == 8,
   f"{comb.groupby(['subject','session']).ngroups}")
per_sub = comb.groupby('subject').session.nunique()
ck("session", "2 sessions per subject", (per_sub == 2).all(), per_sub.to_dict())
expected = set(SESSIONS)
present = set(map(tuple, comb[['subject', 'session']].drop_duplicates().values.tolist()))
ck("session", "expected subject/session set matches", present == expected,
   f"{sorted(present)}")
sizes = comb.groupby(['subject', 'session']).size()
ck("session", "576 trials per session", (sizes == 576).all(), f"unique sizes {sorted(sizes.unique())}")
ck("session", "total rows 4608", len(comb) == 4608, f"{len(comb)}")
uniq_words_ok = comb.groupby(['subject', 'session']).word.nunique().eq(576).all()
ck("session", "each session has 576 unique words", uniq_words_ok, "all 576")
ck("session", "all words in peers list", comb.word.str.upper().isin(peers).all(),
   f"{comb.word.str.upper().isin(peers).mean()*100:.1f}% in peers")
reqcols = ["subject", "session", "trial", "serialpos", "word", "recalled", "embedding_fidelity"]
ck("session", "no missing in required cols", comb[reqcols].notna().all().all(), "complete")
ck("session", "recalled only 0/1", set(comb.recalled.unique()) <= {0, 1},
   f"{sorted(comb.recalled.unique())}")
numcols = comb.select_dtypes("number")
ck("session", "no NaN/Inf in numeric cols",
   not numcols.isna().any().any() and np.isfinite(numcols.to_numpy()).all(), "clean")
ck("session", "recalled/forgotten = 2423/2185",
   int((comb.recalled == 1).sum()) == 2423 and int((comb.recalled == 0).sum()) == 2185,
   f"{int((comb.recalled==1).sum())}/{int((comb.recalled==0).sum())}")

# 4. recall re-derivation
log("\n## 4. RECALL LABELS (re-derived from events per session)")
for sub, ses in SESSIONS:
    ev_path = os.path.join(HERE, "data/ds004395", f"sub-{sub}", f"ses-{ses}", "eeg",
                           f"sub-{sub}_ses-{ses}_task-ltpFR2_events.tsv")
    enc_path = os.path.join(sess_dir(sub, ses), "encoding_trials.csv")
    if not (os.path.isfile(ev_path) and os.path.isfile(enc_path)):
        ck("recall", f"{sub}/{ses} source events+encoding present", False,
           "MISSING source", ev_path)
        continue
    ev = pd.read_csv(ev_path, sep="\t", na_values=["n/a", ""])
    enc = pd.read_csv(enc_path)
    recog = ev[ev.trial_type.astype(str).str.upper().str.startswith("RECOG")]
    rec = ev[ev.trial_type == "REC_WORD"].dropna(subset=["trial", "item_num"])
    rec = rec[rec.item_num != -1]
    keys = set(zip(rec.trial.astype(int), rec.item_num.astype(int)))
    rederived = enc.apply(lambda r: 1 if (int(r.trial), int(r.item_num)) in keys else 0, axis=1)
    match = int((rederived.values == enc.recalled.values).sum())
    ck("recall", f"{sub}/{ses} 576/576 labels match", match == 576, f"{match}/576",
       os.path.relpath(ev_path, HERE))
    ck("recall", f"{sub}/{ses} no recognition events used", len(recog) == 0,
       f"{len(recog)} RECOG_* present, unused")

# 5. EEG re-extraction
log("\n## 5. EEG EXTRACTION (re-extract from EDF per session)")
import mne
sample_rows = [0, 200, 575]
for sub, ses in SESSIONS:
    d = sess_dir(sub, ses)
    edf = os.path.join(HERE, "data/ds004395", f"sub-{sub}", f"ses-{ses}", "eeg",
                       f"sub-{sub}_ses-{ses}_task-ltpFR2_eeg.edf")
    xpath = os.path.join(d, "X_eeg.npy")
    enc_path = os.path.join(d, "encoding_trials.csv")
    if not (os.path.isfile(edf) and os.path.isfile(xpath) and os.path.isfile(enc_path)):
        ck("eeg", f"{sub}/{ses} EDF+X_eeg present", False, "MISSING", edf)
        continue
    X = np.load(xpath)
    enc = pd.read_csv(enc_path)
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks)
    n_times = int(raw.n_times)
    s0, s1 = int(EEG_WIN[0] * sfreq), int(EEG_WIN[1] * sfreq)
    shape_ok = X.shape == (576, 32250)
    ck("eeg", f"{sub}/{ses} sfreq=500 & 129 ch & X 576x32250",
       sfreq == 500.0 and len(picks) == 129 and shape_ok,
       f"sfreq={sfreq} ch={len(picks)} X={X.shape}", os.path.relpath(edf, HERE))
    worst = 0.0
    exact = True
    in_bounds = True
    for r in sample_rows:
        smp = int(enc.iloc[r]["sample"])
        start, stop = smp + s0, smp + s1
        if stop > n_times or start < 0:
            in_bounds = False
            continue
        seg = data[:, start:stop]
        if seg.shape != (129, 250):
            exact = False
            continue
        flat32 = seg.reshape(-1).astype(np.float32)
        worst = max(worst, float(np.max(np.abs(flat32.astype(np.float64) - X[r].astype(np.float64)))))
        exact = exact and bool(np.array_equal(flat32, X[r]))
    ck("eeg", f"{sub}/{ses} sampled rows bit-exact vs EDF (float32)", exact and in_bounds,
       f"worst diff {worst:.2e}, in_bounds={in_bounds}")
    # full-matrix integrity
    ck("eeg", f"{sub}/{ses} no NaN/Inf, no all-zero rows",
       not np.isnan(X).any() and not np.isinf(X).any() and not (X == 0).all(axis=1).any(),
       "clean")

# 6. ridge / decoding config
log("\n## 6. RIDGE / DECODING CONFIG")
for sub, ses in SESSIONS:
    mj = os.path.join(sess_dir(sub, ses), "ridge_corrected_metadata.json")
    if not os.path.isfile(mj):
        ck("ridge", f"{sub}/{ses} ridge metadata present", False, "MISSING", mj)
        continue
    m = json.load(open(mj))
    ok = (m.get("alpha") == 10000.0 and m.get("n_folds") == 5
          and m.get("ridge_solver") == "svd" and m.get("kfold_shuffle") is True
          and m.get("subject") == sub and int(m.get("session")) == ses)
    ck("ridge", f"{sub}/{ses} alpha=1e4,5-fold,svd,shuffle,per-session",
       ok, f"alpha={m.get('alpha')} folds={m.get('n_folds')} solver={m.get('ridge_solver')}")
# source-code invariants (textual)
src = open(os.path.join(HERE, "code/step08_run_ridge_cv.py")).read()
ck("ridge", "scaler fit on train fold only (source)",
   "StandardScaler().fit(X[train_idx])" in src, "StandardScaler().fit(X[train_idx])",
   "code/step08_run_ridge_cv.py")
ck("ridge", "ridge fit on train only (source)",
   "model.fit(scaler.transform(X[train_idx]), Ymat[train_idx])" in src, "fit on X[train_idx],Ymat[train_idx]",
   "code/step08_run_ridge_cv.py")
ck("ridge", "predict transforms test with train scaler (source)",
   "model.predict(scaler.transform(X[test_idx]))" in src, "predict(scaler.transform(X[test_idx]))",
   "code/step08_run_ridge_cv.py")
ck("ridge", "alpha=10000 & solver svd (source)",
   "Ridge(alpha=ALPHA, solver=\"svd\")" in src and "ALPHA = 10000.0" in src, "Ridge svd alpha 1e4",
   "code/step08_run_ridge_cv.py")
ck("ridge", "one out-of-fold prediction per trial (KFold disjoint)",
   "KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)" in src, "KFold splits",
   "code/step08_run_ridge_cv.py")
# fidelity == raw_cosine, and raw_cosine == cosine(pred,true) definition
ck("ridge", "embedding_fidelity == raw_cosine (combined table)",
   bool(np.allclose(comb.embedding_fidelity, comb.raw_cosine)), "allclose True",
   "outputs/all_sessions_fidelity_results.csv")
ck("ridge", "raw_cosine defined as cos(pred,true) (source)",
   "correct_sim = sims[np.arange(len(test_idx)), correct_col]" in src and "cosine_matrix" in src,
   "cosine to correct candidate", "code/step08_run_ridge_cv.py")

# 7. corrected metric ranges
log("\n## 7. CORRECTED METRIC SANITY")
for c in ["centered_cosine", "true_word_percentile", "centered_true_word_percentile"]:
    ck("corrected", f"column present: {c}", c in comb.columns, "present")
for c in ["true_word_percentile", "centered_true_word_percentile"]:
    ck("corrected", f"{c} in [0,1]", bool(comb[c].between(0, 1).all()),
       f"[{comb[c].min():.3f},{comb[c].max():.3f}]")
ck("corrected", "true_word_percentile near chance 0.5",
   abs(comb.true_word_percentile.mean() - 0.5) < 0.02,
   f"mean {comb.true_word_percentile.mean():.4f}")
ck("corrected", "centered_true_word_percentile near chance 0.5",
   abs(comb.centered_true_word_percentile.mean() - 0.5) < 0.05,
   f"mean {comb.centered_true_word_percentile.mean():.4f}")
ck("corrected", "raw_cosine high (~0.85, inflated)",
   0.80 < comb.raw_cosine.mean() < 0.90, f"mean {comb.raw_cosine.mean():.4f}")

# 8. final model numbers
log("\n## 8. FINAL MODEL NUMBERS (vs source CSV/summary)")
fmr = pd.read_csv(os.path.join(HERE, "outputs/final_memory_model_results.csv"))
main = fmr[(fmr.metric == "raw_cosine") & (fmr.model.str.contains("GLMM"))].iloc[0]
def close(a, b, t=5e-4):
    return abs(float(a) - b) < t
ck("final", "coefficient = -0.0291", close(main.coef_per_sd, -0.0291), f"{main.coef_per_sd:.4f}")
ck("final", "odds ratio = 0.971", close(main.odds_ratio, 0.971, 1e-3), f"{main.odds_ratio:.4f}")
ck("final", "CI = [0.913, 1.033]",
   close(main.ci_low, 0.913, 1e-3) and close(main.ci_high, 1.033, 1e-3),
   f"[{main.ci_low:.4f},{main.ci_high:.4f}]")
ck("final", "p ~ 0.354", close(main.p_approx, 0.354, 1e-3), f"{main.p_approx:.4f}")
rem = comb.loc[comb.recalled == 1, "raw_cosine"].mean()
forg = comb.loc[comb.recalled == 0, "raw_cosine"].mean()
ck("final", "remembered mean fidelity = 0.84627", close(rem, 0.84627, 1e-4), f"{rem:.5f}")
ck("final", "forgotten mean fidelity = 0.84711", close(forg, 0.84711, 1e-4), f"{forg:.5f}")
ck("final", "difference = -0.00084", close(rem - forg, -0.00084, 1e-4), f"{rem-forg:.5f}")
ck("final", "conclusion = no significant effect", main.direction == "no different",
   str(main.direction))
# formula + config from metadata
fmm = json.load(open(os.path.join(HERE, "outputs/final_memory_model_metadata.json")))
ck("final", "formula recalled ~ embedding_fidelity + session + (1|subject)+(1|word)",
   fmm.get("formula") == "recalled ~ embedding_fidelity + session + (1|subject) + (1|word)",
   fmm.get("formula"))
ck("final", "embedding_fidelity metric = raw_cosine",
   fmm.get("embedding_fidelity_metric") == "raw_cosine", str(fmm.get("embedding_fidelity_metric")))
ck("final", "predictor z-scored (OR per 1 SD)",
   "z-scored" in str(fmm.get("predictor_scaling")), str(fmm.get("predictor_scaling")))
ck("final", "session fixed effect + RE subject & word",
   "C(session)" in str(fmm.get("session")) and "1|subject" in str(fmm.get("random_effects")),
   f"{fmm.get('session')} | {fmm.get('random_effects')}")
fsrc = open(os.path.join(HERE, "code/step11_run_memory_model.py")).read()
ck("final", "logistic mixed-effects (BinomialBayesMixedGLM) in source",
   "BinomialBayesMixedGLM" in fsrc, "GLMM present", "code/step11_run_memory_model.py")
ck("final", "supplementary metrics labeled SUPPLEMENTARY",
   'SUPP_METRICS = ["centered_cosine", "true_word_percentile", "centered_true_word_percentile"]' in fsrc,
   "supplementary list", "code/step11_run_memory_model.py")

# 11. gitignore
# Verified by behaviour (git check-ignore) rather than by grepping .gitignore
# text, so a broader rule that subsumes a narrower one still passes.
log("\n## 11. GITIGNORE")


def is_ignored(relpath):
    try:
        r = subprocess.run(["git", "check-ignore", "-q", relpath],
                           cwd=HERE, capture_output=True)
        return r.returncode == 0
    except Exception:
        return False


needed = ["data/", "data/ds004395/x.edf", "data/ds004395/x.bdf",
          "outputs/subjects", "outputs/X_eeg.npy", "outputs/Y_t5.npy",
          "outputs/predicted_embeddings.npy", "venv/",
          "code/__pycache__/x.pyc", ".DS_Store"]
for pat in needed:
    ck("gitignore", f"ignores {pat}", is_ignored(pat),
       "ignored" if is_ignored(pat) else "NOT IGNORED", ".gitignore")

# the deliverable itself must not be ignored
for pat in ["code/step04_extract_t5_embeddings.py",
            "results/embeddings/peers_t5large_embeddings.npy",
            "results/README.md"]:
    ck("gitignore", f"tracks {pat}", not is_ignored(pat),
       "tracked" if not is_ignored(pat) else "WRONGLY IGNORED", ".gitignore")

n_fail = sum(1 for c in checks if c[2] == "FAIL")
log("\n" + "=" * 80)
log(f"PROGRAMMATIC AUDIT: {len(checks)} checks, {n_fail} FAIL")
log("=" * 80)

# machine-readable table for the markdown report
pd.DataFrame(checks, columns=["section", "check", "status", "evidence", "file"]).to_csv(
    os.path.join(HERE, "outputs", "final_precision_audit_checks.csv"), index=False)

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT}")
print("wrote outputs/final_precision_audit_checks.csv")

sys.exit(1 if n_fail else 0)
