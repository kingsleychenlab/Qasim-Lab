# Reproducibility Checklist

## Environments

Two virtual environments are used:

- **Project venv** (`venv/`) — EEG + modeling: `mne`, `mne-bids`, `numpy`,
  `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `requests`.
- **T5 venv** (embeddings only) — `torch`, `transformers`, `sentencepiece`,
  `numpy`, `pandas`, `tqdm`. Needed only to (re)build or audit the T5
  embeddings; not required for the EEG/decoding/model stages.

```bash
# project env
python3 -m venv venv && source venv/bin/activate
pip install mne mne-bids numpy pandas scikit-learn statsmodels matplotlib requests

# T5 env (only for embeddings / T5 audit)
python3 -m venv venv_t5 && source venv_t5/bin/activate
pip install torch transformers sentencepiece numpy pandas tqdm
```

## Where the raw data lives

- OpenNeuro **ds004395** (PEERS), public S3 mirror:
  `https://s3.amazonaws.com/openneuro.org/ds004395/…`
- Downloaded **one session at a time** (never the full 8.7 TB) into
  `data/ds004395/` via `scripts/download_one_session.py --sub <ID> --ses <N> --task ltpFR2`.

## Rerun the key stages

```bash
# 0. T5 embeddings (T5 env)               -> peers_t5large_embeddings.npy (576x1024)
python extract_t5_peers_embeddings.py

# 1. find/process valid ltpFR2 sessions (project env)
#    downloads + encoding_trials + Y_t5 + X_eeg + corrected ridge metrics per session
python scripts/scale_multi_session.py --n-subjects 4 --sessions-per-subject 2 \
    --prefer-subjects sub-LTP269,sub-LTP303,sub-LTP293,sub-LTP299
#    -> outputs/all_sessions_fidelity_results.csv   (4608 rows)   [FINAL INPUT TABLE]

# 2. FINAL memory model
python scripts/run_final_memory_model.py
#    -> outputs/final_memory_model_{summary.txt,results.csv,metadata.json}

# 3. results package (tables + figures)
python scripts/build_results_package.py

# 4. audits
python scripts/audit_recall_labels.py
python scripts/audit_eeg_extraction.py
python scripts/final_precision_audit.py
python scripts/audit_t5_embeddings.py        # T5 env
```

## Exact final artifacts

| Role | Path |
| --- | --- |
| Final input table | `outputs/all_sessions_fidelity_results.csv` |
| Final model script | `scripts/run_final_memory_model.py` |
| Final model result | `outputs/final_memory_model_summary.txt` |
| Final model result | `outputs/final_memory_model_results.csv` |
| Final model metadata | `outputs/final_memory_model_metadata.json` |
| Precision audit | `outputs/final_precision_audit.txt` / `results/final_precision_audit.md` |

## Files too large for GitHub (git-ignored, regenerable)

Excluded by `.gitignore`:

- `data/` and all `*.edf` / `*.bdf` (raw EEG, ~0.5–0.7 GB each)
- `outputs/subjects/` (per-session `X_eeg.npy`, predictions)
- `outputs/X_eeg.npy`, `outputs/Y_t5.npy`, `outputs/predicted_embeddings*.npy`
- `venv/`, `__pycache__/`, `.DS_Store`

Everything needed to reproduce is either small and tracked (scripts, the 576×1024
embedding matrix ~2.3 MB, `all_sessions_fidelity_results.csv`, model outputs,
results/) or downloadable from OpenNeuro with the commands above.

## Determinism notes

- CV: `KFold(shuffle=True, random_state=42)`; shuffled-control seed 2024.
- T5 recompute agrees with the saved matrix to float32 precision (~4e-7 relative;
  larger absolute values ~1e-4 because T5 vector elements are large).
- EEG features re-extract **bit-exactly** from the EDF (float32).
