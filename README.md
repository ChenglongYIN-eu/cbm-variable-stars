<div align="center">

**English** · [中文](README.zh-CN.md) · [Čeština](README.cs.md) · [Български](README.bg.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

# CBM Variable Stars

**Interpretable variable-star classification with Concept Bottleneck Models — every prediction traced through 12 physically meaningful stellar concepts.**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## Overview

`cbm_variable_stars` is, to our knowledge, the first application of **Concept Bottleneck Models (CBMs)** to astronomical variable-star classification. It classifies variable stars from Gaia DR3 photometry into **6 classes** — RR Lyrae fundamental and overtone, classical Cepheids, Delta Scuti / SX Phoenicis, eclipsing binaries, and Mira / semi-regular long-period variables — by routing every decision through **12 interpretable, astronomer-inspectable concepts** rather than an opaque feature space.

With this package you get:

- a complete, reproducible pipeline from Gaia DR3 (and OGLE cross-survey) features to trained classifiers;
- **8 headline model variants** spanning the interpretability spectrum — from a fully transparent 78-parameter linear CBM to black-box Random Forest and XGBoost baselines;
- 5-fold cross-validation, ablation, intervention, and cross-survey experiments, with the metrics, significance tests, figures, and LaTeX tables that back the accompanying paper.

The companion paper is published in *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), and an interactive web demo is available at <https://cbm-variable-stars.yinchenglong.com>.

## Why concept bottleneck models

Most deep classifiers map raw inputs straight to a label, leaving no inspectable account of *why* a given star was assigned to a given class. A Concept Bottleneck Model instead forces every prediction to pass through a narrow layer of human-meaningful **concepts**:

```
raw photometric features  →  12 physical concepts  →  class
```

Because all classification information must flow through the 12-dimensional concept bottleneck, every prediction is traceable to physical quantities an astronomer can read and check — period, amplitude, Fourier shape parameters, colour, and so on. It also makes the model **interactable**: you can overwrite a concept (for example, supply a corrected period) and watch the predicted class respond. This transparency has a measurable cost in raw accuracy, and quantifying that interpretability-versus-performance trade-off is a central theme of this work.

## Highlights / key results

- **First CBM for variable-star classification** — a physically grounded 12-concept bottleneck over 6 variable-star classes from Gaia DR3.
- **Interpretable by construction** — every prediction is traceable to 12 named physical concepts, and concepts can be overridden at inference time to intervene on a prediction.
- **Strong, honest accuracy** — the Hard CBM reaches **94.41% ± 0.36% accuracy** (macro-F1 94.37% ± 0.38%, MCC 0.933) under 5-fold cross-validation on the headline 18,000-source Gaia DR3 dataset (3,000 balanced examples per class).
- **The interpretability cost, measured** — black-box baselines (Random Forest and XGBoost ≈ 99.8%) outperform the transparent Hard CBM by roughly 5 percentage points, isolating the price of an enforced concept bottleneck.
- **8 models across the transparency spectrum** — Hard CBM, Hard CBM-Linear (78 parameters), Hard CBM-Calibrated, Soft CBM, CEM, an MLP baseline, Random Forest, and XGBoost.
- **Cross-survey tested** — evaluated under Gaia → OGLE domain shift, alongside ablation and concept-intervention studies.

## Installation

**Prerequisites**

- **Python ≥ 3.10** (declared via `python_requires=">=3.10"` in `setup.py`; known-good on CPython 3.10–3.13).
- A C/C++ toolchain is **not** required — all dependencies ship as binary wheels.
- **No GPU required.** The models are small (HardCBM ≈ 3K parameters, MLP ≈ 10K parameters) and the default training config (batch size 256, ≤ 200 epochs with early stopping) runs comfortably on CPU. A CUDA GPU only speeds up the full ~18,000-source runs.
- Internet access is needed **only** for the optional data-download steps (Gaia/OGLE archive queries via `astroquery`/`pyvo`); training on an already-built dataset works fully offline.

**Clone**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**Install**

```bash
# (recommended) create + activate a Python >=3.10 virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editable install (equivalent to `make install`)
pip install -e .
```

Alternatively, install the full pinned runtime set:

```bash
pip install -r requirements.txt
```

**Optional extras** (declared in `setup.py`):

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**Core dependencies** (pulled automatically): `numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`.

> For an explicit CUDA build, install the matching PyTorch wheel **before** `pip install -e .`, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`. The project does not pin a CUDA version.

## Quickstart

The smallest useful example: load the shipped, already-standardized concept features and run a HardCBM forward pass to get the 12 concept values and a 6-class prediction.

```python
import pandas as pd
import torch

from cbm_variable_stars.models import create_model
from cbm_variable_stars.shared.constants import CONCEPT_NAMES, CLASS_NAMES

# 1. Load the StandardScaler-normalized concept table shipped under data/processed/
df = pd.read_parquet("data/processed/cv_pool.parquet")           # 2,550 rows
x = torch.tensor(df[CONCEPT_NAMES].values, dtype=torch.float32)  # shape (N, 12)

# 2. Build a Hard Concept Bottleneck Model (12 concepts -> [64, 32] MLP -> 6 classes)
model = create_model("hard_cbm")   # untrained weights; see training/ to fit
model.eval()

# 3. Forward pass -> dict{concepts (N, 12), logits (N, 6), probabilities (N, 6)}
with torch.no_grad():
    out = model(x)

pred_idx   = out["probabilities"].argmax(dim=1)
pred_class = [CLASS_NAMES[i] for i in pred_idx.tolist()]

print("concepts:", CONCEPT_NAMES)
print("first 5 predictions:", pred_class[:5])
print("true labels        :", df["label_name"].head().tolist())
```

This loads real data, instantiates a model through the registry, and exercises the unified `forward(x) -> {concepts, logits, probabilities}` interface shared by every model variant. Swap `"hard_cbm"` for any registry key (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`) to try other architectures.

To **train** instead of using random weights, wrap the features in the dataset and drive the training orchestration:

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **Note on dataset size.** The `data/processed/` parquets shipped here are a reduced split (`cv_pool` 2,550 + `test_in_domain` 450 + cross-survey 1,200 rows). The paper's headline figures (HardCBM 94.41 ± 0.36% accuracy, 5-fold CV) come from the full **18,000-source** table (`data/real/gaia_all_features.parquet`, 3,000 per class), split 15,300 CV + 2,700 test. See the [Reproducing the results](#reproducing-the-results-full-pipeline) and [Dataset](#dataset) sections for how to rebuild the full dataset.

## Repository structure

Annotated tree of the code and shipped data in this repository (the import package is `cbm_variable_stars`).

```
cbm_variable_stars/
├── setup.py                      # Package metadata; name="cbm_variable_stars", python_requires>=3.10
├── requirements.txt              # Pinned runtime + dev dependencies
├── Makefile                      # Convenience targets (install / data / train / experiments / figures)
├── configs/                      # OmegaConf YAML configs (default.yaml, feature_config.yaml, gaia_queries.yaml)
├── scripts/                      # Numbered pipeline stages 01–08 + experiment drivers
├── test_full_pipeline.py         # End-to-end smoke test of the data -> train -> eval pipeline
│
├── cbm_variable_stars/           # The importable Python package
│   ├── __init__.py               # Sets __version__ ("0.1.0"); no symbol re-exports
│   ├── shared/                   # Single source of truth: constants, seeds, imputation, IO, config, logging
│   │   └── constants.py          #   CLASS_NAMES (6), CONCEPT_NAMES_12, RANDOM_SEED=42, N_CV_FOLDS=5, ...
│   ├── data/                     # Gaia/OGLE download, cross-match, alias detection, PyTorch datasets, splits
│   │   └── dataset.py            #   VariableStarDataset + create_dataloader (expects pre-scaled features)
│   ├── features/                 # Light-curve -> 12-concept feature extraction (period, Fourier, amplitude, stats)
│   ├── dataset/                  # Builds train/test/CV splits + fits the global StandardScaler
│   ├── models/                   # All neural architectures + factory (MODEL_REGISTRY, create_model)
│   │   ├── cbm_hard.py           #   HardCBM, HardCBM_Linear, HardCBM_Calibrated, EndToEndHardCBM
│   │   ├── cbm_soft.py           #   SoftCBM (per-concept embeddings, 48-dim bottleneck)
│   │   ├── cem.py                #   ConceptEmbeddingModel (Espinosa Zarlenga et al. 2022)
│   │   ├── mlp_baseline.py       #   BaselineMLP (no bottleneck; accuracy-cost reference)
│   │   └── concept_encoder.py    #   PhaseCurveEncoder (1D-CNN over phase-folded light curves)
│   ├── losses/                   # CBM training losses (joint / calibration / sequential / independent)
│   ├── baselines/                # Classical baselines: random_forest.py, xgboost_model.py
│   ├── training/                 # Trainer, cross-validation, parallel CV, callbacks, hyperparam search
│   ├── evaluation/               # Metrics, statistical significance tests, results aggregation/reporting
│   ├── experiments/              # Full experiment drivers (ablation, intervention, cross-survey, ...)
│   ├── visualization/            # Figures + LaTeX table export
│   └── paper/                    # Reproduces manuscript figures + tables
│
└── data/                         # Shipped Gaia DR3 variable-star dataset (Apache Parquet / pickle / JSON)
    ├── raw/                      # Original survey inputs
    │   ├── gaia/epoch_photometry/   # Per-source light curves <source_id>.parquet (time, mag, mag_err)
    │   ├── gaia/metadata/           # Per-class + combined Gaia metadata tables
    │   └── ogle/                    # Empty by design — OGLE curves are downloaded on demand
    ├── interim/                  # Extracted features, not yet split (gaia_features_raw.parquet, ...)
    ├── processed/                # CV-ready, StandardScaler-normalized deliverable:
    │   ├── cv_pool.parquet           #   2,550 rows (425/class) — 85% pool for 5-fold CV
    │   ├── test_in_domain.parquet    #   450 rows — 15% Gaia hold-out
    │   ├── test_cross_survey.parquet #   1,200 rows — OGLE out-of-domain test (source='ogle')
    │   ├── cv_folds.pkl              #   5-fold StratifiedKFold indices
    │   ├── scaler.pkl                #   Global scaler + medians (fit on cv_pool only)
    │   └── label_mapping.json        #   Class -> index map (RRAB=0 ... MIRA_SR=5)
    ├── expanded/                 # Larger augmented variant (gaia_expanded_features.parquet, 30,000 rows)
    └── real/                     # Headline study table: gaia_all_features.parquet (18,000 rows, physical units)
```

> The 6 classes are `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR`; the 12 concepts are `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` — both defined authoritatively in `cbm_variable_stars/shared/constants.py`.

## Reproducing the results (full pipeline)

The end-to-end pipeline goes from raw Gaia DR3 (and OGLE) photometry to trained models, cross-validated metrics, and the paper's figures/tables. Every stage is a numbered script under `scripts/`, driven by a single config file (`configs/default.yaml`). Running the pipeline regenerates a local `results/` directory; the published trained models, metrics, and figures live in the *Astronomy & Astrophysics* paper (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)) rather than in this code tree.

### Prerequisites

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

Stages 01–02 make network calls to the Gaia TAP and OGLE archives, so they need internet access. Training (06) runs on CPU by default; pass `--device cuda` for a GPU.

### Step-by-step (the actual scripts on disk)

All commands run from the repo root; every script reads `configs/default.yaml` by default (override with `--config`).

```bash
# --- Data ---
python scripts/01_download_gaia.py        # Gaia DR3 metadata + epoch photometry (all 6 classes)
python scripts/02_download_ogle.py        # OGLE-IV catalog + light curves (cross-survey test)
python scripts/03_extract_features.py     # light curves -> 12 concepts; Gaia-OGLE cross-match for C11/C12
python scripts/04_validate_features.py    # sigma-clip + physical-bounds QC, quality report
python scripts/05_build_dataset.py        # 15% hold-out + 85% 5-fold CV pool; fit StandardScaler

# --- Train ---
python scripts/06_train_models.py \
    --data_path data/processed/cv_pool.parquet \
    --ogle_data_path data/processed/test_cross_survey.parquet \
    --models hard_cbm hard_cbm_linear hard_cbm_cal soft_cbm cem mlp rf xgb \
    --output_dir results/ --seed 42 --device cpu

# --- Experiments ---
python scripts/07_run_experiments.py      # ablations, interventions, cross-survey, learning curve

# --- Figures + tables ---
python scripts/08_generate_figures.py     # all paper figures (PDF) + LaTeX tables
```

Notes on the most-used flags:

- **06** requires `--data_path`; its default `--models` list is the 5-model subset `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`. Pass the full 8-model list (above) to reproduce the paper's headline comparison. Other defaults: `--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`.
- **05** options: `--no-ogle` (Gaia-only), `--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (default `10dim`), `--verify-folds`.
- **07** options: `--device cuda`, and `--skip-training`/`--skip-ablation` etc. to run subsets.
- **08** options: `--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`.

### Step → script → output

| Step | Script | Key inputs | Key outputs |
|------|--------|------------|-------------|
| 1. Download Gaia | `scripts/01_download_gaia.py` | Gaia TAP archive | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. Download OGLE | `scripts/02_download_ogle.py` | OGLE-IV archive | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. Extract features | `scripts/03_extract_features.py` | raw light curves | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. Validate features | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, quality report |
| 5. Build dataset | `scripts/05_build_dataset.py` | validated features | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. Train models | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | per-model CV results + checkpoints, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. Run experiments | `scripts/07_run_experiments.py` | processed data + trained models | ablation / intervention / cross-survey / learning-curve JSON |
| 8. Generate figures | `scripts/08_generate_figures.py` | results JSON | paper figures (PDF) + LaTeX tables |

### Configuration

All runs are driven by **`configs/default.yaml`** (loaded via OmegaConf; pass `--config <file>` to any script to override). It is the single source of truth for: random seed (`project.random_seed: 42`); split scheme (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`, stratified, StandardScaler); per-class download targets (`var_types.*`); feature-extraction parameters (`features.*` — period search, Fourier harmonics, alias detection at the 63-day Gaia precession period, etc.); training hyperparameters (`training.*` — batch 256, lr 1e-3, max 200 epochs, patience 15, cosine warm-restart schedule); per-model architecture (`models.*`); and experiment grids (`experiments.*`). Two auxiliary configs exist: `configs/feature_config.yaml` and `configs/gaia_queries.yaml`.

### Reproducibility / data-scope notes

- **Seed & folds.** `random_seed=42` and 5-fold stratified CV are fixed in `configs/default.yaml`; `05_build_dataset.py` writes deterministic fold indices to `data/processed/cv_folds.pkl`.
- **Headline dataset vs. shipped split.** The paper's headline numbers (HardCBM 94.41% ± 0.36% accuracy) come from the full **18,000-source** balanced Gaia matrix (3,000/class; 15,300 CV + 2,700 test). The `data/processed/cv_pool.parquet` shipped *in this tree* is a **reduced demo split** (2,550 rows, 425/class); running on it will not reproduce the headline accuracy. Re-running the data stages rebuilds the full dataset from the archives; the full feature matrix is also available as `data/real/gaia_all_features.parquet` (18,000 rows).
- **OGLE on demand.** `data/raw/ogle/` is shipped empty by design — step 02 downloads OGLE light curves at run time (network required for the cross-survey test).

> **Makefile note.** The `make` targets mirror the phases above (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`). The committed `data:` and `train:` targets are slightly out of step with the current scripts — `make data` references older script names and `make train` omits the required `--data_path` argument — so the explicit `python scripts/0N_*.py` commands above are the canonical, working invocations.

## Dataset

The repository ships a **Gaia DR3 variable-star feature dataset** under `data/`, organized as a reproducible cascade from raw photometry to cross-validation-ready, normalized splits. All tabular data is stored as Apache Parquet; cross-validation fold indices and the fitted scaler are pickle (`.pkl`); the label map is JSON.

The dataset covers **6 variable-star classes** described by **12 physically meaningful concepts** (see the [Concepts](#concepts) table). The headline study table contains **18,000 Gaia DR3 sources** balanced at **3,000 per class** (`data/real/gaia_all_features.parquet`, in physical/unscaled units), split into a 15,300-source 5-fold cross-validation pool and a 2,700-source hold-out test set.

Organization under `data/`:

| Directory | Contents |
|---|---|
| `raw/gaia/epoch_photometry/` | Per-source Gaia DR3 G-band light curves, one Parquet per `<source_id>` (columns `time, mag, mag_err`). |
| `raw/gaia/metadata/` | Per-class and combined source metadata (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`). |
| `raw/ogle/` | Empty by design; OGLE cross-survey light curves are downloaded on demand. |
| `interim/` | Extracted features before splitting (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`). |
| `processed/` | Cross-validation-ready, StandardScaler-normalized (z-score) splits: `cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE, out-of-domain), plus `cv_folds.pkl` (5-fold StratifiedKFold indices), `scaler.pkl`, and `label_mapping.json`. |
| `expanded/` | Larger augmented variant (`gaia_expanded_features.parquet`, 30,000 rows). |
| `real/` | The headline 18,000-source feature table in physical units (`gaia_all_features.parquet`) plus raw metadata. |

Each feature row carries identifier/label/quality columns (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`) followed by the 12 concept columns. The global `StandardScaler` is fit on the cross-validation pool only and applied to both test sets; `period_snr` is median-imputed at scaling time.

> **Note.** The `data/processed/` parquets shipped in this tree are a reduced demonstration split (`cv_pool` 2,550 + `test_in_domain` 450 = 3,000 rows; cross-survey test 1,200). The full 18,000-source matrix used for the published headline results corresponds to `data/real/gaia_all_features.parquet`.

## Models

Eight model variants are compared: six neural networks and two classical tree baselines. (The neural package additionally ships an `EndToEndHardCBM` (`e2e_hard_cbm`) variant — a 1D-CNN concept encoder over phase-folded light curves — beyond the eight headline variants below.)

| Model | Registry key | Description |
|---|---|---|
| **HardCBM** | `hard_cbm` | Hard Concept Bottleneck Model; input features serve as the 12-concept bottleneck feeding an MLP predictor (`12→64→32→6`). The reference interpretable model. |
| **HardCBM-Linear** | `hard_cbm_linear` | Hard CBM with a single `Linear(12, 6)` predictor (78 parameters); weights read directly as concept-to-class contributions. Maximally interpretable. |
| **HardCBM-Cal** | `hard_cbm_cal` | Calibrated Hard CBM with 12 independent calibration heads that denoise the extracted concepts before an MLP predictor; primary architecture for intervention experiments. |
| **SoftCBM** | `soft_cbm` | Soft CBM with per-concept continuous embeddings (48-dimensional bottleneck); wider bottleneck trades interpretability for higher accuracy. |
| **CEM** | `cem` | Concept Embedding Model (Espinosa Zarlenga et al. 2022); each concept is a positive/negative embedding pair mixed by its activation. |
| **MLP** | `mlp` | Plain multilayer perceptron baseline (`12→128→64→6`), no bottleneck; measures the accuracy cost of the bottleneck. |
| **Random Forest** | `rf` | Classical black-box baseline (500 trees, balanced class weights), aligned with the Gaia DR3 official classifier (Rimoldini et al. 2023). |
| **XGBoost** | `xgb` | Gradient-boosted tree black-box baseline; SHAP values computed for comparison with CBM concept importance. |

### Headline metrics

5-fold cross-validation (`RANDOM_SEED=42`), macro-F1 as the primary metric, on the full 18,000-source Gaia DR3 dataset (15,300 CV + 2,700 test; 3,000 per class). These are the published *Astronomy & Astrophysics* paper values (paper Table 2):

| Model | Accuracy (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

The interpretability-versus-accuracy gap is visible directly: the transparent HardCBM (94.4%) trails the black-box tree baselines (≈ 99.8%) by roughly 5 percentage points, which is precisely the cost of forcing every decision through a 12-concept physical bottleneck.

## Concepts

The 12 physical concepts forming the bottleneck, in fixed order (authoritatively defined in `cbm_variable_stars/shared/constants.py`).

| # | Concept | Unit / Range | Meaning |
|---|---|---|---|
| 1 | `period` | days | Primary pulsation/variability period. |
| 2 | `amplitude` | mag | Peak-to-peak light-curve amplitude. |
| 3 | `rise_fraction` | dimensionless, [0, 1] | Fraction of the cycle spent rising in brightness. |
| 4 | `R21` | dimensionless ratio | Fourier amplitude ratio A2/A1. |
| 5 | `R31` | dimensionless ratio | Fourier amplitude ratio A3/A1. |
| 6 | `phi21` | radians, [0, 2π) | Fourier phase difference φ2 − 2φ1. |
| 7 | `skewness` | dimensionless | Skewness of the magnitude distribution. |
| 8 | `kurtosis` | dimensionless | Excess (Fisher) kurtosis of the magnitude distribution. |
| 9 | `stetson_K` | dimensionless | Stetson K variability index. |
| 10 | `period_snr` | dimensionless | Period significance, −log₁₀(false-alarm probability). |
| 11 | `color_bp_rp` | mag | Gaia BP − RP colour index. |
| 12 | `mean_mag` | mag | Mean Gaia G-band magnitude. |

The six classes are `RRAB` (RR Lyrae, fundamental mode), `RRC` (RR Lyrae, first overtone), `DCEP` (classical Cepheid), `DSCT_SXPHE` (Delta Scuti / SX Phoenicis), `ECL` (eclipsing binary), and `MIRA_SR` (Mira / semi-regular long-period variable), indexed 0–5 in that order.

## Citation

If you use this work, please cite the accompanying paper:

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

Plain text: Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990.

## License

This repository does not currently ship an explicit open-source license file, so no general grant of reuse, redistribution, or modification rights should be assumed. If you would like to reuse the code or data beyond the citation terms above, please contact the author (Chenglong Yin, Sofia University) to arrange permission. A formal license may be added in a future release.

## Links

- **Paper (Astronomy & Astrophysics):** <https://doi.org/10.1051/0004-6361/202659990>
- **Interactive web companion:** <https://cbm-variable-stars.yinchenglong.com>
