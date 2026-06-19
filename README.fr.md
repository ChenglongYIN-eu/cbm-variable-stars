<div align="center">

[English](README.md) · [中文](README.zh-CN.md) · [Čeština](README.cs.md) · [Български](README.bg.md) · [Español](README.es.md) · **Français** · [Deutsch](README.de.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

# CBM Variable Stars

**Classification interprétable d'étoiles variables avec des modèles à goulot d'étranglement conceptuel (Concept Bottleneck Models) — chaque prédiction tracée à travers 12 concepts stellaires physiquement significatifs.**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## Vue d'ensemble

`cbm_variable_stars` est, à notre connaissance, la première application des **modèles à goulot d'étranglement conceptuel (Concept Bottleneck Models, CBM)** à la classification astronomique des étoiles variables. Il classe les étoiles variables à partir de la photométrie Gaia DR3 en **6 classes** — RR Lyrae fondamentales et de premier harmonique, Céphéides classiques, Delta Scuti / SX Phoenicis, binaires à éclipses, et variables à longue période de type Mira / semi-régulières — en faisant transiter chaque décision par **12 concepts interprétables et inspectables par un astronome** plutôt que par un espace de caractéristiques opaque.

Avec ce paquet, vous obtenez :

- un pipeline complet et reproductible, des caractéristiques Gaia DR3 (et du croisement inter-relevés OGLE) jusqu'aux classifieurs entraînés ;
- **8 variantes de modèles phares** couvrant tout le spectre de l'interprétabilité — d'un CBM linéaire totalement transparent à 78 paramètres jusqu'aux modèles de référence boîte noire Random Forest et XGBoost ;
- une validation croisée à 5 plis, des études d'ablation, d'intervention et inter-relevés, avec les métriques, tests de significativité, figures et tableaux LaTeX qui étayent l'article associé.

L'article associé est publié dans *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), et une démonstration web interactive est disponible à l'adresse <https://cbm-variable-stars.yinchenglong.com>.

## Pourquoi des modèles à goulot d'étranglement conceptuel

La plupart des classifieurs profonds projettent directement les entrées brutes sur une étiquette, sans fournir aucune explication inspectable de la raison *pour laquelle* une étoile donnée a été attribuée à une classe donnée. Un modèle à goulot d'étranglement conceptuel contraint au contraire chaque prédiction à passer par une couche étroite de **concepts** humainement significatifs :

```
raw photometric features  →  12 physical concepts  →  class
```

Parce que toute l'information de classification doit transiter par le goulot d'étranglement conceptuel à 12 dimensions, chaque prédiction est traçable jusqu'à des grandeurs physiques qu'un astronome peut lire et vérifier — période, amplitude, paramètres de forme de Fourier, couleur, etc. Cela rend également le modèle **interactif** : vous pouvez réécrire un concept (par exemple, fournir une période corrigée) et observer la réponse de la classe prédite. Cette transparence a un coût mesurable en précision brute, et la quantification de ce compromis interprétabilité-performance est un thème central de ce travail.

## Points forts / résultats clés

- **Premier CBM pour la classification d'étoiles variables** — un goulot d'étranglement à 12 concepts physiquement fondé, sur 6 classes d'étoiles variables issues de Gaia DR3.
- **Interprétable par construction** — chaque prédiction est traçable jusqu'à 12 concepts physiques nommés, et les concepts peuvent être réécrits au moment de l'inférence pour intervenir sur une prédiction.
- **Précision forte et honnête** — le Hard CBM atteint une **précision de 94.41 % ± 0.36 %** (macro-F1 94.37 % ± 0.38 %, MCC 0.933) en validation croisée à 5 plis sur le jeu de données phare Gaia DR3 de 18 000 sources (3 000 exemples équilibrés par classe).
- **Le coût de l'interprétabilité, mesuré** — les modèles de référence boîte noire (Random Forest et XGBoost ≈ 99.8 %) surpassent le Hard CBM transparent d'environ 5 points de pourcentage, isolant ainsi le prix d'un goulot d'étranglement conceptuel imposé.
- **8 modèles couvrant le spectre de la transparence** — Hard CBM, Hard CBM-Linear (78 paramètres), Hard CBM-Calibrated, Soft CBM, CEM, un modèle de référence MLP, Random Forest et XGBoost.
- **Testé en inter-relevés** — évalué sous décalage de domaine Gaia → OGLE, parallèlement à des études d'ablation et d'intervention conceptuelle.

## Installation

**Prérequis**

- **Python ≥ 3.10** (déclaré via `python_requires=">=3.10"` dans `setup.py` ; validé sur CPython 3.10–3.13).
- Une chaîne d'outils C/C++ n'est **pas** requise — toutes les dépendances sont fournies sous forme de wheels binaires.
- **Aucun GPU requis.** Les modèles sont petits (HardCBM ≈ 3K paramètres, MLP ≈ 10K paramètres) et la configuration d'entraînement par défaut (taille de lot 256, ≤ 200 époques avec arrêt anticipé) s'exécute confortablement sur CPU. Un GPU CUDA ne fait qu'accélérer les exécutions complètes sur ~18 000 sources.
- Un accès Internet n'est nécessaire **que** pour les étapes facultatives de téléchargement des données (requêtes aux archives Gaia/OGLE via `astroquery`/`pyvo`) ; l'entraînement sur un jeu de données déjà construit fonctionne entièrement hors ligne.

**Cloner**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**Installer**

```bash
# (recommended) create + activate a Python >=3.10 virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editable install (equivalent to `make install`)
pip install -e .
```

Vous pouvez aussi installer l'ensemble complet des dépendances d'exécution épinglées :

```bash
pip install -r requirements.txt
```

**Extras facultatifs** (déclarés dans `setup.py`) :

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**Dépendances de base** (récupérées automatiquement) : `numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`.

> Pour une compilation CUDA explicite, installez le wheel PyTorch correspondant **avant** `pip install -e .`, par exemple `pip install torch --index-url https://download.pytorch.org/whl/cu121`. Le projet n'épingle aucune version de CUDA.

## Démarrage rapide

Le plus petit exemple utile : charger les caractéristiques conceptuelles déjà standardisées fournies avec le paquet et exécuter une passe avant HardCBM pour obtenir les 12 valeurs de concepts et une prédiction à 6 classes.

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

Ceci charge des données réelles, instancie un modèle via le registre, et exerce l'interface unifiée `forward(x) -> {concepts, logits, probabilities}` partagée par chaque variante de modèle. Remplacez `"hard_cbm"` par n'importe quelle clé du registre (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`) pour essayer d'autres architectures.

Pour **entraîner** au lieu d'utiliser des poids aléatoires, encapsulez les caractéristiques dans le jeu de données et pilotez l'orchestration de l'entraînement :

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **Remarque sur la taille du jeu de données.** Les fichiers parquet de `data/processed/` fournis ici sont une partition réduite (`cv_pool` 2 550 + `test_in_domain` 450 + inter-relevés 1 200 lignes). Les chiffres phares de l'article (HardCBM 94.41 ± 0.36 % de précision, CV à 5 plis) proviennent du tableau complet de **18 000 sources** (`data/real/gaia_all_features.parquet`, 3 000 par classe), partitionné en 15 300 CV + 2 700 test. Voir les sections [Reproduire les résultats](#reproduire-les-résultats-pipeline-complet) et [Jeu de données](#jeu-de-données) pour savoir comment reconstruire le jeu de données complet.

## Structure du dépôt

Arborescence annotée du code et des données fournies dans ce dépôt (le paquet importable est `cbm_variable_stars`).

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

> Les 6 classes sont `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR` ; les 12 concepts sont `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` — tous deux définis de manière faisant autorité dans `cbm_variable_stars/shared/constants.py`.

## Reproduire les résultats (pipeline complet)

Le pipeline de bout en bout va de la photométrie brute Gaia DR3 (et OGLE) jusqu'aux modèles entraînés, aux métriques validées par croisement, et aux figures/tableaux de l'article. Chaque étape est un script numéroté sous `scripts/`, piloté par un unique fichier de configuration (`configs/default.yaml`). L'exécution du pipeline régénère un répertoire local `results/` ; les modèles entraînés, métriques et figures publiés se trouvent dans l'article *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)) plutôt que dans cette arborescence de code.

### Prérequis

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

Les étapes 01–02 effectuent des appels réseau aux archives Gaia TAP et OGLE, elles nécessitent donc un accès Internet. L'entraînement (06) s'exécute sur CPU par défaut ; passez `--device cuda` pour un GPU.

### Pas à pas (les scripts réellement présents sur le disque)

Toutes les commandes s'exécutent depuis la racine du dépôt ; chaque script lit `configs/default.yaml` par défaut (à surcharger avec `--config`).

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

Remarques sur les options les plus utilisées :

- **06** requiert `--data_path` ; sa liste `--models` par défaut est le sous-ensemble de 5 modèles `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`. Passez la liste complète des 8 modèles (ci-dessus) pour reproduire la comparaison phare de l'article. Autres valeurs par défaut : `--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`.
- **05** options : `--no-ogle` (Gaia uniquement), `--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (par défaut `10dim`), `--verify-folds`.
- **07** options : `--device cuda`, et `--skip-training`/`--skip-ablation` etc. pour exécuter des sous-ensembles.
- **08** options : `--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`.

### Étape → script → sortie

| Étape | Script | Entrées clés | Sorties clés |
|------|--------|------------|-------------|
| 1. Télécharger Gaia | `scripts/01_download_gaia.py` | archive Gaia TAP | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. Télécharger OGLE | `scripts/02_download_ogle.py` | archive OGLE-IV | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. Extraire les caractéristiques | `scripts/03_extract_features.py` | courbes de lumière brutes | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. Valider les caractéristiques | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, rapport de qualité |
| 5. Construire le jeu de données | `scripts/05_build_dataset.py` | caractéristiques validées | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. Entraîner les modèles | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | résultats de CV par modèle + points de contrôle, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. Exécuter les expériences | `scripts/07_run_experiments.py` | données traitées + modèles entraînés | JSON d'ablation / intervention / inter-relevés / courbe d'apprentissage |
| 8. Générer les figures | `scripts/08_generate_figures.py` | JSON de résultats | figures de l'article (PDF) + tableaux LaTeX |

### Configuration

Toutes les exécutions sont pilotées par **`configs/default.yaml`** (chargé via OmegaConf ; passez `--config <file>` à n'importe quel script pour le surcharger). C'est la source unique de vérité pour : la graine aléatoire (`project.random_seed: 42`) ; le schéma de partition (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`, stratifié, StandardScaler) ; les cibles de téléchargement par classe (`var_types.*`) ; les paramètres d'extraction de caractéristiques (`features.*` — recherche de période, harmoniques de Fourier, détection d'alias à la période de précession Gaia de 63 jours, etc.) ; les hyperparamètres d'entraînement (`training.*` — lot 256, lr 1e-3, max 200 époques, patience 15, planification cosinus avec redémarrages à chaud) ; l'architecture par modèle (`models.*`) ; et les grilles d'expériences (`experiments.*`). Deux configurations auxiliaires existent : `configs/feature_config.yaml` et `configs/gaia_queries.yaml`.

### Remarques sur la reproductibilité / la portée des données

- **Graine et plis.** `random_seed=42` et la CV stratifiée à 5 plis sont fixés dans `configs/default.yaml` ; `05_build_dataset.py` écrit des indices de plis déterministes dans `data/processed/cv_folds.pkl`.
- **Jeu de données phare vs. partition fournie.** Les chiffres phares de l'article (HardCBM 94.41 % ± 0.36 % de précision) proviennent de la matrice Gaia complète et équilibrée de **18 000 sources** (3 000/classe ; 15 300 CV + 2 700 test). Le `data/processed/cv_pool.parquet` fourni *dans cette arborescence* est une **partition de démonstration réduite** (2 550 lignes, 425/classe) ; son exécution ne reproduira pas la précision phare. Réexécuter les étapes de données reconstruit le jeu de données complet à partir des archives ; la matrice de caractéristiques complète est également disponible sous `data/real/gaia_all_features.parquet` (18 000 lignes).
- **OGLE à la demande.** `data/raw/ogle/` est fourni vide à dessein — l'étape 02 télécharge les courbes de lumière OGLE au moment de l'exécution (réseau requis pour le test inter-relevés).

> **Remarque sur le Makefile.** Les cibles `make` reflètent les phases ci-dessus (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`). Les cibles `data:` et `train:` commitées sont légèrement décalées par rapport aux scripts actuels — `make data` référence d'anciens noms de scripts et `make train` omet l'argument `--data_path` requis — de sorte que les commandes explicites `python scripts/0N_*.py` ci-dessus constituent les invocations canoniques et fonctionnelles.

## Jeu de données

Le dépôt fournit un **jeu de données de caractéristiques d'étoiles variables Gaia DR3** sous `data/`, organisé comme une cascade reproductible allant de la photométrie brute jusqu'à des partitions normalisées et prêtes pour la validation croisée. Toutes les données tabulaires sont stockées au format Apache Parquet ; les indices de plis de validation croisée et le scaler ajusté sont en pickle (`.pkl`) ; la table d'étiquettes est en JSON.

Le jeu de données couvre **6 classes d'étoiles variables** décrites par **12 concepts physiquement significatifs** (voir le tableau [Concepts](#concepts)). Le tableau d'étude phare contient **18 000 sources Gaia DR3** équilibrées à **3 000 par classe** (`data/real/gaia_all_features.parquet`, en unités physiques/non mises à l'échelle), partitionné en un pool de validation croisée à 5 plis de 15 300 sources et un ensemble de test de réserve de 2 700 sources.

Organisation sous `data/` :

| Répertoire | Contenu |
|---|---|
| `raw/gaia/epoch_photometry/` | Courbes de lumière en bande G de Gaia DR3 par source, un Parquet par `<source_id>` (colonnes `time, mag, mag_err`). |
| `raw/gaia/metadata/` | Métadonnées de source par classe et combinées (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`). |
| `raw/ogle/` | Vide à dessein ; les courbes de lumière inter-relevés OGLE sont téléchargées à la demande. |
| `interim/` | Caractéristiques extraites avant partitionnement (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`). |
| `processed/` | Partitions prêtes pour la validation croisée, normalisées par StandardScaler (score z) : `cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE, hors domaine), plus `cv_folds.pkl` (indices StratifiedKFold à 5 plis), `scaler.pkl`, et `label_mapping.json`. |
| `expanded/` | Variante augmentée plus grande (`gaia_expanded_features.parquet`, 30 000 lignes). |
| `real/` | Le tableau phare de caractéristiques de 18 000 sources en unités physiques (`gaia_all_features.parquet`) plus les métadonnées brutes. |

Chaque ligne de caractéristiques porte des colonnes d'identifiant/étiquette/qualité (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`) suivies des 12 colonnes de concepts. Le `StandardScaler` global est ajusté uniquement sur le pool de validation croisée et appliqué aux deux ensembles de test ; `period_snr` est imputé par la médiane au moment de la mise à l'échelle.

> **Remarque.** Les fichiers parquet de `data/processed/` fournis dans cette arborescence sont une partition de démonstration réduite (`cv_pool` 2 550 + `test_in_domain` 450 = 3 000 lignes ; test inter-relevés 1 200). La matrice complète de 18 000 sources utilisée pour les résultats phares publiés correspond à `data/real/gaia_all_features.parquet`.

## Modèles

Huit variantes de modèles sont comparées : six réseaux de neurones et deux modèles de référence classiques à arbres. (Le paquet neuronal fournit en outre une variante `EndToEndHardCBM` (`e2e_hard_cbm`) — un encodeur de concepts à CNN 1D sur des courbes de lumière repliées en phase — au-delà des huit variantes phares ci-dessous.)

| Modèle | Clé de registre | Description |
|---|---|---|
| **HardCBM** | `hard_cbm` | Modèle à goulot d'étranglement conceptuel dur ; les caractéristiques d'entrée servent de goulot d'étranglement à 12 concepts alimentant un prédicteur MLP (`12→64→32→6`). Le modèle interprétable de référence. |
| **HardCBM-Linear** | `hard_cbm_linear` | Hard CBM avec un prédicteur unique `Linear(12, 6)` (78 paramètres) ; les poids se lisent directement comme des contributions concept-vers-classe. Interprétabilité maximale. |
| **HardCBM-Cal** | `hard_cbm_cal` | Hard CBM calibré avec 12 têtes de calibration indépendantes qui débruitent les concepts extraits avant un prédicteur MLP ; architecture principale pour les expériences d'intervention. |
| **SoftCBM** | `soft_cbm` | Soft CBM avec des plongements continus par concept (goulot d'étranglement à 48 dimensions) ; un goulot d'étranglement plus large échange de l'interprétabilité contre une précision plus élevée. |
| **CEM** | `cem` | Concept Embedding Model (Espinosa Zarlenga et al. 2022) ; chaque concept est une paire de plongements positif/négatif mélangée par son activation. |
| **MLP** | `mlp` | Perceptron multicouche de référence simple (`12→128→64→6`), sans goulot d'étranglement ; mesure le coût en précision du goulot d'étranglement. |
| **Random Forest** | `rf` | Modèle de référence boîte noire classique (500 arbres, poids de classes équilibrés), aligné sur le classifieur officiel Gaia DR3 (Rimoldini et al. 2023). |
| **XGBoost** | `xgb` | Modèle de référence boîte noire à arbres avec gradient boosting ; valeurs SHAP calculées pour comparaison avec l'importance conceptuelle du CBM. |

### Métriques phares

Validation croisée à 5 plis (`RANDOM_SEED=42`), macro-F1 comme métrique principale, sur le jeu de données complet Gaia DR3 de 18 000 sources (15 300 CV + 2 700 test ; 3 000 par classe). Ce sont les valeurs publiées dans l'article *Astronomy & Astrophysics* (Tableau 2 de l'article) :

| Modèle | Précision (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

L'écart interprétabilité-précision est directement visible : le HardCBM transparent (94.4 %) accuse un retard d'environ 5 points de pourcentage par rapport aux modèles de référence à arbres boîte noire (≈ 99.8 %), ce qui est précisément le coût de l'obligation de faire transiter chaque décision par un goulot d'étranglement physique à 12 concepts.

## Concepts

Les 12 concepts physiques formant le goulot d'étranglement, dans un ordre fixe (définis de manière faisant autorité dans `cbm_variable_stars/shared/constants.py`).

| # | Concept | Unité / Plage | Signification |
|---|---|---|---|
| 1 | `period` | jours | Période primaire de pulsation/variabilité. |
| 2 | `amplitude` | mag | Amplitude crête à crête de la courbe de lumière. |
| 3 | `rise_fraction` | sans dimension, [0, 1] | Fraction du cycle passée en montée de luminosité. |
| 4 | `R21` | rapport sans dimension | Rapport d'amplitudes de Fourier A2/A1. |
| 5 | `R31` | rapport sans dimension | Rapport d'amplitudes de Fourier A3/A1. |
| 6 | `phi21` | radians, [0, 2π) | Différence de phase de Fourier φ2 − 2φ1. |
| 7 | `skewness` | sans dimension | Asymétrie de la distribution des magnitudes. |
| 8 | `kurtosis` | sans dimension | Aplatissement (de Fisher) en excès de la distribution des magnitudes. |
| 9 | `stetson_K` | sans dimension | Indice de variabilité K de Stetson. |
| 10 | `period_snr` | sans dimension | Significativité de la période, −log₁₀(probabilité de fausse alarme). |
| 11 | `color_bp_rp` | mag | Indice de couleur Gaia BP − RP. |
| 12 | `mean_mag` | mag | Magnitude moyenne en bande G de Gaia. |

Les six classes sont `RRAB` (RR Lyrae, mode fondamental), `RRC` (RR Lyrae, premier harmonique), `DCEP` (Céphéide classique), `DSCT_SXPHE` (Delta Scuti / SX Phoenicis), `ECL` (binaire à éclipses) et `MIRA_SR` (variable à longue période de type Mira / semi-régulière), indexées de 0 à 5 dans cet ordre.

## Citation

Si vous utilisez ce travail, veuillez citer l'article associé :

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

Texte brut : Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI : 10.1051/0004-6361/202659990.

## Licence

Ce dépôt ne fournit actuellement aucun fichier de licence open source explicite, aucun octroi général de droits de réutilisation, redistribution ou modification ne doit donc être présumé. Si vous souhaitez réutiliser le code ou les données au-delà des conditions de citation ci-dessus, veuillez contacter l'auteur (Chenglong Yin, Université de Sofia) pour obtenir une autorisation. Une licence formelle pourra être ajoutée dans une version future.

## Liens

- **Article (Astronomy & Astrophysics) :** <https://doi.org/10.1051/0004-6361/202659990>
- **Compagnon web interactif :** <https://cbm-variable-stars.yinchenglong.com>
