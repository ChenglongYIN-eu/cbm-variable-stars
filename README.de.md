<div align="center">

[English](README.md) · [中文](README.zh-CN.md) · [Čeština](README.cs.md) · [Български](README.bg.md) · [Español](README.es.md) · [Français](README.fr.md) · **Deutsch** · [Русский](README.ru.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

# CBM Veränderliche Sterne

**Interpretierbare Klassifikation veränderlicher Sterne mit Concept-Bottleneck-Modellen — jede Vorhersage über 12 physikalisch bedeutsame stellare Konzepte nachvollziehbar.**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## Überblick

`cbm_variable_stars` ist nach unserem Kenntnisstand die erste Anwendung von **Concept-Bottleneck-Modellen (CBMs)** auf die astronomische Klassifikation veränderlicher Sterne. Es klassifiziert veränderliche Sterne aus der Gaia-DR3-Photometrie in **6 Klassen** — RR-Lyrae-Sterne im Grund- und Obertonmodus, klassische Cepheiden, Delta Scuti / SX Phoenicis, Bedeckungsveränderliche sowie langperiodische Veränderliche vom Typ Mira / halbregelmäßig — indem es jede Entscheidung durch **12 interpretierbare, für Astronomen prüfbare Konzepte** leitet statt durch einen undurchsichtigen Merkmalsraum.

Mit diesem Paket erhalten Sie:

- eine vollständige, reproduzierbare Pipeline von Gaia-DR3-Merkmalen (sowie OGLE-Merkmalen für die surveyübergreifende Auswertung) bis zu trainierten Klassifikatoren;
- **8 zentrale Modellvarianten** über das gesamte Interpretierbarkeitsspektrum — von einem vollständig transparenten linearen CBM mit 78 Parametern bis hin zu den Black-Box-Baselines Random Forest und XGBoost;
- 5-fache Kreuzvalidierung, Ablations-, Interventions- und surveyübergreifende Experimente, samt den Metriken, Signifikanztests, Abbildungen und LaTeX-Tabellen, die die begleitende Veröffentlichung untermauern.

Die begleitende Veröffentlichung ist in *Astronomy & Astrophysics* erschienen (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), und eine interaktive Web-Demo ist unter <https://cbm-variable-stars.yinchenglong.com> verfügbar.

## Warum Concept-Bottleneck-Modelle

Die meisten Deep-Learning-Klassifikatoren bilden Rohdaten direkt auf ein Label ab und lassen keine prüfbare Begründung dafür, *warum* ein bestimmter Stern einer bestimmten Klasse zugeordnet wurde. Ein Concept-Bottleneck-Modell zwingt stattdessen jede Vorhersage, eine schmale Schicht für Menschen bedeutsamer **Konzepte** zu durchlaufen:

```
raw photometric features  →  12 physical concepts  →  class
```

Da sämtliche Klassifikationsinformation durch den 12-dimensionalen Konzept-Bottleneck fließen muss, ist jede Vorhersage auf physikalische Größen zurückführbar, die ein Astronom lesen und prüfen kann — Periode, Amplitude, Fourier-Formparameter, Farbe und so weiter. Außerdem wird das Modell dadurch **interaktiv**: Sie können ein Konzept überschreiben (zum Beispiel eine korrigierte Periode angeben) und beobachten, wie die vorhergesagte Klasse reagiert. Diese Transparenz hat messbare Kosten bei der reinen Genauigkeit, und die Quantifizierung dieses Kompromisses zwischen Interpretierbarkeit und Leistung ist ein zentrales Thema dieser Arbeit.

## Höhepunkte / zentrale Ergebnisse

- **Erstes CBM für die Klassifikation veränderlicher Sterne** — ein physikalisch fundierter Bottleneck aus 12 Konzepten über 6 Klassen veränderlicher Sterne aus Gaia DR3.
- **Interpretierbar durch Konstruktion** — jede Vorhersage ist auf 12 benannte physikalische Konzepte zurückführbar, und Konzepte können zur Inferenzzeit überschrieben werden, um in eine Vorhersage einzugreifen.
- **Starke, ehrliche Genauigkeit** — das Hard CBM erreicht **94,41 % ± 0,36 % Genauigkeit** (Makro-F1 94,37 % ± 0,38 %, MCC 0,933) unter 5-facher Kreuzvalidierung auf dem zentralen Gaia-DR3-Datensatz mit 18.000 Quellen (3.000 ausbalancierte Beispiele pro Klasse).
- **Die Interpretierbarkeitskosten, gemessen** — die Black-Box-Baselines (Random Forest und XGBoost ≈ 99,8 %) übertreffen das transparente Hard CBM um rund 5 Prozentpunkte und isolieren damit den Preis eines erzwungenen Konzept-Bottlenecks.
- **8 Modelle über das gesamte Transparenzspektrum** — Hard CBM, Hard CBM-Linear (78 Parameter), Hard CBM-Calibrated, Soft CBM, CEM, eine MLP-Baseline, Random Forest und XGBoost.
- **Surveyübergreifend getestet** — ausgewertet unter dem Domänenwechsel Gaia → OGLE, neben Ablations- und Konzeptinterventionsstudien.

## Installation

**Voraussetzungen**

- **Python ≥ 3.10** (deklariert über `python_requires=">=3.10"` in `setup.py`; nachweislich funktionsfähig unter CPython 3.10–3.13).
- Eine C/C++-Toolchain ist **nicht** erforderlich — alle Abhängigkeiten werden als Binär-Wheels ausgeliefert.
- **Keine GPU erforderlich.** Die Modelle sind klein (HardCBM ≈ 3K Parameter, MLP ≈ 10K Parameter), und die Standard-Trainingskonfiguration (Batch-Größe 256, ≤ 200 Epochen mit Early Stopping) läuft problemlos auf der CPU. Eine CUDA-GPU beschleunigt lediglich die vollständigen Läufe mit ~18.000 Quellen.
- Internetzugang wird **nur** für die optionalen Datendownload-Schritte benötigt (Gaia/OGLE-Archivabfragen über `astroquery`/`pyvo`); das Training auf einem bereits aufgebauten Datensatz funktioniert vollständig offline.

**Klonen**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**Installieren**

```bash
# (recommended) create + activate a Python >=3.10 virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editable install (equivalent to `make install`)
pip install -e .
```

Alternativ können Sie den vollständigen, fixierten Laufzeit-Satz installieren:

```bash
pip install -r requirements.txt
```

**Optionale Extras** (deklariert in `setup.py`):

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**Kernabhängigkeiten** (werden automatisch eingebunden): `numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`.

> Für einen expliziten CUDA-Build installieren Sie das passende PyTorch-Wheel **vor** `pip install -e .`, z. B. `pip install torch --index-url https://download.pytorch.org/whl/cu121`. Das Projekt fixiert keine CUDA-Version.

## Schnellstart

Das kleinste nützliche Beispiel: Laden Sie die mitgelieferten, bereits standardisierten Konzeptmerkmale und führen Sie einen HardCBM-Vorwärtsdurchlauf aus, um die 12 Konzeptwerte und eine 6-Klassen-Vorhersage zu erhalten.

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

Dies lädt echte Daten, instanziiert ein Modell über die Registry und durchläuft die einheitliche `forward(x) -> {concepts, logits, probabilities}`-Schnittstelle, die alle Modellvarianten teilen. Ersetzen Sie `"hard_cbm"` durch einen beliebigen Registry-Schlüssel (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`), um andere Architekturen auszuprobieren.

Um stattdessen zu **trainieren**, statt zufällige Gewichte zu verwenden, kapseln Sie die Merkmale im Datensatz und steuern Sie die Trainingsorchestrierung:

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **Hinweis zur Datensatzgröße.** Die hier mitgelieferten Parquet-Dateien unter `data/processed/` sind ein reduzierter Split (`cv_pool` 2.550 + `test_in_domain` 450 + surveyübergreifend 1.200 Zeilen). Die zentralen Kennzahlen der Veröffentlichung (HardCBM 94,41 ± 0,36 % Genauigkeit, 5-fache CV) stammen aus der vollständigen Tabelle mit **18.000 Quellen** (`data/real/gaia_all_features.parquet`, 3.000 pro Klasse), aufgeteilt in 15.300 CV + 2.700 Test. Wie der vollständige Datensatz neu aufgebaut wird, beschreiben die Abschnitte [Reproduktion der Ergebnisse](#reproduktion-der-ergebnisse-vollständige-pipeline) und [Datensatz](#datensatz).

## Repository-Struktur

Kommentierter Baum des Codes und der mitgelieferten Daten in diesem Repository (das Import-Paket heißt `cbm_variable_stars`).

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

> Die 6 Klassen sind `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR`; die 12 Konzepte sind `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` — beide maßgeblich definiert in `cbm_variable_stars/shared/constants.py`.

## Reproduktion der Ergebnisse (vollständige Pipeline)

Die End-to-End-Pipeline reicht von der rohen Gaia-DR3- (und OGLE-)Photometrie bis zu trainierten Modellen, kreuzvalidierten Metriken und den Abbildungen/Tabellen der Veröffentlichung. Jede Stufe ist ein nummeriertes Skript unter `scripts/`, gesteuert durch eine einzige Konfigurationsdatei (`configs/default.yaml`). Das Ausführen der Pipeline erzeugt erneut ein lokales `results/`-Verzeichnis; die veröffentlichten trainierten Modelle, Metriken und Abbildungen finden sich in der *Astronomy & Astrophysics*-Veröffentlichung (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)) und nicht in diesem Code-Baum.

### Voraussetzungen

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

Die Stufen 01–02 führen Netzwerkaufrufe an die Gaia-TAP- und OGLE-Archive durch und benötigen daher Internetzugang. Das Training (06) läuft standardmäßig auf der CPU; übergeben Sie `--device cuda` für eine GPU.

### Schritt für Schritt (die tatsächlichen Skripte auf der Festplatte)

Alle Befehle werden vom Repository-Wurzelverzeichnis aus ausgeführt; jedes Skript liest standardmäßig `configs/default.yaml` (überschreibbar mit `--config`).

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

Hinweise zu den am häufigsten verwendeten Flags:

- **06** erfordert `--data_path`; die Standard-`--models`-Liste ist die 5-Modell-Teilmenge `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`. Übergeben Sie die vollständige 8-Modell-Liste (oben), um den zentralen Vergleich der Veröffentlichung zu reproduzieren. Weitere Standardwerte: `--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`.
- **05** Optionen: `--no-ogle` (nur Gaia), `--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (Standard `10dim`), `--verify-folds`.
- **07** Optionen: `--device cuda` sowie `--skip-training`/`--skip-ablation` usw., um Teilmengen auszuführen.
- **08** Optionen: `--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`.

### Schritt → Skript → Ausgabe

| Schritt | Skript | Wichtige Eingaben | Wichtige Ausgaben |
|------|--------|------------|-------------|
| 1. Gaia herunterladen | `scripts/01_download_gaia.py` | Gaia-TAP-Archiv | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. OGLE herunterladen | `scripts/02_download_ogle.py` | OGLE-IV-Archiv | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. Merkmale extrahieren | `scripts/03_extract_features.py` | rohe Lichtkurven | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. Merkmale validieren | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, Qualitätsbericht |
| 5. Datensatz aufbauen | `scripts/05_build_dataset.py` | validierte Merkmale | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. Modelle trainieren | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | CV-Ergebnisse + Checkpoints pro Modell, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. Experimente ausführen | `scripts/07_run_experiments.py` | verarbeitete Daten + trainierte Modelle | Ablation / Intervention / surveyübergreifend / Lernkurve als JSON |
| 8. Abbildungen erzeugen | `scripts/08_generate_figures.py` | Ergebnis-JSON | Abbildungen der Veröffentlichung (PDF) + LaTeX-Tabellen |

### Konfiguration

Alle Läufe werden durch **`configs/default.yaml`** gesteuert (geladen über OmegaConf; übergeben Sie `--config <file>` an ein beliebiges Skript, um sie zu überschreiben). Sie ist die maßgebliche Quelle für: Zufalls-Seed (`project.random_seed: 42`); Aufteilungsschema (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`, stratifiziert, StandardScaler); Download-Ziele pro Klasse (`var_types.*`); Parameter der Merkmalsextraktion (`features.*` — Periodensuche, Fourier-Harmonische, Alias-Erkennung bei der Gaia-Präzessionsperiode von 63 Tagen usw.); Trainings-Hyperparameter (`training.*` — Batch 256, lr 1e-3, max. 200 Epochen, Patience 15, Cosine-Warm-Restart-Zeitplan); Architektur pro Modell (`models.*`); und Experiment-Gitter (`experiments.*`). Zwei zusätzliche Konfigurationen existieren: `configs/feature_config.yaml` und `configs/gaia_queries.yaml`.

### Hinweise zu Reproduzierbarkeit / Datenumfang

- **Seed & Folds.** `random_seed=42` und die 5-fache stratifizierte CV sind in `configs/default.yaml` festgelegt; `05_build_dataset.py` schreibt deterministische Fold-Indizes nach `data/processed/cv_folds.pkl`.
- **Zentraler Datensatz vs. mitgelieferter Split.** Die zentralen Zahlen der Veröffentlichung (HardCBM 94,41 % ± 0,36 % Genauigkeit) stammen aus der vollständigen ausbalancierten Gaia-Matrix mit **18.000 Quellen** (3.000/Klasse; 15.300 CV + 2.700 Test). Die *in diesem Baum* mitgelieferte `data/processed/cv_pool.parquet` ist ein **reduzierter Demo-Split** (2.550 Zeilen, 425/Klasse); ein Lauf darauf reproduziert die zentrale Genauigkeit nicht. Das erneute Ausführen der Datenstufen baut den vollständigen Datensatz aus den Archiven neu auf; die vollständige Merkmalsmatrix ist außerdem als `data/real/gaia_all_features.parquet` verfügbar (18.000 Zeilen).
- **OGLE auf Abruf.** `data/raw/ogle/` wird absichtlich leer ausgeliefert — Schritt 02 lädt OGLE-Lichtkurven zur Laufzeit herunter (Netzwerk für den surveyübergreifenden Test erforderlich).

> **Hinweis zum Makefile.** Die `make`-Ziele spiegeln die obigen Phasen wider (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`). Die committeten Ziele `data:` und `train:` sind leicht aus dem Takt mit den aktuellen Skripten geraten — `make data` verweist auf ältere Skriptnamen und `make train` lässt das erforderliche Argument `--data_path` aus —, daher sind die expliziten `python scripts/0N_*.py`-Befehle oben die maßgeblichen, funktionierenden Aufrufe.

## Datensatz

Das Repository liefert einen **Merkmalsdatensatz veränderlicher Sterne aus Gaia DR3** unter `data/`, organisiert als reproduzierbare Kaskade von der rohen Photometrie bis zu kreuzvalidierungsbereiten, normalisierten Splits. Alle tabellarischen Daten werden als Apache Parquet gespeichert; die Indizes der Kreuzvalidierungs-Folds und der angepasste Scaler liegen als Pickle (`.pkl`) vor; die Label-Zuordnung als JSON.

Der Datensatz umfasst **6 Klassen veränderlicher Sterne**, beschrieben durch **12 physikalisch bedeutsame Konzepte** (siehe die Tabelle [Konzepte](#konzepte)). Die zentrale Studientabelle enthält **18.000 Gaia-DR3-Quellen**, ausbalanciert auf **3.000 pro Klasse** (`data/real/gaia_all_features.parquet`, in physikalischen/unskalierten Einheiten), aufgeteilt in einen Kreuzvalidierungs-Pool mit 15.300 Quellen für die 5-fache CV und einen Hold-out-Testsatz mit 2.700 Quellen.

Organisation unter `data/`:

| Verzeichnis | Inhalt |
|---|---|
| `raw/gaia/epoch_photometry/` | Gaia-DR3-Lichtkurven im G-Band pro Quelle, eine Parquet-Datei pro `<source_id>` (Spalten `time, mag, mag_err`). |
| `raw/gaia/metadata/` | Quellmetadaten pro Klasse und kombiniert (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`). |
| `raw/ogle/` | Absichtlich leer; OGLE-Lichtkurven für die surveyübergreifende Auswertung werden auf Abruf heruntergeladen. |
| `interim/` | Extrahierte Merkmale vor der Aufteilung (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`). |
| `processed/` | Kreuzvalidierungsbereite, StandardScaler-normalisierte (z-Score) Splits: `cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE, außerhalb der Domäne), plus `cv_folds.pkl` (5-fache StratifiedKFold-Indizes), `scaler.pkl` und `label_mapping.json`. |
| `expanded/` | Größere augmentierte Variante (`gaia_expanded_features.parquet`, 30.000 Zeilen). |
| `real/` | Die zentrale Merkmalstabelle mit 18.000 Quellen in physikalischen Einheiten (`gaia_all_features.parquet`) plus Rohmetadaten. |

Jede Merkmalszeile trägt Identifikator-/Label-/Qualitätsspalten (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`), gefolgt von den 12 Konzeptspalten. Der globale `StandardScaler` wird ausschließlich auf dem Kreuzvalidierungs-Pool angepasst und auf beide Testsätze angewendet; `period_snr` wird zum Skalierungszeitpunkt mit dem Median imputiert.

> **Hinweis.** Die in diesem Baum mitgelieferten Parquet-Dateien unter `data/processed/` sind ein reduzierter Demonstrations-Split (`cv_pool` 2.550 + `test_in_domain` 450 = 3.000 Zeilen; surveyübergreifender Test 1.200). Die vollständige Matrix mit 18.000 Quellen, die für die veröffentlichten zentralen Ergebnisse verwendet wurde, entspricht `data/real/gaia_all_features.parquet`.

## Modelle

Acht Modellvarianten werden verglichen: sechs neuronale Netze und zwei klassische Baum-Baselines. (Das neuronale Paket liefert zusätzlich eine `EndToEndHardCBM`-Variante (`e2e_hard_cbm`) — einen 1D-CNN-Konzept-Encoder über phasengefaltete Lichtkurven — über die acht zentralen Varianten unten hinaus.)

| Modell | Registry-Schlüssel | Beschreibung |
|---|---|---|
| **HardCBM** | `hard_cbm` | Hard Concept Bottleneck Model; die Eingabemerkmale dienen als 12-Konzept-Bottleneck, der einen MLP-Prädiktor speist (`12→64→32→6`). Das interpretierbare Referenzmodell. |
| **HardCBM-Linear** | `hard_cbm_linear` | Hard CBM mit einem einzelnen `Linear(12, 6)`-Prädiktor (78 Parameter); die Gewichte lassen sich direkt als Konzept-zu-Klasse-Beiträge ablesen. Maximal interpretierbar. |
| **HardCBM-Cal** | `hard_cbm_cal` | Kalibriertes Hard CBM mit 12 unabhängigen Kalibrierungsköpfen, die die extrahierten Konzepte vor einem MLP-Prädiktor entrauschen; primäre Architektur für Interventionsexperimente. |
| **SoftCBM** | `soft_cbm` | Soft CBM mit kontinuierlichen Embeddings pro Konzept (48-dimensionaler Bottleneck); der breitere Bottleneck tauscht Interpretierbarkeit gegen höhere Genauigkeit. |
| **CEM** | `cem` | Concept Embedding Model (Espinosa Zarlenga et al. 2022); jedes Konzept ist ein positiv/negativ-Embedding-Paar, gemischt durch seine Aktivierung. |
| **MLP** | `mlp` | Einfache Multilayer-Perzeptron-Baseline (`12→128→64→6`), ohne Bottleneck; misst die Genauigkeitskosten des Bottlenecks. |
| **Random Forest** | `rf` | Klassische Black-Box-Baseline (500 Bäume, ausbalancierte Klassengewichte), abgestimmt auf den offiziellen Gaia-DR3-Klassifikator (Rimoldini et al. 2023). |
| **XGBoost** | `xgb` | Gradient-Boosted-Tree-Black-Box-Baseline; SHAP-Werte werden zum Vergleich mit der Konzeptwichtigkeit des CBM berechnet. |

### Zentrale Metriken

5-fache Kreuzvalidierung (`RANDOM_SEED=42`), Makro-F1 als primäre Metrik, auf dem vollständigen Gaia-DR3-Datensatz mit 18.000 Quellen (15.300 CV + 2.700 Test; 3.000 pro Klasse). Dies sind die veröffentlichten Werte aus der *Astronomy & Astrophysics*-Veröffentlichung (Tabelle 2 der Veröffentlichung):

| Modell | Genauigkeit (%) | Makro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

Die Lücke zwischen Interpretierbarkeit und Genauigkeit ist unmittelbar sichtbar: Das transparente HardCBM (94,4 %) liegt um rund 5 Prozentpunkte hinter den Black-Box-Baum-Baselines (≈ 99,8 %) zurück, was genau der Preis dafür ist, jede Entscheidung durch einen physikalischen Bottleneck aus 12 Konzepten zu zwingen.

## Konzepte

Die 12 physikalischen Konzepte, die den Bottleneck bilden, in fester Reihenfolge (maßgeblich definiert in `cbm_variable_stars/shared/constants.py`).

| # | Konzept | Einheit / Bereich | Bedeutung |
|---|---|---|---|
| 1 | `period` | Tage | Primäre Pulsations-/Variabilitätsperiode. |
| 2 | `amplitude` | mag | Spitze-zu-Spitze-Amplitude der Lichtkurve. |
| 3 | `rise_fraction` | dimensionslos, [0, 1] | Anteil des Zyklus, der im Helligkeitsanstieg verbracht wird. |
| 4 | `R21` | dimensionsloses Verhältnis | Fourier-Amplitudenverhältnis A2/A1. |
| 5 | `R31` | dimensionsloses Verhältnis | Fourier-Amplitudenverhältnis A3/A1. |
| 6 | `phi21` | Radiant, [0, 2π) | Fourier-Phasendifferenz φ2 − 2φ1. |
| 7 | `skewness` | dimensionslos | Schiefe der Magnitudenverteilung. |
| 8 | `kurtosis` | dimensionslos | Exzess-(Fisher-)Kurtosis der Magnitudenverteilung. |
| 9 | `stetson_K` | dimensionslos | Stetson-K-Variabilitätsindex. |
| 10 | `period_snr` | dimensionslos | Periodensignifikanz, −log₁₀(Falschalarmwahrscheinlichkeit). |
| 11 | `color_bp_rp` | mag | Gaia-BP − RP-Farbindex. |
| 12 | `mean_mag` | mag | Mittlere Gaia-Magnitude im G-Band. |

Die sechs Klassen sind `RRAB` (RR Lyrae, Grundmodus), `RRC` (RR Lyrae, erster Oberton), `DCEP` (klassischer Cepheid), `DSCT_SXPHE` (Delta Scuti / SX Phoenicis), `ECL` (Bedeckungsveränderlicher) und `MIRA_SR` (langperiodischer Veränderlicher vom Typ Mira / halbregelmäßig), in dieser Reihenfolge mit 0–5 indiziert.

## Zitation

Wenn Sie diese Arbeit verwenden, zitieren Sie bitte die begleitende Veröffentlichung:

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

Klartext: Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990.

## Lizenz

Dieses Repository liefert derzeit keine explizite Open-Source-Lizenzdatei mit, daher sollte keine allgemeine Gewährung von Rechten zur Wiederverwendung, Weiterverbreitung oder Modifikation angenommen werden. Wenn Sie den Code oder die Daten über die obigen Zitationsbedingungen hinaus wiederverwenden möchten, wenden Sie sich bitte an den Autor (Chenglong Yin, Sofia University), um eine Genehmigung zu vereinbaren. Eine formelle Lizenz kann in einer zukünftigen Version hinzugefügt werden.

## Links

- **Veröffentlichung (Astronomy & Astrophysics):** <https://doi.org/10.1051/0004-6361/202659990>
- **Interaktiver Web-Begleiter:** <https://cbm-variable-stars.yinchenglong.com>
