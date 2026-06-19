<div align="center">

[English](README.md) · [中文](README.zh-CN.md) · [Čeština](README.cs.md) · [Български](README.bg.md) · **Español** · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

# CBM Variable Stars

**Clasificación interpretable de estrellas variables con modelos de cuello de botella conceptual (Concept Bottleneck Models): cada predicción se rastrea a través de 12 conceptos estelares con significado físico.**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## Resumen

`cbm_variable_stars` es, hasta donde sabemos, la primera aplicación de **modelos de cuello de botella conceptual (Concept Bottleneck Models, CBM)** a la clasificación astronómica de estrellas variables. Clasifica estrellas variables a partir de la fotometría de Gaia DR3 en **6 clases** — RR Lyrae fundamental y de sobretono, Cefeidas clásicas, Delta Scuti / SX Phoenicis, binarias eclipsantes y variables Mira / semirregulares de período largo — encaminando cada decisión a través de **12 conceptos interpretables e inspeccionables por un astrónomo** en lugar de un espacio de características opaco.

Con este paquete obtienes:

- una canalización completa y reproducible desde las características de Gaia DR3 (y el cruce de catálogos OGLE) hasta clasificadores entrenados;
- **8 variantes de modelo principales** que abarcan todo el espectro de la interpretabilidad — desde un CBM lineal totalmente transparente de 78 parámetros hasta líneas base de caja negra Random Forest y XGBoost;
- validación cruzada de 5 particiones (5-fold), ablación, intervención y experimentos entre catálogos, con las métricas, pruebas de significancia, figuras y tablas LaTeX que respaldan el artículo asociado.

El artículo asociado está publicado en *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), y hay una demostración web interactiva disponible en <https://cbm-variable-stars.yinchenglong.com>.

## Por qué modelos de cuello de botella conceptual

La mayoría de los clasificadores profundos asignan las entradas crudas directamente a una etiqueta, sin dejar ningún registro inspeccionable de *por qué* una estrella dada fue asignada a una clase dada. En cambio, un modelo de cuello de botella conceptual obliga a que cada predicción pase a través de una capa estrecha de **conceptos** con significado para los humanos:

```
raw photometric features  →  12 physical concepts  →  class
```

Dado que toda la información de clasificación debe fluir a través del cuello de botella conceptual de 12 dimensiones, cada predicción es rastreable a magnitudes físicas que un astrónomo puede leer y verificar — período, amplitud, parámetros de forma de Fourier, color, etc. También hace que el modelo sea **interactivo**: puedes sobrescribir un concepto (por ejemplo, suministrar un período corregido) y observar cómo responde la clase predicha. Esta transparencia tiene un costo medible en exactitud bruta, y cuantificar ese compromiso entre interpretabilidad y rendimiento es un tema central de este trabajo.

## Aspectos destacados / resultados clave

- **Primer CBM para clasificación de estrellas variables** — un cuello de botella de 12 conceptos con fundamento físico sobre 6 clases de estrellas variables de Gaia DR3.
- **Interpretable por construcción** — cada predicción es rastreable a 12 conceptos físicos con nombre, y los conceptos pueden sobrescribirse en tiempo de inferencia para intervenir sobre una predicción.
- **Exactitud sólida y honesta** — el Hard CBM alcanza una **exactitud del 94.41% ± 0.36%** (macro-F1 94.37% ± 0.38%, MCC 0.933) bajo validación cruzada de 5 particiones sobre el conjunto de datos principal de Gaia DR3 de 18 000 fuentes (3 000 ejemplos equilibrados por clase).
- **El costo de la interpretabilidad, medido** — las líneas base de caja negra (Random Forest y XGBoost ≈ 99.8%) superan al transparente Hard CBM en aproximadamente 5 puntos porcentuales, aislando el precio de un cuello de botella conceptual impuesto.
- **8 modelos a lo largo del espectro de transparencia** — Hard CBM, Hard CBM-Linear (78 parámetros), Hard CBM-Calibrated, Soft CBM, CEM, una línea base MLP, Random Forest y XGBoost.
- **Probado entre catálogos** — evaluado bajo el cambio de dominio Gaia → OGLE, junto con estudios de ablación y de intervención sobre conceptos.

## Instalación

**Requisitos previos**

- **Python ≥ 3.10** (declarado mediante `python_requires=">=3.10"` en `setup.py`; verificado como correcto en CPython 3.10–3.13).
- **No** se requiere una cadena de herramientas C/C++ — todas las dependencias se distribuyen como ruedas binarias (wheels).
- **No se requiere GPU.** Los modelos son pequeños (HardCBM ≈ 3K parámetros, MLP ≈ 10K parámetros) y la configuración de entrenamiento por defecto (tamaño de lote 256, ≤ 200 épocas con parada temprana) se ejecuta cómodamente en CPU. Una GPU CUDA solo acelera las ejecuciones completas de ~18 000 fuentes.
- Se necesita acceso a internet **solo** para los pasos opcionales de descarga de datos (consultas a los archivos Gaia/OGLE mediante `astroquery`/`pyvo`); el entrenamiento sobre un conjunto de datos ya construido funciona completamente sin conexión.

**Clonar**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**Instalar**

```bash
# (recommended) create + activate a Python >=3.10 virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editable install (equivalent to `make install`)
pip install -e .
```

Como alternativa, instala el conjunto completo de tiempo de ejecución con versiones fijadas:

```bash
pip install -r requirements.txt
```

**Extras opcionales** (declarados en `setup.py`):

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**Dependencias principales** (incorporadas automáticamente): `numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`.

> Para una compilación CUDA explícita, instala la rueda de PyTorch correspondiente **antes** de `pip install -e .`, p. ej. `pip install torch --index-url https://download.pytorch.org/whl/cu121`. El proyecto no fija una versión de CUDA.

## Inicio rápido

El ejemplo útil más pequeño: carga las características conceptuales ya estandarizadas que se incluyen y ejecuta una pasada hacia adelante de HardCBM para obtener los 12 valores de los conceptos y una predicción de 6 clases.

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

Esto carga datos reales, instancia un modelo a través del registro y ejercita la interfaz unificada `forward(x) -> {concepts, logits, probabilities}` compartida por cada variante de modelo. Cambia `"hard_cbm"` por cualquier clave del registro (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`) para probar otras arquitecturas.

Para **entrenar** en lugar de usar pesos aleatorios, envuelve las características en el conjunto de datos e impulsa la orquestación del entrenamiento:

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **Nota sobre el tamaño del conjunto de datos.** Los archivos parquet de `data/processed/` que se incluyen aquí son una partición reducida (`cv_pool` 2 550 + `test_in_domain` 450 + entre catálogos 1 200 filas). Las cifras principales del artículo (HardCBM 94.41 ± 0.36% de exactitud, validación cruzada de 5 particiones) provienen de la tabla completa de **18 000 fuentes** (`data/real/gaia_all_features.parquet`, 3 000 por clase), dividida en 15 300 para CV + 2 700 para test. Consulta las secciones [Reproducción de los resultados](#reproducción-de-los-resultados-canalización-completa) y [Conjunto de datos](#conjunto-de-datos) para saber cómo reconstruir el conjunto de datos completo.

## Estructura del repositorio

Árbol anotado del código y los datos incluidos en este repositorio (el paquete de importación es `cbm_variable_stars`).

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

> Las 6 clases son `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR`; los 12 conceptos son `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` — ambos definidos con autoridad en `cbm_variable_stars/shared/constants.py`.

## Reproducción de los resultados (canalización completa)

La canalización de extremo a extremo va desde la fotometría cruda de Gaia DR3 (y OGLE) hasta modelos entrenados, métricas validadas de forma cruzada y las figuras/tablas del artículo. Cada etapa es un script numerado bajo `scripts/`, impulsado por un único archivo de configuración (`configs/default.yaml`). Ejecutar la canalización regenera un directorio local `results/`; los modelos entrenados publicados, las métricas y las figuras residen en el artículo de *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)) en lugar de en este árbol de código.

### Requisitos previos

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

Las etapas 01–02 realizan llamadas de red a los archivos Gaia TAP y OGLE, por lo que necesitan acceso a internet. El entrenamiento (06) se ejecuta en CPU por defecto; pasa `--device cuda` para una GPU.

### Paso a paso (los scripts reales en disco)

Todos los comandos se ejecutan desde la raíz del repositorio; cada script lee `configs/default.yaml` por defecto (anúlalo con `--config`).

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

Notas sobre las opciones más usadas:

- **06** requiere `--data_path`; su lista `--models` por defecto es el subconjunto de 5 modelos `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`. Pasa la lista completa de 8 modelos (arriba) para reproducir la comparación principal del artículo. Otros valores por defecto: `--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`.
- **05** opciones: `--no-ogle` (solo Gaia), `--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (por defecto `10dim`), `--verify-folds`.
- **07** opciones: `--device cuda`, y `--skip-training`/`--skip-ablation`, etc., para ejecutar subconjuntos.
- **08** opciones: `--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`.

### Paso → script → salida

| Paso | Script | Entradas clave | Salidas clave |
|------|--------|------------|-------------|
| 1. Descargar Gaia | `scripts/01_download_gaia.py` | Archivo Gaia TAP | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. Descargar OGLE | `scripts/02_download_ogle.py` | Archivo OGLE-IV | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. Extraer características | `scripts/03_extract_features.py` | curvas de luz crudas | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. Validar características | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, informe de calidad |
| 5. Construir conjunto de datos | `scripts/05_build_dataset.py` | características validadas | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. Entrenar modelos | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | resultados de CV por modelo + puntos de control, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. Ejecutar experimentos | `scripts/07_run_experiments.py` | datos procesados + modelos entrenados | JSON de ablación / intervención / entre catálogos / curva de aprendizaje |
| 8. Generar figuras | `scripts/08_generate_figures.py` | JSON de resultados | figuras del artículo (PDF) + tablas LaTeX |

### Configuración

Todas las ejecuciones se impulsan mediante **`configs/default.yaml`** (cargado a través de OmegaConf; pasa `--config <file>` a cualquier script para anularlo). Es la única fuente de verdad para: semilla aleatoria (`project.random_seed: 42`); esquema de partición (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`, estratificado, StandardScaler); objetivos de descarga por clase (`var_types.*`); parámetros de extracción de características (`features.*` — búsqueda de período, armónicos de Fourier, detección de alias en el período de precesión de Gaia de 63 días, etc.); hiperparámetros de entrenamiento (`training.*` — lote 256, lr 1e-3, máximo 200 épocas, paciencia 15, programación de reinicio cálido con coseno); arquitectura por modelo (`models.*`); y rejillas de experimentos (`experiments.*`). Existen dos configuraciones auxiliares: `configs/feature_config.yaml` y `configs/gaia_queries.yaml`.

### Notas de reproducibilidad / alcance de los datos

- **Semilla y particiones.** `random_seed=42` y la CV estratificada de 5 particiones están fijados en `configs/default.yaml`; `05_build_dataset.py` escribe índices de partición deterministas en `data/processed/cv_folds.pkl`.
- **Conjunto de datos principal vs. partición incluida.** Las cifras principales del artículo (HardCBM 94.41% ± 0.36% de exactitud) provienen de la matriz Gaia equilibrada completa de **18 000 fuentes** (3 000/clase; 15 300 CV + 2 700 test). El `data/processed/cv_pool.parquet` incluido *en este árbol* es una **partición de demostración reducida** (2 550 filas, 425/clase); ejecutar sobre ella no reproducirá la exactitud principal. Volver a ejecutar las etapas de datos reconstruye el conjunto de datos completo a partir de los archivos; la matriz de características completa también está disponible como `data/real/gaia_all_features.parquet` (18 000 filas).
- **OGLE bajo demanda.** `data/raw/ogle/` se distribuye vacío por diseño — el paso 02 descarga las curvas de luz de OGLE en tiempo de ejecución (se requiere red para el test entre catálogos).

> **Nota sobre el Makefile.** Los objetivos de `make` reflejan las fases anteriores (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`). Los objetivos `data:` y `train:` confirmados están ligeramente desfasados respecto a los scripts actuales — `make data` referencia nombres de script más antiguos y `make train` omite el argumento requerido `--data_path` — por lo que los comandos explícitos `python scripts/0N_*.py` de arriba son las invocaciones canónicas y funcionales.

## Conjunto de datos

El repositorio incluye un **conjunto de datos de características de estrellas variables de Gaia DR3** bajo `data/`, organizado como una cascada reproducible desde la fotometría cruda hasta particiones normalizadas y listas para validación cruzada. Todos los datos tabulares se almacenan como Apache Parquet; los índices de partición de validación cruzada y el escalador ajustado son pickle (`.pkl`); el mapa de etiquetas es JSON.

El conjunto de datos abarca **6 clases de estrellas variables** descritas por **12 conceptos con significado físico** (consulta la tabla de [Conceptos](#conceptos)). La tabla principal del estudio contiene **18 000 fuentes de Gaia DR3** equilibradas a **3 000 por clase** (`data/real/gaia_all_features.parquet`, en unidades físicas/sin escalar), dividida en un grupo de validación cruzada de 5 particiones de 15 300 fuentes y un conjunto de test reservado de 2 700 fuentes.

Organización bajo `data/`:

| Directorio | Contenidos |
|---|---|
| `raw/gaia/epoch_photometry/` | Curvas de luz en banda G de Gaia DR3 por fuente, un Parquet por `<source_id>` (columnas `time, mag, mag_err`). |
| `raw/gaia/metadata/` | Metadatos de fuentes por clase y combinados (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`). |
| `raw/ogle/` | Vacío por diseño; las curvas de luz de OGLE para el test entre catálogos se descargan bajo demanda. |
| `interim/` | Características extraídas antes de la partición (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`). |
| `processed/` | Particiones listas para validación cruzada y normalizadas con StandardScaler (z-score): `cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE, fuera de dominio), más `cv_folds.pkl` (índices de 5 particiones StratifiedKFold), `scaler.pkl` y `label_mapping.json`. |
| `expanded/` | Variante aumentada más grande (`gaia_expanded_features.parquet`, 30 000 filas). |
| `real/` | La tabla principal de características de 18 000 fuentes en unidades físicas (`gaia_all_features.parquet`) más metadatos crudos. |

Cada fila de características lleva columnas de identificador/etiqueta/calidad (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`) seguidas de las 12 columnas de conceptos. El `StandardScaler` global se ajusta únicamente sobre el grupo de validación cruzada y se aplica a ambos conjuntos de test; `period_snr` se imputa con la mediana en el momento del escalado.

> **Nota.** Los archivos parquet de `data/processed/` incluidos en este árbol son una partición de demostración reducida (`cv_pool` 2 550 + `test_in_domain` 450 = 3 000 filas; test entre catálogos 1 200). La matriz completa de 18 000 fuentes usada para los resultados principales publicados corresponde a `data/real/gaia_all_features.parquet`.

## Modelos

Se comparan ocho variantes de modelo: seis redes neuronales y dos líneas base clásicas de árboles. (El paquete neuronal incluye además una variante `EndToEndHardCBM` (`e2e_hard_cbm`) — un codificador de conceptos 1D-CNN sobre curvas de luz plegadas en fase — más allá de las ocho variantes principales que se listan a continuación.)

| Modelo | Clave del registro | Descripción |
|---|---|---|
| **HardCBM** | `hard_cbm` | Modelo de cuello de botella conceptual duro (Hard Concept Bottleneck Model); las características de entrada sirven como cuello de botella de 12 conceptos que alimenta un predictor MLP (`12→64→32→6`). El modelo interpretable de referencia. |
| **HardCBM-Linear** | `hard_cbm_linear` | Hard CBM con un único predictor `Linear(12, 6)` (78 parámetros); los pesos se leen directamente como contribuciones de concepto a clase. Máximamente interpretable. |
| **HardCBM-Cal** | `hard_cbm_cal` | Hard CBM calibrado con 12 cabezas de calibración independientes que eliminan el ruido de los conceptos extraídos antes de un predictor MLP; arquitectura principal para los experimentos de intervención. |
| **SoftCBM** | `soft_cbm` | Soft CBM con incrustaciones continuas por concepto (cuello de botella de 48 dimensiones); el cuello de botella más ancho intercambia interpretabilidad por mayor exactitud. |
| **CEM** | `cem` | Modelo de incrustación de conceptos (Concept Embedding Model; Espinosa Zarlenga et al. 2022); cada concepto es un par de incrustaciones positiva/negativa mezcladas por su activación. |
| **MLP** | `mlp` | Línea base de perceptrón multicapa simple (`12→128→64→6`), sin cuello de botella; mide el costo en exactitud del cuello de botella. |
| **Random Forest** | `rf` | Línea base clásica de caja negra (500 árboles, pesos de clase equilibrados), alineada con el clasificador oficial de Gaia DR3 (Rimoldini et al. 2023). |
| **XGBoost** | `xgb` | Línea base de caja negra de árboles potenciados por gradiente; se calculan valores SHAP para comparar con la importancia de conceptos del CBM. |

### Métricas principales

Validación cruzada de 5 particiones (`RANDOM_SEED=42`), con macro-F1 como métrica primaria, sobre el conjunto de datos completo de Gaia DR3 de 18 000 fuentes (15 300 CV + 2 700 test; 3 000 por clase). Estos son los valores publicados en el artículo de *Astronomy & Astrophysics* (Tabla 2 del artículo):

| Modelo | Exactitud (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

La brecha entre interpretabilidad y exactitud es visible directamente: el transparente HardCBM (94.4%) queda por detrás de las líneas base de árboles de caja negra (≈ 99.8%) en aproximadamente 5 puntos porcentuales, que es precisamente el costo de forzar cada decisión a través de un cuello de botella físico de 12 conceptos.

## Conceptos

Los 12 conceptos físicos que forman el cuello de botella, en orden fijo (definidos con autoridad en `cbm_variable_stars/shared/constants.py`).

| # | Concepto | Unidad / Rango | Significado |
|---|---|---|---|
| 1 | `period` | días | Período primario de pulsación/variabilidad. |
| 2 | `amplitude` | mag | Amplitud pico a pico de la curva de luz. |
| 3 | `rise_fraction` | adimensional, [0, 1] | Fracción del ciclo dedicada al aumento de brillo. |
| 4 | `R21` | razón adimensional | Razón de amplitudes de Fourier A2/A1. |
| 5 | `R31` | razón adimensional | Razón de amplitudes de Fourier A3/A1. |
| 6 | `phi21` | radianes, [0, 2π) | Diferencia de fase de Fourier φ2 − 2φ1. |
| 7 | `skewness` | adimensional | Asimetría de la distribución de magnitudes. |
| 8 | `kurtosis` | adimensional | Curtosis (de Fisher) en exceso de la distribución de magnitudes. |
| 9 | `stetson_K` | adimensional | Índice de variabilidad K de Stetson. |
| 10 | `period_snr` | adimensional | Significancia del período, −log₁₀(probabilidad de falsa alarma). |
| 11 | `color_bp_rp` | mag | Índice de color BP − RP de Gaia. |
| 12 | `mean_mag` | mag | Magnitud media en banda G de Gaia. |

Las seis clases son `RRAB` (RR Lyrae, modo fundamental), `RRC` (RR Lyrae, primer sobretono), `DCEP` (Cefeida clásica), `DSCT_SXPHE` (Delta Scuti / SX Phoenicis), `ECL` (binaria eclipsante) y `MIRA_SR` (Mira / variable semirregular de período largo), indexadas de 0 a 5 en ese orden.

## Cómo citar

Si utilizas este trabajo, por favor cita el artículo asociado:

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

Texto plano: Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990.

## Licencia

Este repositorio no incluye actualmente un archivo de licencia de código abierto explícito, por lo que no debe asumirse ninguna concesión general de derechos de reutilización, redistribución o modificación. Si deseas reutilizar el código o los datos más allá de los términos de citación anteriores, por favor contacta con el autor (Chenglong Yin, Universidad de Sofía) para acordar el permiso. Es posible que se añada una licencia formal en una versión futura.

## Enlaces

- **Artículo (Astronomy & Astrophysics):** <https://doi.org/10.1051/0004-6361/202659990>
- **Compañero web interactivo:** <https://cbm-variable-stars.yinchenglong.com>
