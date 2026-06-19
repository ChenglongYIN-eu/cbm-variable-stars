<div align="center">

[English](README.md) · [中文](README.zh-CN.md) · [Čeština](README.cs.md) · **Български** · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

# CBM Variable Stars

**Интерпретируема класификация на променливи звезди с модели с концептуално гърло (Concept Bottleneck Models) — всяко предсказание е проследимо през 12 физически смислени звездни концепции.**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## Преглед

`cbm_variable_stars` е, доколкото ни е известно, първото приложение на **модели с концептуално гърло (Concept Bottleneck Models, CBMs)** към астрономическата класификация на променливи звезди. Той класифицира променливи звезди от фотометрия на Gaia DR3 в **6 класа** — RR Lyrae в основна и обертонна мода, класически цефеиди, Delta Scuti / SX Phoenicis, затъмнително-двойни и Mira / полуправилни дълъгопериодични променливи — като насочва всяко решение през **12 интерпретируеми, проверими от астроном концепции** вместо през непрозрачно пространство от признаци.

С този пакет получавате:

- цялостен, възпроизводим конвейер от признаци на Gaia DR3 (и кръстосано-обзорни OGLE) до обучени класификатори;
- **8 водещи варианта на модели**, обхващащи спектъра на интерпретируемостта — от напълно прозрачен линеен CBM със 78 параметъра до базови модели от тип „черна кутия“ Random Forest и XGBoost;
- 5-кратна кръстосана валидация, аблация, интервенция и кръстосано-обзорни експерименти, заедно с метриките, тестовете за значимост, фигурите и LaTeX таблиците, които подкрепят придружаващата статия.

Придружаващата статия е публикувана в *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), а интерактивно уеб демо е достъпно на <https://cbm-variable-stars.yinchenglong.com>.

## Защо модели с концептуално гърло

Повечето дълбоки класификатори преобразуват суровите входни данни директно в етикет, без да оставят проверимо обяснение *защо* дадена звезда е отнесена към даден клас. Вместо това моделът с концептуално гърло принуждава всяко предсказание да премине през тесен слой от смислени за човека **концепции**:

```
raw photometric features  →  12 physical concepts  →  class
```

Тъй като цялата класификационна информация трябва да премине през 12-измерното концептуално гърло, всяко предсказание е проследимо до физически величини, които астрономът може да прочете и провери — период, амплитуда, параметри на формата по Фурие, цвят и т.н. Това прави модела и **интерактивен**: можете да презапишете дадена концепция (например да зададете коригиран период) и да наблюдавате как реагира предсказаният клас. Тази прозрачност има измерима цена в чистата точност, а количественото определяне на този компромис между интерпретируемост и производителност е централна тема на тази работа.

## Акценти / основни резултати

- **Първият CBM за класификация на променливи звезди** — физически обосновано концептуално гърло от 12 концепции върху 6 класа променливи звезди от Gaia DR3.
- **Интерпретируем по конструкция** — всяко предсказание е проследимо до 12 именувани физически концепции, а концепциите могат да бъдат презаписвани по време на извод, за да се извърши интервенция върху предсказание.
- **Силна, честна точност** — Hard CBM достига **94.41% ± 0.36% точност** (macro-F1 94.37% ± 0.38%, MCC 0.933) при 5-кратна кръстосана валидация върху водещия набор от данни от Gaia DR3 с 18 000 източника (3 000 балансирани примера на клас).
- **Цената на интерпретируемостта, измерена** — базовите модели тип „черна кутия“ (Random Forest и XGBoost ≈ 99.8%) превъзхождат прозрачния Hard CBM с около 5 процентни пункта, изолирайки цената на наложеното концептуално гърло.
- **8 модела по спектъра на прозрачността** — Hard CBM, Hard CBM-Linear (78 параметъра), Hard CBM-Calibrated, Soft CBM, CEM, базов MLP, Random Forest и XGBoost.
- **Тестван кръстосано между обзори** — оценен при доменно изместване Gaia → OGLE, заедно с проучвания за аблация и концептуална интервенция.

## Инсталация

**Предпоставки**

- **Python ≥ 3.10** (деклариран чрез `python_requires=">=3.10"` в `setup.py`; проверено добре функциониращ на CPython 3.10–3.13).
- C/C++ инструментариум **не** е необходим — всички зависимости се доставят като двоични wheel пакети.
- **Не е необходим GPU.** Моделите са малки (HardCBM ≈ 3K параметъра, MLP ≈ 10K параметъра), а конфигурацията за обучение по подразбиране (размер на партидата 256, ≤ 200 епохи с ранно спиране) работи удобно на CPU. CUDA GPU само ускорява пълните прогони с ~18 000 източника.
- Достъп до интернет е необходим **само** за незадължителните стъпки за изтегляне на данни (заявки към архивите на Gaia/OGLE чрез `astroquery`/`pyvo`); обучението върху вече изграден набор от данни работи напълно офлайн.

**Клониране**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**Инсталиране**

```bash
# (recommended) create + activate a Python >=3.10 virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editable install (equivalent to `make install`)
pip install -e .
```

Като алтернатива, инсталирайте пълния фиксиран набор от изпълнителни зависимости:

```bash
pip install -r requirements.txt
```

**Незадължителни допълнения** (декларирани в `setup.py`):

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**Основни зависимости** (изтеглят се автоматично): `numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`.

> За изрично CUDA изграждане инсталирайте съответстващия PyTorch wheel **преди** `pip install -e .`, например `pip install torch --index-url https://download.pytorch.org/whl/cu121`. Проектът не фиксира конкретна версия на CUDA.

## Бърз старт

Най-малкият полезен пример: зареждане на доставените, вече стандартизирани концептуални признаци и изпълнение на права итерация (forward pass) на HardCBM, за да се получат 12-те стойности на концепциите и предсказание за 6 класа.

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

Това зарежда реални данни, инстанцира модел през регистъра и упражнява унифицирания интерфейс `forward(x) -> {concepts, logits, probabilities}`, споделен от всеки вариант на модела. Заменете `"hard_cbm"` с който и да е ключ от регистъра (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`), за да изпробвате други архитектури.

За да **обучите** вместо да използвате случайни тегла, обвийте признаците в набора от данни и задвижете оркестрацията на обучението:

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **Бележка за размера на набора от данни.** Parquet файловете в `data/processed/`, доставени тук, са намален дял (`cv_pool` 2 550 + `test_in_domain` 450 + кръстосано-обзорни 1 200 реда). Водещите числа на статията (HardCBM 94.41 ± 0.36% точност, 5-кратна CV) идват от пълната таблица с **18 000 източника** (`data/real/gaia_all_features.parquet`, 3 000 на клас), разделена на 15 300 CV + 2 700 тест. Вижте секциите [Възпроизвеждане на резултатите](#възпроизвеждане-на-резултатите-пълен-конвейер) и [Набор от данни](#набор-от-данни) за това как да възстановите пълния набор от данни.

## Структура на хранилището

Анотирано дърво на кода и доставените данни в това хранилище (импортируемият пакет е `cbm_variable_stars`).

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

> 6-те класа са `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR`; 12-те концепции са `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` — и двете са дефинирани авторитетно в `cbm_variable_stars/shared/constants.py`.

## Възпроизвеждане на резултатите (пълен конвейер)

Конвейерът от край до край върви от сурова фотометрия на Gaia DR3 (и OGLE) до обучени модели, кръстосано валидирани метрики и фигурите/таблиците на статията. Всеки етап е номериран скрипт в `scripts/`, задвижван от един конфигурационен файл (`configs/default.yaml`). Изпълнението на конвейера регенерира локална директория `results/`; публикуваните обучени модели, метрики и фигури са в статията в *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), а не в това кодово дърво.

### Предпоставки

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

Етапи 01–02 правят мрежови заявки към архивите на Gaia TAP и OGLE, така че се нуждаят от достъп до интернет. Обучението (06) се изпълнява на CPU по подразбиране; подайте `--device cuda` за GPU.

### Стъпка по стъпка (действителните скриптове на диска)

Всички команди се изпълняват от корена на хранилището; всеки скрипт чете `configs/default.yaml` по подразбиране (заменете с `--config`).

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

Бележки за най-използваните флагове:

- **06** изисква `--data_path`; неговият списък по подразбиране `--models` е подмножеството от 5 модела `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`. Подайте пълния списък от 8 модела (по-горе), за да възпроизведете водещото сравнение на статията. Други стойности по подразбиране: `--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`.
- **05** опции: `--no-ogle` (само Gaia), `--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (по подразбиране `10dim`), `--verify-folds`.
- **07** опции: `--device cuda`, както и `--skip-training`/`--skip-ablation` и др. за изпълнение на подмножества.
- **08** опции: `--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`.

### Стъпка → скрипт → изход

| Стъпка | Скрипт | Основни входове | Основни изходи |
|------|--------|------------|-------------|
| 1. Изтегляне на Gaia | `scripts/01_download_gaia.py` | архив Gaia TAP | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. Изтегляне на OGLE | `scripts/02_download_ogle.py` | архив OGLE-IV | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. Извличане на признаци | `scripts/03_extract_features.py` | сурови криви на блясъка | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. Валидиране на признаци | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, доклад за качеството |
| 5. Изграждане на набор от данни | `scripts/05_build_dataset.py` | валидирани признаци | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. Обучение на модели | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | CV резултати + контролни точки за всеки модел, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. Изпълнение на експерименти | `scripts/07_run_experiments.py` | обработени данни + обучени модели | JSON за аблация / интервенция / кръстосано-обзорни / крива на обучение |
| 8. Генериране на фигури | `scripts/08_generate_figures.py` | резултати в JSON | фигури на статията (PDF) + LaTeX таблици |

### Конфигурация

Всички прогони се задвижват от **`configs/default.yaml`** (зареждан чрез OmegaConf; подайте `--config <file>` на който и да е скрипт, за да го замените). Той е единственият източник на истина за: случайно семе (`project.random_seed: 42`); схема на разделяне (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`, стратифицирана, StandardScaler); цели за изтегляне по класове (`var_types.*`); параметри за извличане на признаци (`features.*` — търсене на период, хармоници по Фурие, откриване на алиаси при периода на прецесия на Gaia от 63 дни и т.н.); хиперпараметри на обучението (`training.*` — партида 256, lr 1e-3, макс. 200 епохи, търпение 15, косинусов график с топли рестарти); архитектура за всеки модел (`models.*`); и експериментални мрежи (`experiments.*`). Съществуват две спомагателни конфигурации: `configs/feature_config.yaml` и `configs/gaia_queries.yaml`.

### Бележки за възпроизводимостта / обхвата на данните

- **Семе и дялове.** `random_seed=42` и 5-кратната стратифицирана CV са фиксирани в `configs/default.yaml`; `05_build_dataset.py` записва детерминистични индекси на дяловете в `data/processed/cv_folds.pkl`.
- **Водещ набор от данни срещу доставен дял.** Водещите числа на статията (HardCBM 94.41% ± 0.36% точност) идват от пълната балансирана матрица на Gaia с **18 000 източника** (3 000/клас; 15 300 CV + 2 700 тест). Файлът `data/processed/cv_pool.parquet`, доставен *в това дърво*, е **намален демонстрационен дял** (2 550 реда, 425/клас); изпълнението върху него няма да възпроизведе водещата точност. Повторното изпълнение на етапите за данни възстановява пълния набор от данни от архивите; пълната матрица от признаци също е достъпна като `data/real/gaia_all_features.parquet` (18 000 реда).
- **OGLE при поискване.** `data/raw/ogle/` се доставя празна по проект — стъпка 02 изтегля кривите на блясъка на OGLE по време на изпълнение (мрежа е необходима за кръстосано-обзорния тест).

> **Бележка за Makefile.** `make` целите отразяват фазите по-горе (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`). Комитнатите цели `data:` и `train:` са леко разсинхронизирани с текущите скриптове — `make data` препраща към по-стари имена на скриптове, а `make train` пропуска задължителния аргумент `--data_path` — така че изричните команди `python scripts/0N_*.py` по-горе са каноничните, работещи извиквания.

## Набор от данни

Хранилището доставя **набор от данни с признаци на променливи звезди от Gaia DR3** в `data/`, организиран като възпроизводим каскаден процес от сурова фотометрия до нормализирани дялове, готови за кръстосана валидация. Всички таблични данни се съхраняват като Apache Parquet; индексите на дяловете за кръстосана валидация и напасканият скейлер са pickle (`.pkl`); картата на етикетите е JSON.

Наборът от данни обхваща **6 класа променливи звезди**, описани с **12 физически смислени концепции** (вижте таблицата [Концепции](#концепции)). Водещата изследователска таблица съдържа **18 000 източника от Gaia DR3**, балансирани при **3 000 на клас** (`data/real/gaia_all_features.parquet`, във физически/немащабирани единици), разделени на пул за 5-кратна кръстосана валидация с 15 300 източника и резервен тестов набор с 2 700 източника.

Организация в `data/`:

| Директория | Съдържание |
|---|---|
| `raw/gaia/epoch_photometry/` | Криви на блясъка в G-лента на Gaia DR3 за всеки източник, по един Parquet за `<source_id>` (колони `time, mag, mag_err`). |
| `raw/gaia/metadata/` | Метаданни за източниците по класове и обединени (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`). |
| `raw/ogle/` | Празна по проект; кръстосано-обзорните криви на блясъка на OGLE се изтеглят при поискване. |
| `interim/` | Извлечени признаци преди разделяне (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`). |
| `processed/` | Готови за кръстосана валидация, StandardScaler-нормализирани (z-score) дялове: `cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE, извън домейна), плюс `cv_folds.pkl` (индекси за 5-кратен StratifiedKFold), `scaler.pkl` и `label_mapping.json`. |
| `expanded/` | По-голям аугментиран вариант (`gaia_expanded_features.parquet`, 30 000 реда). |
| `real/` | Водещата таблица с признаци от 18 000 източника във физически единици (`gaia_all_features.parquet`) плюс сурови метаданни. |

Всеки ред с признаци носи колони за идентификатор/етикет/качество (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`), последвани от 12-те колони с концепции. Глобалният `StandardScaler` се напасва само върху пула за кръстосана валидация и се прилага към двата тестови набора; `period_snr` се импутира с медиана по време на мащабиране.

> **Бележка.** Parquet файловете в `data/processed/`, доставени в това дърво, са намален демонстрационен дял (`cv_pool` 2 550 + `test_in_domain` 450 = 3 000 реда; кръстосано-обзорен тест 1 200). Пълната матрица от 18 000 източника, използвана за публикуваните водещи резултати, съответства на `data/real/gaia_all_features.parquet`.

## Модели

Сравняват се осем варианта на модели: шест невронни мрежи и два класически дървесни базови модела. (Невронният пакет допълнително доставя вариант `EndToEndHardCBM` (`e2e_hard_cbm`) — 1D-CNN концептуален енкодер върху фазово-нагънати криви на блясъка — отвъд осемте водещи варианта по-долу.)

| Модел | Ключ в регистъра | Описание |
|---|---|---|
| **HardCBM** | `hard_cbm` | Hard Concept Bottleneck Model; входните признаци служат като 12-концептуалното гърло, захранващо MLP предиктор (`12→64→32→6`). Референтният интерпретируем модел. |
| **HardCBM-Linear** | `hard_cbm_linear` | Hard CBM с единичен `Linear(12, 6)` предиктор (78 параметъра); теглата се четат директно като приноси концепция-към-клас. Максимално интерпретируем. |
| **HardCBM-Cal** | `hard_cbm_cal` | Калибриран Hard CBM с 12 независими калибрационни глави, които премахват шума от извлечените концепции преди MLP предиктор; основна архитектура за интервенционни експерименти. |
| **SoftCBM** | `soft_cbm` | Soft CBM с непрекъснати вграждания за всяка концепция (48-измерно гърло); по-широкото гърло разменя интерпретируемост за по-висока точност. |
| **CEM** | `cem` | Concept Embedding Model (Espinosa Zarlenga et al. 2022); всяка концепция е положителна/отрицателна двойка вграждания, смесена според своята активация. |
| **MLP** | `mlp` | Обикновен базов многослоен перцептрон (`12→128→64→6`), без гърло; измерва цената в точност на гърлото. |
| **Random Forest** | `rf` | Класически базов модел тип „черна кутия“ (500 дървета, балансирани тегла на класовете), съгласуван с официалния класификатор на Gaia DR3 (Rimoldini et al. 2023). |
| **XGBoost** | `xgb` | Базов модел тип „черна кутия“ с градиентно усилени дървета; SHAP стойностите се изчисляват за сравнение с концептуалната важност на CBM. |

### Водещи метрики

5-кратна кръстосана валидация (`RANDOM_SEED=42`), macro-F1 като основна метрика, върху пълния набор от данни от Gaia DR3 с 18 000 източника (15 300 CV + 2 700 тест; 3 000 на клас). Това са публикуваните стойности от статията в *Astronomy & Astrophysics* (Таблица 2 на статията):

| Модел | Точност (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

Разликата между интерпретируемост и точност е видима директно: прозрачният HardCBM (94.4%) изостава от дървесните базови модели тип „черна кутия“ (≈ 99.8%) с около 5 процентни пункта, което е именно цената на принуждаването на всяко решение да премине през физическо гърло от 12 концепции.

## Концепции

12-те физически концепции, образуващи гърлото, във фиксиран ред (авторитетно дефинирани в `cbm_variable_stars/shared/constants.py`).

| # | Концепция | Единица / Диапазон | Значение |
|---|---|---|---|
| 1 | `period` | дни | Основен период на пулсация/променливост. |
| 2 | `amplitude` | mag | Амплитуда връх-до-връх на кривата на блясъка. |
| 3 | `rise_fraction` | безразмерна, [0, 1] | Дял от цикъла, прекаран във възходящ блясък. |
| 4 | `R21` | безразмерно отношение | Отношение на амплитудите по Фурие A2/A1. |
| 5 | `R31` | безразмерно отношение | Отношение на амплитудите по Фурие A3/A1. |
| 6 | `phi21` | радиани, [0, 2π) | Фазова разлика по Фурие φ2 − 2φ1. |
| 7 | `skewness` | безразмерна | Асиметрия на разпределението на величините. |
| 8 | `kurtosis` | безразмерна | Излишен (по Фишер) ексцес на разпределението на величините. |
| 9 | `stetson_K` | безразмерен | Индекс на променливост K на Stetson. |
| 10 | `period_snr` | безразмерен | Значимост на периода, −log₁₀(вероятност за фалшива тревога). |
| 11 | `color_bp_rp` | mag | Цветови индекс BP − RP на Gaia. |
| 12 | `mean_mag` | mag | Средна величина в G-лента на Gaia. |

Шестте класа са `RRAB` (RR Lyrae, основна мода), `RRC` (RR Lyrae, първи обертон), `DCEP` (класически цефеид), `DSCT_SXPHE` (Delta Scuti / SX Phoenicis), `ECL` (затъмнително-двойна) и `MIRA_SR` (Mira / полуправилна дълъгопериодична променлива), индексирани 0–5 в този ред.

## Цитиране

Ако използвате тази работа, моля цитирайте придружаващата статия:

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

Чист текст: Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990.

## Лиценз

Това хранилище в момента не доставя изричен файл с лиценз с отворен код, така че не следва да се предполага общо предоставяне на права за повторно използване, преразпространение или модификация. Ако желаете да използвате повторно кода или данните отвъд горните условия за цитиране, моля свържете се с автора (Chenglong Yin, Sofia University), за да уговорите разрешение. Официален лиценз може да бъде добавен в бъдещо издание.

## Връзки

- **Статия (Astronomy & Astrophysics):** <https://doi.org/10.1051/0004-6361/202659990>
- **Интерактивен уеб придружител:** <https://cbm-variable-stars.yinchenglong.com>
