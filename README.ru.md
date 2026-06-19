<div align="center">

[English](README.md) · [中文](README.zh-CN.md) · [Čeština](README.cs.md) · [Български](README.bg.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · **Русский** · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

# CBM Variable Stars

**Интерпретируемая классификация переменных звёзд с помощью моделей концептуального бутылочного горлышка (Concept Bottleneck Models) — каждое предсказание прослеживается через 12 физически осмысленных звёздных концептов.**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## Обзор

`cbm_variable_stars` — это, насколько нам известно, первое применение **моделей концептуального бутылочного горлышка (Concept Bottleneck Models, CBM)** к астрономической классификации переменных звёзд. Он классифицирует переменные звёзды по фотометрии Gaia DR3 на **6 классов** — RR Лиры основного тона и обертона, классические цефеиды, Delta Scuti / SX Phoenicis, затменные двойные и переменные типа Mira / полуправильные долгопериодические переменные — направляя каждое решение через **12 интерпретируемых, доступных для проверки астрономом концептов** вместо непрозрачного пространства признаков.

С этим пакетом вы получаете:

- полный воспроизводимый конвейер от признаков Gaia DR3 (и кросс-обзора OGLE) до обученных классификаторов;
- **8 основных вариантов модели**, охватывающих весь спектр интерпретируемости — от полностью прозрачной линейной CBM на 78 параметров до моделей-«чёрных ящиков» Random Forest и XGBoost;
- 5-кратную перекрёстную проверку, абляцию, эксперименты с вмешательством и кросс-обзорные эксперименты, вместе с метриками, тестами значимости, рисунками и таблицами LaTeX, которые лежат в основе сопроводительной статьи.

Сопроводительная статья опубликована в журнале *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), а интерактивная веб-демонстрация доступна по адресу <https://cbm-variable-stars.yinchenglong.com>.

## Почему модели концептуального бутылочного горлышка

Большинство глубоких классификаторов отображают исходные входные данные напрямую в метку, не оставляя пригодного для проверки объяснения того, *почему* данная звезда была отнесена к данному классу. Модель концептуального бутылочного горлышка вместо этого заставляет каждое предсказание проходить через узкий слой осмысленных для человека **концептов**:

```
raw photometric features  →  12 physical concepts  →  class
```

Поскольку вся информация для классификации должна проходить через 12-мерное концептуальное бутылочное горлышко, каждое предсказание прослеживается до физических величин, которые астроном может прочитать и проверить — период, амплитуда, параметры формы Фурье, цвет и так далее. Это также делает модель **интерактивной**: вы можете перезаписать концепт (например, задать исправленный период) и наблюдать, как реагирует предсказанный класс. У этой прозрачности есть измеримая цена в исходной точности, и количественная оценка этого компромисса между интерпретируемостью и производительностью — центральная тема данной работы.

## Основные моменты / ключевые результаты

- **Первая CBM для классификации переменных звёзд** — физически обоснованное концептуальное бутылочное горлышко из 12 концептов над 6 классами переменных звёзд по данным Gaia DR3.
- **Интерпретируемость по построению** — каждое предсказание прослеживается до 12 названных физических концептов, и концепты можно переопределить во время вывода, чтобы вмешаться в предсказание.
- **Высокая, честная точность** — Hard CBM достигает **точности 94.41% ± 0.36%** (macro-F1 94.37% ± 0.38%, MCC 0.933) при 5-кратной перекрёстной проверке на основном наборе данных Gaia DR3 из 18 000 источников (3 000 сбалансированных примеров на класс).
- **Цена интерпретируемости, измеренная** — базовые модели-«чёрные ящики» (Random Forest и XGBoost ≈ 99.8%) превосходят прозрачную Hard CBM примерно на 5 процентных пунктов, что изолирует цену принудительного концептуального бутылочного горлышка.
- **8 моделей по всему спектру прозрачности** — Hard CBM, Hard CBM-Linear (78 параметров), Hard CBM-Calibrated, Soft CBM, CEM, базовая MLP, Random Forest и XGBoost.
- **Проверено на кросс-обзоре** — оценено в условиях доменного сдвига Gaia → OGLE, наряду с исследованиями абляции и концептуального вмешательства.

## Установка

**Предварительные требования**

- **Python ≥ 3.10** (объявлен через `python_requires=">=3.10"` в `setup.py`; проверенно работает на CPython 3.10–3.13).
- Набор инструментов C/C++ **не** требуется — все зависимости поставляются в виде бинарных колёс (wheels).
- **GPU не требуется.** Модели небольшие (HardCBM ≈ 3K параметров, MLP ≈ 10K параметров), и стандартная конфигурация обучения (размер пакета 256, ≤ 200 эпох с ранней остановкой) комфортно работает на CPU. GPU с CUDA лишь ускоряет полные запуски на ~18 000 источников.
- Доступ в интернет нужен **только** для опциональных шагов загрузки данных (запросы к архивам Gaia/OGLE через `astroquery`/`pyvo`); обучение на уже собранном наборе данных полностью работает офлайн.

**Клонирование**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**Установка**

```bash
# (recommended) create + activate a Python >=3.10 virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editable install (equivalent to `make install`)
pip install -e .
```

В качестве альтернативы установите полный закреплённый набор зависимостей времени выполнения:

```bash
pip install -r requirements.txt
```

**Опциональные дополнения** (объявлены в `setup.py`):

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**Основные зависимости** (подтягиваются автоматически): `numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`.

> Для явной сборки с CUDA установите подходящее колесо PyTorch **до** `pip install -e .`, например `pip install torch --index-url https://download.pytorch.org/whl/cu121`. Проект не закрепляет версию CUDA.

## Быстрый старт

Наименьший полезный пример: загрузить поставляемые, уже стандартизованные концептуальные признаки и выполнить прямой проход HardCBM, чтобы получить 12 значений концептов и предсказание из 6 классов.

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

Это загружает реальные данные, инстанцирует модель через реестр и задействует единый интерфейс `forward(x) -> {concepts, logits, probabilities}`, общий для каждого варианта модели. Замените `"hard_cbm"` на любой ключ реестра (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`), чтобы опробовать другие архитектуры.

Чтобы **обучать** модель вместо использования случайных весов, оберните признаки в набор данных и запустите оркестрацию обучения:

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **Замечание о размере набора данных.** Поставляемые здесь parquet-файлы в `data/processed/` представляют собой сокращённую выборку (`cv_pool` 2 550 + `test_in_domain` 450 + кросс-обзор 1 200 строк). Основные показатели статьи (HardCBM точность 94.41 ± 0.36%, 5-кратная CV) получены из полной таблицы на **18 000 источников** (`data/real/gaia_all_features.parquet`, 3 000 на класс), разбитой на 15 300 CV + 2 700 test. См. разделы [Воспроизведение результатов](#воспроизведение-результатов-полный-конвейер) и [Набор данных](#набор-данных) о том, как пересобрать полный набор данных.

## Структура репозитория

Аннотированное дерево кода и поставляемых данных в этом репозитории (импортируемый пакет — `cbm_variable_stars`).

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

> 6 классов — это `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR`; 12 концептов — это `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` — оба авторитетно определены в `cbm_variable_stars/shared/constants.py`.

## Воспроизведение результатов (полный конвейер)

Сквозной конвейер идёт от исходной фотометрии Gaia DR3 (и OGLE) к обученным моделям, перекрёстно проверенным метрикам и рисункам/таблицам статьи. Каждый этап — это пронумерованный скрипт в `scripts/`, управляемый единым файлом конфигурации (`configs/default.yaml`). Запуск конвейера заново генерирует локальный каталог `results/`; опубликованные обученные модели, метрики и рисунки находятся в статье *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), а не в этом дереве кода.

### Предварительные требования

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

Этапы 01–02 делают сетевые вызовы к архивам Gaia TAP и OGLE, поэтому им нужен доступ в интернет. Обучение (06) по умолчанию выполняется на CPU; передайте `--device cuda` для GPU.

### Пошагово (реальные скрипты на диске)

Все команды выполняются из корня репозитория; каждый скрипт по умолчанию читает `configs/default.yaml` (переопределяется через `--config`).

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

Замечания по наиболее используемым флагам:

- **06** требует `--data_path`; его список `--models` по умолчанию — это подмножество из 5 моделей `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`. Передайте полный список из 8 моделей (выше), чтобы воспроизвести основное сравнение статьи. Другие значения по умолчанию: `--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`.
- **05** опции: `--no-ogle` (только Gaia), `--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (по умолчанию `10dim`), `--verify-folds`.
- **07** опции: `--device cuda`, а также `--skip-training`/`--skip-ablation` и т. д. для запуска подмножеств.
- **08** опции: `--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`.

### Этап → скрипт → результат

| Этап | Скрипт | Ключевые входы | Ключевые выходы |
|------|--------|------------|-------------|
| 1. Загрузка Gaia | `scripts/01_download_gaia.py` | архив Gaia TAP | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. Загрузка OGLE | `scripts/02_download_ogle.py` | архив OGLE-IV | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. Извлечение признаков | `scripts/03_extract_features.py` | исходные кривые блеска | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. Валидация признаков | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, отчёт о качестве |
| 5. Сборка набора данных | `scripts/05_build_dataset.py` | валидированные признаки | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. Обучение моделей | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | результаты CV по каждой модели + контрольные точки, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. Запуск экспериментов | `scripts/07_run_experiments.py` | обработанные данные + обученные модели | JSON по абляции / вмешательству / кросс-обзору / кривой обучения |
| 8. Генерация рисунков | `scripts/08_generate_figures.py` | JSON результатов | рисунки статьи (PDF) + таблицы LaTeX |

### Конфигурация

Все запуски управляются файлом **`configs/default.yaml`** (загружается через OmegaConf; передайте `--config <file>` любому скрипту для переопределения). Это единственный источник истины для: начального значения генератора случайных чисел (`project.random_seed: 42`); схемы разбиения (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`, стратифицированное, StandardScaler); целевых количеств загрузки по классам (`var_types.*`); параметров извлечения признаков (`features.*` — поиск периода, гармоники Фурье, обнаружение алиасов на периоде прецессии Gaia 63 дня и т. д.); гиперпараметров обучения (`training.*` — пакет 256, lr 1e-3, макс. 200 эпох, терпение 15, расписание косинусного «тёплого» перезапуска); архитектуры по каждой модели (`models.*`); и сеток экспериментов (`experiments.*`). Существуют два вспомогательных конфига: `configs/feature_config.yaml` и `configs/gaia_queries.yaml`.

### Замечания о воспроизводимости / охвате данных

- **Seed и фолды.** `random_seed=42` и 5-кратная стратифицированная CV зафиксированы в `configs/default.yaml`; `05_build_dataset.py` записывает детерминированные индексы фолдов в `data/processed/cv_folds.pkl`.
- **Основной набор данных против поставляемой выборки.** Основные числа статьи (HardCBM точность 94.41% ± 0.36%) получены из полной сбалансированной матрицы Gaia на **18 000 источников** (3 000/класс; 15 300 CV + 2 700 test). Файл `data/processed/cv_pool.parquet`, поставляемый *в этом дереве*, представляет собой **сокращённую демонстрационную выборку** (2 550 строк, 425/класс); запуск на ней не воспроизведёт основную точность. Повторный запуск этапов данных пересобирает полный набор данных из архивов; полная матрица признаков также доступна как `data/real/gaia_all_features.parquet` (18 000 строк).
- **OGLE по запросу.** `data/raw/ogle/` поставляется пустым по замыслу — этап 02 загружает кривые блеска OGLE во время выполнения (для кросс-обзорного теста требуется сеть).

> **Замечание о Makefile.** Цели `make` отражают фазы выше (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`). Зафиксированные цели `data:` и `train:` немного рассинхронизированы с текущими скриптами — `make data` ссылается на старые имена скриптов, а `make train` опускает обязательный аргумент `--data_path` — поэтому явные команды `python scripts/0N_*.py` выше являются каноническими, рабочими вызовами.

## Набор данных

Репозиторий поставляет **набор данных признаков переменных звёзд Gaia DR3** в каталоге `data/`, организованный как воспроизводимый каскад от исходной фотометрии до нормализованных выборок, готовых к перекрёстной проверке. Все табличные данные хранятся в формате Apache Parquet; индексы фолдов перекрёстной проверки и подогнанный масштабатор — в pickle (`.pkl`); карта меток — в JSON.

Набор данных охватывает **6 классов переменных звёзд**, описанных **12 физически осмысленными концептами** (см. таблицу [Концепты](#концепты)). Основная исследовательская таблица содержит **18 000 источников Gaia DR3**, сбалансированных по **3 000 на класс** (`data/real/gaia_all_features.parquet`, в физических/немасштабированных единицах), разбитых на пул 5-кратной перекрёстной проверки на 15 300 источников и отложенный тестовый набор на 2 700 источников.

Организация в каталоге `data/`:

| Каталог | Содержимое |
|---|---|
| `raw/gaia/epoch_photometry/` | Кривые блеска Gaia DR3 в полосе G по каждому источнику, один Parquet на `<source_id>` (столбцы `time, mag, mag_err`). |
| `raw/gaia/metadata/` | Метаданные источников по классам и сводные (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`). |
| `raw/ogle/` | Пуст по замыслу; кросс-обзорные кривые блеска OGLE загружаются по запросу. |
| `interim/` | Извлечённые признаки до разбиения (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`). |
| `processed/` | Готовые к перекрёстной проверке, нормализованные StandardScaler (z-оценка) выборки: `cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE, вне домена), плюс `cv_folds.pkl` (индексы 5-кратного StratifiedKFold), `scaler.pkl` и `label_mapping.json`. |
| `expanded/` | Больший аугментированный вариант (`gaia_expanded_features.parquet`, 30 000 строк). |
| `real/` | Основная таблица признаков на 18 000 источников в физических единицах (`gaia_all_features.parquet`) плюс исходные метаданные. |

Каждая строка признаков несёт столбцы идентификатора/метки/качества (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`), за которыми следуют 12 столбцов концептов. Глобальный `StandardScaler` подгоняется только на пуле перекрёстной проверки и применяется к обоим тестовым наборам; `period_snr` импутируется медианой во время масштабирования.

> **Замечание.** Поставляемые в этом дереве parquet-файлы `data/processed/` представляют собой сокращённую демонстрационную выборку (`cv_pool` 2 550 + `test_in_domain` 450 = 3 000 строк; кросс-обзорный тест 1 200). Полная матрица на 18 000 источников, использованная для опубликованных основных результатов, соответствует `data/real/gaia_all_features.parquet`.

## Модели

Сравниваются восемь вариантов модели: шесть нейронных сетей и две классические древесные базовые модели. (Нейронный пакет дополнительно поставляет вариант `EndToEndHardCBM` (`e2e_hard_cbm`) — концептуальный кодировщик 1D-CNN над свёрнутыми по фазе кривыми блеска — помимо восьми основных вариантов ниже.)

| Модель | Ключ реестра | Описание |
|---|---|---|
| **HardCBM** | `hard_cbm` | Модель жёсткого концептуального бутылочного горлышка; входные признаки служат 12-концептным бутылочным горлышком, питающим MLP-предиктор (`12→64→32→6`). Эталонная интерпретируемая модель. |
| **HardCBM-Linear** | `hard_cbm_linear` | Hard CBM с единственным предиктором `Linear(12, 6)` (78 параметров); веса читаются напрямую как вклады «концепт-в-класс». Максимально интерпретируема. |
| **HardCBM-Cal** | `hard_cbm_cal` | Калиброванная Hard CBM с 12 независимыми калибровочными головами, которые очищают извлечённые концепты от шума перед MLP-предиктором; основная архитектура для экспериментов с вмешательством. |
| **SoftCBM** | `soft_cbm` | Soft CBM с непрерывными вложениями по каждому концепту (48-мерное бутылочное горлышко); более широкое бутылочное горлышко обменивает интерпретируемость на более высокую точность. |
| **CEM** | `cem` | Модель концептуальных вложений (Concept Embedding Model, Espinosa Zarlenga et al. 2022); каждый концепт — это пара положительного/отрицательного вложений, смешиваемая своей активацией. |
| **MLP** | `mlp` | Простая базовая модель многослойного персептрона (`12→128→64→6`), без бутылочного горлышка; измеряет цену точности бутылочного горлышка. |
| **Random Forest** | `rf` | Классическая базовая модель-«чёрный ящик» (500 деревьев, сбалансированные веса классов), согласованная с официальным классификатором Gaia DR3 (Rimoldini et al. 2023). |
| **XGBoost** | `xgb` | Базовая модель-«чёрный ящик» на градиентном бустинге деревьев; значения SHAP вычисляются для сравнения с важностью концептов CBM. |

### Основные метрики

5-кратная перекрёстная проверка (`RANDOM_SEED=42`), macro-F1 как первичная метрика, на полном наборе данных Gaia DR3 из 18 000 источников (15 300 CV + 2 700 test; 3 000 на класс). Это опубликованные значения статьи *Astronomy & Astrophysics* (Таблица 2 статьи):

| Модель | Точность (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

Разрыв между интерпретируемостью и точностью виден напрямую: прозрачная HardCBM (94.4%) отстаёт от древесных базовых моделей-«чёрных ящиков» (≈ 99.8%) примерно на 5 процентных пунктов, что в точности является ценой принуждения каждого решения проходить через физическое бутылочное горлышко из 12 концептов.

## Концепты

12 физических концептов, образующих бутылочное горлышко, в фиксированном порядке (авторитетно определены в `cbm_variable_stars/shared/constants.py`).

| # | Концепт | Единица / Диапазон | Значение |
|---|---|---|---|
| 1 | `period` | сутки | Основной период пульсации/переменности. |
| 2 | `amplitude` | mag | Амплитуда кривой блеска от пика до пика. |
| 3 | `rise_fraction` | безразмерная, [0, 1] | Доля цикла, проведённая в нарастании яркости. |
| 4 | `R21` | безразмерное отношение | Отношение амплитуд Фурье A2/A1. |
| 5 | `R31` | безразмерное отношение | Отношение амплитуд Фурье A3/A1. |
| 6 | `phi21` | радианы, [0, 2π) | Разность фаз Фурье φ2 − 2φ1. |
| 7 | `skewness` | безразмерная | Асимметрия распределения звёздных величин. |
| 8 | `kurtosis` | безразмерная | Избыточный (по Фишеру) эксцесс распределения звёздных величин. |
| 9 | `stetson_K` | безразмерная | Индекс переменности Stetson K. |
| 10 | `period_snr` | безразмерная | Значимость периода, −log₁₀(вероятность ложной тревоги). |
| 11 | `color_bp_rp` | mag | Показатель цвета Gaia BP − RP. |
| 12 | `mean_mag` | mag | Средняя звёздная величина Gaia в полосе G. |

Шесть классов — это `RRAB` (RR Лиры, основной тон), `RRC` (RR Лиры, первый обертон), `DCEP` (классическая цефеида), `DSCT_SXPHE` (Delta Scuti / SX Phoenicis), `ECL` (затменная двойная) и `MIRA_SR` (Mira / полуправильная долгопериодическая переменная), проиндексированные 0–5 в этом порядке.

## Цитирование

Если вы используете эту работу, пожалуйста, цитируйте сопроводительную статью:

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

Простой текст: Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990.

## Лицензия

В настоящее время этот репозиторий не поставляет явного файла лицензии с открытым исходным кодом, поэтому не следует предполагать какого-либо общего предоставления прав на повторное использование, распространение или модификацию. Если вы хотите повторно использовать код или данные сверх условий цитирования выше, пожалуйста, свяжитесь с автором (Chenglong Yin, Sofia University) для получения разрешения. Формальная лицензия может быть добавлена в будущем выпуске.

## Ссылки

- **Статья (Astronomy & Astrophysics):** <https://doi.org/10.1051/0004-6361/202659990>
- **Интерактивный веб-компаньон:** <https://cbm-variable-stars.yinchenglong.com>
