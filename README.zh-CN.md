<div align="center">

[English](README.md) · **中文** · [Čeština](README.cs.md) · [Български](README.bg.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

# CBM Variable Stars

**基于概念瓶颈模型（Concept Bottleneck Models）的可解释变星分类——每一个预测都可追溯到 12 个具有物理意义的恒星概念。**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## 概述

据我们所知，`cbm_variable_stars` 是**概念瓶颈模型（Concept Bottleneck Models，CBMs）**首次应用于天文变星分类。它将来自 Gaia DR3 测光数据的变星分类为 **6 个类别**——RR Lyrae 基模与泛音、经典造父变星、Delta Scuti / SX Phoenicis、食双星，以及 Mira / 半规则长周期变星——其方式是让每一个决策都经过 **12 个可解释、可供天文学家检视的概念**，而非一个不透明的特征空间。

借助本工具包，你可以获得：

- 一条完整、可复现的流水线，从 Gaia DR3（以及 OGLE 跨巡天）特征一直到训练完成的分类器；
- 横跨可解释性谱系的 **8 个主要模型变体**——从完全透明的 78 参数线性 CBM，到黑盒的 Random Forest 与 XGBoost 基线；
- 5 折交叉验证、消融、干预与跨巡天实验，并附有支撑配套论文的各项指标、显著性检验、图表与 LaTeX 表格。

配套论文已发表于 *Astronomy & Astrophysics*（DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)），并提供了一个交互式网页演示，地址为 <https://cbm-variable-stars.yinchenglong.com>。

## 为何采用概念瓶颈模型

大多数深度分类器将原始输入直接映射到标签，没有留下任何可检视的说明来解释*为什么*某颗恒星被划入某个类别。概念瓶颈模型则强制每个预测都通过一层狭窄的、对人类有意义的**概念**：

```
raw photometric features  →  12 physical concepts  →  class
```

由于所有分类信息都必须流经这个 12 维的概念瓶颈，每一个预测都可以追溯到天文学家能够读取与核查的物理量——周期、振幅、傅里叶形状参数、颜色等等。这也使模型具有**可交互性**：你可以覆盖某个概念（例如，提供一个经过修正的周期），并观察预测类别如何随之变化。这种透明性在原始准确率上有可度量的代价，而量化这种可解释性与性能之间的权衡，正是本工作的一个核心主题。

## 亮点 / 关键结果

- **首个用于变星分类的 CBM**——一个基于物理、由 12 个概念构成的瓶颈，覆盖来自 Gaia DR3 的 6 个变星类别。
- **构造上即可解释**——每个预测都可追溯到 12 个具名物理概念，并且概念可在推理时被覆盖，从而对预测进行干预。
- **强劲而诚实的准确率**——在以 18,000 个源的 Gaia DR3 数据集为主的设置下（每类 3,000 个均衡样本），Hard CBM 在 5 折交叉验证中达到 **94.41% ± 0.36% 的准确率**（macro-F1 94.37% ± 0.38%，MCC 0.933）。
- **可解释性代价，已量化**——黑盒基线（Random Forest 与 XGBoost ≈ 99.8%）比透明的 Hard CBM 高出约 5 个百分点，从而单独刻画出强制施加概念瓶颈所付出的代价。
- **横跨透明性谱系的 8 个模型**——Hard CBM、Hard CBM-Linear（78 个参数）、Hard CBM-Calibrated、Soft CBM、CEM、一个 MLP 基线、Random Forest，以及 XGBoost。
- **经过跨巡天测试**——在 Gaia → OGLE 的域偏移下进行评估，并辅以消融与概念干预研究。

## 安装

**先决条件**

- **Python ≥ 3.10**（在 `setup.py` 中通过 `python_requires=">=3.10"` 声明；已在 CPython 3.10–3.13 上验证良好）。
- **不**需要 C/C++ 工具链——所有依赖均以二进制 wheel 形式提供。
- **无需 GPU。** 模型规模很小（HardCBM ≈ 3K 参数，MLP ≈ 10K 参数），默认训练配置（批大小 256，≤ 200 个 epoch 并启用早停）在 CPU 上即可舒适运行。CUDA GPU 仅会加速完整的约 18,000 源规模的运行。
- 仅在可选的数据下载步骤中（通过 `astroquery`/`pyvo` 查询 Gaia/OGLE 档案库）才需要联网；在已构建好的数据集上训练完全可离线进行。

**克隆**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**安装**

```bash
# (recommended) create + activate a Python >=3.10 virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editable install (equivalent to `make install`)
pip install -e .
```

或者，安装完整的固定版本运行时集合：

```bash
pip install -r requirements.txt
```

**可选附加项**（在 `setup.py` 中声明）：

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**核心依赖**（自动拉取）：`numpy>=1.24`、`pandas>=2.1`、`scipy>=1.11`、`scikit-learn>=1.3`、`torch>=2.1`、`xgboost>=2.0`、`astropy>=6.0`、`astroquery>=0.4.7`、`pyvo>=1.5`、`pyarrow>=14.0`、`pyyaml>=6.0`、`omegaconf>=2.3`、`matplotlib>=3.8`、`seaborn>=0.13`、`loguru>=0.7`、`tqdm>=4.66`、`requests>=2.31`。

> 若需显式的 CUDA 构建，请在 `pip install -e .` **之前**安装匹配的 PyTorch wheel，例如 `pip install torch --index-url https://download.pytorch.org/whl/cu121`。本项目不固定 CUDA 版本。

## 快速开始

最小的有用示例：加载随包附带的、已经标准化的概念特征，并运行一次 HardCBM 前向传播，以得到 12 个概念值和一个 6 类预测。

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

这段代码加载真实数据，通过注册表实例化一个模型，并演练每个模型变体所共享的统一 `forward(x) -> {concepts, logits, probabilities}` 接口。将 `"hard_cbm"` 替换为任意注册表键（`hard_cbm_linear`、`hard_cbm_cal`、`e2e_hard_cbm`、`soft_cbm`、`cem`、`mlp`），即可尝试其他架构。

若要**训练**而不是使用随机权重，请将特征封装进数据集，并驱动训练编排：

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **关于数据集规模的说明。** 这里随包附带的 `data/processed/` parquet 文件是一个缩减后的划分（`cv_pool` 2,550 + `test_in_domain` 450 + 跨巡天 1,200 行）。论文中的主要数据（HardCBM 94.41 ± 0.36% 准确率，5 折 CV）来自完整的 **18,000 源**表（`data/real/gaia_all_features.parquet`，每类 3,000 个），按 15,300 CV + 2,700 test 划分。关于如何重建完整数据集，参见 [复现结果](#复现结果完整流水线) 与 [数据集](#数据集) 两节。

## 仓库结构

本仓库中代码与随包数据的带注释目录树（导入包名为 `cbm_variable_stars`）。

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

> 这 6 个类别为 `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR`；这 12 个概念为 `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag`——两者均在 `cbm_variable_stars/shared/constants.py` 中权威地定义。

## 复现结果（完整流水线）

端到端流水线从原始的 Gaia DR3（以及 OGLE）测光数据，一直走到训练完成的模型、经交叉验证的指标，以及论文的图表/表格。每个阶段都是 `scripts/` 下的一个编号脚本，由单一配置文件（`configs/default.yaml`）驱动。运行该流水线会重新生成一个本地的 `results/` 目录；已发表的训练模型、指标与图表位于 *Astronomy & Astrophysics* 论文中（DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)），而非本代码树中。

### 先决条件

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

阶段 01–02 会对 Gaia TAP 与 OGLE 档案库发起网络调用，因此需要联网。训练（06）默认在 CPU 上运行；传入 `--device cuda` 可使用 GPU。

### 分步操作（磁盘上实际的脚本）

所有命令均在仓库根目录下运行；每个脚本默认读取 `configs/default.yaml`（用 `--config` 覆盖）。

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

关于最常用标志的说明：

- **06** 需要 `--data_path`；其默认的 `--models` 列表是 5 模型子集 `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`。传入完整的 8 模型列表（如上）即可复现论文的主要对比。其他默认值：`--output_dir results`、`--seed 42`、`--max_epochs 200`、`--patience 15`、`--device cpu`。
- **05** 选项：`--no-ogle`（仅 Gaia）、`--ogle-mode {10dim,12dim_with_match,12dim_fill_median}`（默认 `10dim`）、`--verify-folds`。
- **07** 选项：`--device cuda`，以及 `--skip-training`/`--skip-ablation` 等以运行子集。
- **08** 选项：`--figures-only`、`--tables-only`、`--results-dir`、`--output-dir`。

### 步骤 → 脚本 → 输出

| 步骤 | 脚本 | 关键输入 | 关键输出 |
|------|--------|------------|------------|
| 1. 下载 Gaia | `scripts/01_download_gaia.py` | Gaia TAP archive | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. 下载 OGLE | `scripts/02_download_ogle.py` | OGLE-IV archive | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. 提取特征 | `scripts/03_extract_features.py` | raw light curves | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. 校验特征 | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, quality report |
| 5. 构建数据集 | `scripts/05_build_dataset.py` | validated features | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. 训练模型 | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | per-model CV results + checkpoints, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. 运行实验 | `scripts/07_run_experiments.py` | processed data + trained models | ablation / intervention / cross-survey / learning-curve JSON |
| 8. 生成图表 | `scripts/08_generate_figures.py` | results JSON | paper figures (PDF) + LaTeX tables |

### 配置

所有运行都由 **`configs/default.yaml`** 驱动（通过 OmegaConf 加载；向任意脚本传入 `--config <file>` 可覆盖）。它是以下各项的单一事实来源：随机种子（`project.random_seed: 42`）；划分方案（`dataset.test_in_domain_ratio: 0.15`、`n_cv_folds: 5`、分层、StandardScaler）；每类下载目标（`var_types.*`）；特征提取参数（`features.*`——周期搜索、傅里叶谐波、在 63 天 Gaia 进动周期处的别名检测等）；训练超参数（`training.*`——批大小 256、lr 1e-3、最多 200 个 epoch、patience 15、余弦热重启调度）；每个模型的架构（`models.*`）；以及实验网格（`experiments.*`）。还存在两个辅助配置：`configs/feature_config.yaml` 与 `configs/gaia_queries.yaml`。

### 可复现性 / 数据范围说明

- **种子与折。** `random_seed=42` 与 5 折分层 CV 在 `configs/default.yaml` 中固定；`05_build_dataset.py` 将确定性的折索引写入 `data/processed/cv_folds.pkl`。
- **主数据集 vs. 随包划分。** 论文的主要数字（HardCBM 94.41% ± 0.36% 准确率）来自完整的 **18,000 源**均衡 Gaia 矩阵（每类 3,000；15,300 CV + 2,700 test）。*本代码树中*随包附带的 `data/processed/cv_pool.parquet` 是一个**缩减的演示划分**（2,550 行，每类 425）；在其上运行不会复现主准确率。重新运行数据阶段会从档案库重建完整数据集；完整的特征矩阵也以 `data/real/gaia_all_features.parquet`（18,000 行）形式提供。
- **OGLE 按需下载。** `data/raw/ogle/` 按设计随包为空——步骤 02 会在运行时下载 OGLE 光变曲线（跨巡天测试需要联网）。

> **关于 Makefile 的说明。** `make` 目标镜像了上述各阶段（`make install`、`make data`、`make train`、`make experiments`、`make figures`、`make all`）。已提交的 `data:` 与 `train:` 目标与当前脚本略有脱节——`make data` 引用了较旧的脚本名，而 `make train` 省略了必需的 `--data_path` 参数——因此上面显式的 `python scripts/0N_*.py` 命令才是规范、可用的调用方式。

## 数据集

本仓库在 `data/` 下随包提供了一个 **Gaia DR3 变星特征数据集**，组织为一条可复现的级联，从原始测光数据一直到交叉验证就绪、已归一化的划分。所有表格数据以 Apache Parquet 存储；交叉验证折索引与拟合好的 scaler 为 pickle（`.pkl`）；标签映射为 JSON。

该数据集覆盖 **6 个变星类别**，由 **12 个具有物理意义的概念**描述（参见 [概念](#概念) 表）。主研究表包含 **18,000 个 Gaia DR3 源**，按**每类 3,000 个**均衡（`data/real/gaia_all_features.parquet`，采用物理/未缩放单位），划分为一个 15,300 源的 5 折交叉验证池和一个 2,700 源的留出测试集。

`data/` 下的组织：

| 目录 | 内容 |
|---|---|
| `raw/gaia/epoch_photometry/` | 逐源的 Gaia DR3 G 波段光变曲线，每个 `<source_id>` 一个 Parquet（列为 `time, mag, mag_err`）。 |
| `raw/gaia/metadata/` | 逐类与合并后的源元数据（`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`）。 |
| `raw/ogle/` | 按设计为空；OGLE 跨巡天光变曲线按需下载。 |
| `interim/` | 划分前提取的特征（`gaia_features_raw.parquet`, `ogle_features_raw.parquet`）。 |
| `processed/` | 交叉验证就绪、经 StandardScaler 归一化（z-score）的划分：`cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet`（OGLE，域外），以及 `cv_folds.pkl`（5 折 StratifiedKFold 索引）、`scaler.pkl` 和 `label_mapping.json`。 |
| `expanded/` | 更大的增强变体（`gaia_expanded_features.parquet`，30,000 行）。 |
| `real/` | 采用物理单位的主 18,000 源特征表（`gaia_all_features.parquet`）以及原始元数据。 |

每个特征行都带有标识/标签/质量列（`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`），随后是 12 个概念列。全局 `StandardScaler` 仅在交叉验证池上拟合，并应用于两个测试集；`period_snr` 在缩放时按中位数填补。

> **说明。** 本代码树中随包附带的 `data/processed/` parquet 文件是一个缩减的演示划分（`cv_pool` 2,550 + `test_in_domain` 450 = 3,000 行；跨巡天测试 1,200）。用于已发表主结果的完整 18,000 源矩阵对应于 `data/real/gaia_all_features.parquet`。

## 模型

共对比八个模型变体：六个神经网络和两个经典树基线。（除了下面的八个主要变体之外，神经网络包还另外提供了一个 `EndToEndHardCBM`（`e2e_hard_cbm`）变体——一个在相位折叠光变曲线上的 1D-CNN 概念编码器。）

| 模型 | 注册表键 | 描述 |
|---|---|---|
| **HardCBM** | `hard_cbm` | Hard 概念瓶颈模型；输入特征充当 12 概念瓶颈，馈入一个 MLP 预测器（`12→64→32→6`）。参考的可解释模型。 |
| **HardCBM-Linear** | `hard_cbm_linear` | 带单个 `Linear(12, 6)` 预测器的 Hard CBM（78 个参数）；权重可直接读作概念到类别的贡献。可解释性最大化。 |
| **HardCBM-Cal** | `hard_cbm_cal` | 带 12 个独立校准头的校准型 Hard CBM，在馈入 MLP 预测器之前对提取出的概念去噪；干预实验的主要架构。 |
| **SoftCBM** | `soft_cbm` | 带逐概念连续嵌入的 Soft CBM（48 维瓶颈）；更宽的瓶颈以可解释性换取更高准确率。 |
| **CEM** | `cem` | 概念嵌入模型（Espinosa Zarlenga et al. 2022）；每个概念是一对正/负嵌入，由其激活值混合。 |
| **MLP** | `mlp` | 普通多层感知机基线（`12→128→64→6`），无瓶颈；用于度量瓶颈带来的准确率代价。 |
| **Random Forest** | `rf` | 经典黑盒基线（500 棵树，均衡类别权重），与 Gaia DR3 官方分类器对齐（Rimoldini et al. 2023）。 |
| **XGBoost** | `xgb` | 梯度提升树黑盒基线；计算 SHAP 值以与 CBM 概念重要性作比较。 |

### 主要指标

5 折交叉验证（`RANDOM_SEED=42`），以 macro-F1 为主要指标，基于完整的 18,000 源 Gaia DR3 数据集（15,300 CV + 2,700 test；每类 3,000）。以下为已发表的 *Astronomy & Astrophysics* 论文数值（论文表 2）：

| 模型 | 准确率 (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

可解释性与准确率之间的差距直接可见：透明的 HardCBM（94.4%）落后于黑盒树基线（≈ 99.8%）约 5 个百分点，而这恰恰是强制每个决策都通过一个 12 概念物理瓶颈所付出的代价。

## 概念

构成瓶颈的 12 个物理概念，按固定顺序排列（在 `cbm_variable_stars/shared/constants.py` 中权威地定义）。

| # | 概念 | 单位 / 取值范围 | 含义 |
|---|---|---|---|
| 1 | `period` | days | 主要脉动/变化周期。 |
| 2 | `amplitude` | mag | 光变曲线的峰峰振幅。 |
| 3 | `rise_fraction` | dimensionless, [0, 1] | 一个周期中亮度处于上升阶段的比例。 |
| 4 | `R21` | dimensionless ratio | 傅里叶振幅比 A2/A1。 |
| 5 | `R31` | dimensionless ratio | 傅里叶振幅比 A3/A1。 |
| 6 | `phi21` | radians, [0, 2π) | 傅里叶相位差 φ2 − 2φ1。 |
| 7 | `skewness` | dimensionless | 星等分布的偏度。 |
| 8 | `kurtosis` | dimensionless | 星等分布的超额（Fisher）峰度。 |
| 9 | `stetson_K` | dimensionless | Stetson K 变化指数。 |
| 10 | `period_snr` | dimensionless | 周期显著性，−log₁₀(虚警概率)。 |
| 11 | `color_bp_rp` | mag | Gaia BP − RP 颜色指数。 |
| 12 | `mean_mag` | mag | Gaia G 波段平均星等。 |

这六个类别为 `RRAB`（RR Lyrae，基模）、`RRC`（RR Lyrae，第一泛音）、`DCEP`（经典造父变星）、`DSCT_SXPHE`（Delta Scuti / SX Phoenicis）、`ECL`（食双星），以及 `MIRA_SR`（Mira / 半规则长周期变星），按该顺序索引为 0–5。

## 引用

如果你使用了本工作，请引用配套论文：

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

纯文本：Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990。

## 许可

本仓库目前未随包提供显式的开源许可文件，因此不应假定有任何关于复用、再分发或修改权利的一般性授予。如果你希望在上述引用条款之外复用代码或数据，请联系作者（Chenglong Yin，Sofia University）以安排授权。未来版本中可能会添加正式许可。

## 链接

- **论文（Astronomy & Astrophysics）：** <https://doi.org/10.1051/0004-6361/202659990>
- **交互式网页配套：** <https://cbm-variable-stars.yinchenglong.com>
