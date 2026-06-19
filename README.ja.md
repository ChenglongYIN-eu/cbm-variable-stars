<div align="center">

[English](README.md) · [中文](README.zh-CN.md) · [Čeština](README.cs.md) · [Български](README.bg.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · **日本語** · [Português](README.pt.md)

</div>

# CBM 変光星

**概念ボトルネックモデルによる解釈可能な変光星分類 — すべての予測を物理的に意味のある 12 個の恒星概念を通じて追跡する。**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## 概要

`cbm_variable_stars` は、我々の知る限り、天文学的な変光星分類への**概念ボトルネックモデル (Concept Bottleneck Models, CBM)** の初めての応用である。Gaia DR3 測光から変光星を **6 クラス** — 基本モードおよび第一倍音の RR Lyrae 型、古典的セファイド、たて座デルタ型 / SX Phoenicis 型、食連星、Mira 型 / 半規則型長周期変光星 — に分類するが、不透明な特徴量空間を介するのではなく、すべての判断を**解釈可能で天文学者が検証できる 12 個の概念**を経由させて行う。

このパッケージで得られるものは以下のとおりである。

- Gaia DR3 (および OGLE クロスサーベイ) の特徴量から学習済み分類器までの、完全で再現可能なパイプライン。
- 解釈可能性のスペクトル全体にわたる **8 つの主要モデルバリアント** — 完全に透明な 78 パラメータの線形 CBM から、ブラックボックスの Random Forest および XGBoost ベースラインまで。
- 5 分割交差検証、アブレーション、介入、クロスサーベイ実験、ならびに付随論文を裏付ける指標、有意性検定、図、LaTeX 表。

付随論文は *Astronomy & Astrophysics* に掲載されており (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990))、インタラクティブな Web デモは <https://cbm-variable-stars.yinchenglong.com> で利用可能である。

## なぜ概念ボトルネックモデルなのか

ほとんどの深層分類器は生の入力をラベルへ直接写像し、ある恒星がなぜ特定のクラスに割り当てられたのか、検証可能な説明を残さない。概念ボトルネックモデルはその代わりに、すべての予測を人間にとって意味のある**概念**からなる狭い層を通過させることを強制する。

```
raw photometric features  →  12 physical concepts  →  class
```

すべての分類情報は 12 次元の概念ボトルネックを通って流れなければならないため、あらゆる予測は天文学者が読んで確認できる物理量 — 周期、振幅、フーリエ形状パラメータ、色など — まで追跡可能である。これはまたモデルを**対話可能**にする。すなわち、概念を上書きし (例えば補正された周期を与え)、予測されるクラスがどう応答するかを観察できる。この透明性は生の精度に測定可能なコストを伴い、その解釈可能性対性能のトレードオフを定量化することが本研究の中心テーマである。

## ハイライト / 主要な結果

- **変光星分類のための初めての CBM** — Gaia DR3 由来の 6 つの変光星クラスにわたる、物理的に裏付けられた 12 概念のボトルネック。
- **構成上、解釈可能** — すべての予測は 12 個の名前付き物理概念まで追跡可能であり、推論時に概念を上書きして予測に介入できる。
- **強力かつ誠実な精度** — Hard CBM は、主要な 18,000 ソースの Gaia DR3 データセット (クラスごとに 3,000 のバランスサンプル) に対する 5 分割交差検証で、**精度 94.41% ± 0.36%** (マクロ F1 94.37% ± 0.38%、MCC 0.933) に到達する。
- **解釈可能性のコストを測定** — ブラックボックスのベースライン (Random Forest および XGBoost ≈ 99.8%) は透明な Hard CBM を約 5 パーセントポイント上回り、強制された概念ボトルネックの代償を切り分ける。
- **透明性スペクトル全体にわたる 8 モデル** — Hard CBM、Hard CBM-Linear (78 パラメータ)、Hard CBM-Calibrated、Soft CBM、CEM、MLP ベースライン、Random Forest、XGBoost。
- **クロスサーベイで検証済み** — Gaia → OGLE のドメインシフト下で評価し、アブレーションおよび概念介入の研究も併せて実施。

## インストール

**前提条件**

- **Python ≥ 3.10** (`setup.py` 内で `python_requires=">=3.10"` として宣言。CPython 3.10–3.13 で動作確認済み)。
- C/C++ ツールチェーンは**不要** — すべての依存関係はバイナリホイールとして配布される。
- **GPU は不要。** モデルは小さく (HardCBM ≈ 3K パラメータ、MLP ≈ 10K パラメータ)、デフォルトの学習設定 (バッチサイズ 256、早期終了付きで ≤ 200 エポック) は CPU で快適に動作する。CUDA GPU はフルの約 18,000 ソースの実行を高速化するのみである。
- インターネットアクセスが必要なのは**オプションの**データダウンロード手順 (`astroquery`/`pyvo` 経由の Gaia/OGLE アーカイブ照会) **のみ**であり、構築済みデータセットでの学習は完全にオフラインで動作する。

**クローン**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**インストール**

```bash
# (推奨) Python >=3.10 の仮想環境を作成 + 有効化
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 編集可能インストール (`make install` と等価)
pip install -e .
```

あるいは、固定された完全なランタイムセットをインストールする。

```bash
pip install -r requirements.txt
```

**オプションのエクストラ** (`setup.py` で宣言)：

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**コア依存関係** (自動的に取得される)：`numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`。

> 明示的な CUDA ビルドを行うには、`pip install -e .` の**前に**対応する PyTorch ホイールをインストールする。例：`pip install torch --index-url https://download.pytorch.org/whl/cu121`。本プロジェクトは CUDA バージョンを固定しない。

## クイックスタート

最小限の有用な例：配布済みの、すでに標準化された概念特徴量を読み込み、HardCBM のフォワードパスを実行して 12 個の概念値と 6 クラスの予測を得る。

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

これは実データを読み込み、レジストリを介してモデルをインスタンス化し、すべてのモデルバリアントが共有する統一された `forward(x) -> {concepts, logits, probabilities}` インターフェースを実行する。`"hard_cbm"` を任意のレジストリキー (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`) に置き換えて、他のアーキテクチャを試すことができる。

ランダムな重みを使う代わりに**学習**するには、特徴量をデータセットでラップし、学習オーケストレーションを駆動する。

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **データセットサイズに関する注意。** ここに配布されている `data/processed/` の parquet は縮小された分割 (`cv_pool` 2,550 + `test_in_domain` 450 + クロスサーベイ 1,200 行) である。論文の主要数値 (HardCBM 精度 94.41 ± 0.36%、5 分割 CV) はフルの **18,000 ソース**の表 (`data/real/gaia_all_features.parquet`、クラスごとに 3,000) に由来し、CV 15,300 + テスト 2,700 に分割される。フルデータセットを再構築する方法については [結果の再現](#結果の再現-フルパイプライン) および [データセット](#データセット) のセクションを参照のこと。

## リポジトリ構造

本リポジトリのコードと配布データの注釈付きツリー (インポートパッケージは `cbm_variable_stars`)。

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

> 6 クラスは `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR` であり、12 概念は `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` である — いずれも `cbm_variable_stars/shared/constants.py` に正式に定義されている。

## 結果の再現 (フルパイプライン)

エンドツーエンドのパイプラインは、生の Gaia DR3 (および OGLE) 測光から、学習済みモデル、交差検証済みの指標、論文の図 / 表までを通す。各ステージは `scripts/` 配下の番号付きスクリプトであり、単一の設定ファイル (`configs/default.yaml`) によって駆動される。パイプラインを実行するとローカルの `results/` ディレクトリが再生成される。公開された学習済みモデル、指標、図は、本コードツリーではなく *Astronomy & Astrophysics* 論文 (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)) に存在する。

### 前提条件

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

ステージ 01–02 は Gaia TAP および OGLE アーカイブへネットワーク呼び出しを行うため、インターネットアクセスが必要である。学習 (06) はデフォルトで CPU 上で実行される。GPU を使うには `--device cuda` を渡す。

### ステップバイステップ (ディスク上の実際のスクリプト)

すべてのコマンドはリポジトリのルートから実行する。各スクリプトはデフォルトで `configs/default.yaml` を読み込む (`--config` で上書き可能)。

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

最もよく使われるフラグに関する注意：

- **06** は `--data_path` を必要とする。そのデフォルトの `--models` リストは 5 モデルのサブセット `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf` である。論文の主要比較を再現するには、(上記の) 完全な 8 モデルのリストを渡す。その他のデフォルト：`--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`。
- **05** のオプション：`--no-ogle` (Gaia のみ)、`--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (デフォルト `10dim`)、`--verify-folds`。
- **07** のオプション：`--device cuda`、およびサブセットを実行するための `--skip-training`/`--skip-ablation` など。
- **08** のオプション：`--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`。

### ステップ → スクリプト → 出力

| ステップ | スクリプト | 主な入力 | 主な出力 |
|------|--------|------------|------------|
| 1. Gaia ダウンロード | `scripts/01_download_gaia.py` | Gaia TAP アーカイブ | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. OGLE ダウンロード | `scripts/02_download_ogle.py` | OGLE-IV アーカイブ | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. 特徴量抽出 | `scripts/03_extract_features.py` | 生の光度曲線 | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. 特徴量検証 | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, 品質レポート |
| 5. データセット構築 | `scripts/05_build_dataset.py` | 検証済み特徴量 | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. モデル学習 | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | モデルごとの CV 結果 + チェックポイント, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. 実験実行 | `scripts/07_run_experiments.py` | 処理済みデータ + 学習済みモデル | アブレーション / 介入 / クロスサーベイ / 学習曲線の JSON |
| 8. 図の生成 | `scripts/08_generate_figures.py` | 結果 JSON | 論文の図 (PDF) + LaTeX 表 |

### 設定

すべての実行は **`configs/default.yaml`** によって駆動される (OmegaConf 経由で読み込まれる。任意のスクリプトに `--config <file>` を渡して上書きする)。これは以下に関する唯一の信頼できる情報源である：乱数シード (`project.random_seed: 42`)、分割スキーム (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`、層化、StandardScaler)、クラスごとのダウンロード目標数 (`var_types.*`)、特徴量抽出パラメータ (`features.*` — 周期探索、フーリエ高調波、Gaia の 63 日歳差周期でのエイリアス検出など)、学習ハイパーパラメータ (`training.*` — バッチ 256、lr 1e-3、最大 200 エポック、patience 15、コサインウォームリスタートスケジュール)、モデルごとのアーキテクチャ (`models.*`)、および実験グリッド (`experiments.*`)。2 つの補助設定が存在する：`configs/feature_config.yaml` と `configs/gaia_queries.yaml`。

### 再現性 / データスコープに関する注意

- **シードと分割。** `random_seed=42` と 5 分割層化 CV は `configs/default.yaml` に固定されている。`05_build_dataset.py` は決定論的な分割インデックスを `data/processed/cv_folds.pkl` に書き出す。
- **主要データセット対配布分割。** 論文の主要数値 (HardCBM 精度 94.41% ± 0.36%) は、フルの **18,000 ソース**のバランスされた Gaia 行列 (クラスごとに 3,000、CV 15,300 + テスト 2,700) に由来する。*本ツリーに*配布されている `data/processed/cv_pool.parquet` は**縮小されたデモ分割** (2,550 行、クラスごとに 425) であり、これで実行しても主要精度は再現されない。データステージを再実行するとアーカイブからフルデータセットが再構築される。フル特徴量行列は `data/real/gaia_all_features.parquet` (18,000 行) としても利用可能である。
- **OGLE はオンデマンド。** `data/raw/ogle/` は設計上空のまま配布される — ステップ 02 が実行時に OGLE 光度曲線をダウンロードする (クロスサーベイテストにはネットワークが必要)。

> **Makefile に関する注意。** `make` ターゲットは上記のフェーズを反映する (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`)。コミットされた `data:` および `train:` ターゲットは現在のスクリプトとわずかにずれている — `make data` は古いスクリプト名を参照し、`make train` は必須の `--data_path` 引数を省略している — ため、上記の明示的な `python scripts/0N_*.py` コマンドが正規かつ動作する呼び出し方法である。

## データセット

本リポジトリは `data/` 配下に **Gaia DR3 変光星特徴量データセット**を配布しており、生の測光から交差検証可能で正規化された分割までの再現可能なカスケードとして構成されている。すべての表形式データは Apache Parquet として保存され、交差検証の分割インデックスと適合済みスケーラは pickle (`.pkl`)、ラベルマップは JSON である。

このデータセットは **6 つの変光星クラス**を対象とし、**12 個の物理的に意味のある概念**で記述される ([概念](#概念) の表を参照)。主要な研究用の表は **18,000 個の Gaia DR3 ソース**を **クラスごとに 3,000** でバランスして含み (`data/real/gaia_all_features.parquet`、物理的/未スケーリング単位)、15,300 ソースの 5 分割交差検証プールと 2,700 ソースのホールドアウトテストセットに分割される。

`data/` 配下の構成：

| ディレクトリ | 内容 |
|---|---|
| `raw/gaia/epoch_photometry/` | ソースごとの Gaia DR3 G バンド光度曲線。`<source_id>` ごとに 1 つの Parquet (列 `time, mag, mag_err`)。 |
| `raw/gaia/metadata/` | クラスごとおよび結合されたソースメタデータ (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`)。 |
| `raw/ogle/` | 設計上空。OGLE クロスサーベイ光度曲線はオンデマンドでダウンロードされる。 |
| `interim/` | 分割前の抽出済み特徴量 (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`)。 |
| `processed/` | 交差検証可能で StandardScaler 正規化済み (z スコア) の分割：`cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE、ドメイン外)、ならびに `cv_folds.pkl` (5 分割 StratifiedKFold インデックス), `scaler.pkl`, `label_mapping.json`。 |
| `expanded/` | より大きな拡張バリアント (`gaia_expanded_features.parquet`、30,000 行)。 |
| `real/` | 主要な 18,000 ソースの特徴量表 (`gaia_all_features.parquet`) を物理単位で、加えて生のメタデータ。 |

各特徴量行は識別子/ラベル/品質の列 (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`) を持ち、その後に 12 個の概念列が続く。グローバルな `StandardScaler` は交差検証プールのみで適合され、両方のテストセットに適用される。`period_snr` はスケーリング時に中央値で補完される。

> **注意。** 本ツリーに配布されている `data/processed/` の parquet は縮小されたデモンストレーション分割 (`cv_pool` 2,550 + `test_in_domain` 450 = 3,000 行、クロスサーベイテスト 1,200) である。公開された主要結果に使われたフルの 18,000 ソース行列は `data/real/gaia_all_features.parquet` に対応する。

## モデル

8 つのモデルバリアントを比較する：6 つのニューラルネットワークと 2 つの古典的な木ベースライン。(ニューラルパッケージは、以下の 8 つの主要バリアントに加えて、`EndToEndHardCBM` (`e2e_hard_cbm`) バリアント — 位相折りたたみ光度曲線上の 1D-CNN 概念エンコーダ — も追加で配布する。)

| モデル | レジストリキー | 説明 |
|---|---|---|
| **HardCBM** | `hard_cbm` | Hard 概念ボトルネックモデル。入力特徴量が 12 概念のボトルネックとして機能し、MLP 予測器 (`12→64→32→6`) に供給される。基準となる解釈可能モデル。 |
| **HardCBM-Linear** | `hard_cbm_linear` | 単一の `Linear(12, 6)` 予測器 (78 パラメータ) を持つ Hard CBM。重みは概念からクラスへの寄与として直接読み取れる。最大限に解釈可能。 |
| **HardCBM-Cal** | `hard_cbm_cal` | MLP 予測器の前に抽出済み概念をノイズ除去する 12 個の独立した較正ヘッドを持つ較正済み Hard CBM。介入実験のための主要アーキテクチャ。 |
| **SoftCBM** | `soft_cbm` | 概念ごとの連続埋め込み (48 次元ボトルネック) を持つ Soft CBM。より広いボトルネックが解釈可能性をより高い精度と引き換えにする。 |
| **CEM** | `cem` | 概念埋め込みモデル (Espinosa Zarlenga et al. 2022)。各概念は、その活性化によって混合される正/負の埋め込みペアである。 |
| **MLP** | `mlp` | 素の多層パーセプトロンベースライン (`12→128→64→6`)、ボトルネックなし。ボトルネックの精度コストを測定する。 |
| **Random Forest** | `rf` | 古典的なブラックボックスベースライン (500 本の木、バランスされたクラス重み)、Gaia DR3 公式分類器 (Rimoldini et al. 2023) に整合。 |
| **XGBoost** | `xgb` | 勾配ブースティング木のブラックボックスベースライン。CBM の概念重要度との比較のために SHAP 値を計算する。 |

### 主要指標

5 分割交差検証 (`RANDOM_SEED=42`)、主要指標としてマクロ F1、フルの 18,000 ソースの Gaia DR3 データセット (CV 15,300 + テスト 2,700、クラスごとに 3,000) 上。これらは公開された *Astronomy & Astrophysics* 論文の値である (論文 Table 2)：

| モデル | 精度 (%) | マクロ F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

解釈可能性対精度のギャップは直接見て取れる：透明な HardCBM (94.4%) はブラックボックスの木ベースライン (≈ 99.8%) を約 5 パーセントポイント下回り、これはまさにすべての判断を 12 概念の物理ボトルネックに通すことの代償である。

## 概念

ボトルネックを形成する 12 個の物理概念を、固定された順序で示す (`cbm_variable_stars/shared/constants.py` に正式に定義)。

| # | 概念 | 単位 / 範囲 | 意味 |
|---|---|---|---|
| 1 | `period` | 日 | 主たる脈動/変光周期。 |
| 2 | `amplitude` | mag | 光度曲線のピークツーピーク振幅。 |
| 3 | `rise_fraction` | 無次元, [0, 1] | サイクルのうち明るさが上昇している割合。 |
| 4 | `R21` | 無次元の比 | フーリエ振幅比 A2/A1。 |
| 5 | `R31` | 無次元の比 | フーリエ振幅比 A3/A1。 |
| 6 | `phi21` | ラジアン, [0, 2π) | フーリエ位相差 φ2 − 2φ1。 |
| 7 | `skewness` | 無次元 | 等級分布の歪度。 |
| 8 | `kurtosis` | 無次元 | 等級分布の超過 (フィッシャー) 尖度。 |
| 9 | `stetson_K` | 無次元 | Stetson K 変光指数。 |
| 10 | `period_snr` | 無次元 | 周期の有意性、−log₁₀(誤警報確率)。 |
| 11 | `color_bp_rp` | mag | Gaia BP − RP 色指数。 |
| 12 | `mean_mag` | mag | Gaia G バンドの平均等級。 |

6 つのクラスは `RRAB` (RR Lyrae 型、基本モード)、`RRC` (RR Lyrae 型、第一倍音)、`DCEP` (古典的セファイド)、`DSCT_SXPHE` (たて座デルタ型 / SX Phoenicis 型)、`ECL` (食連星)、`MIRA_SR` (Mira 型 / 半規則型長周期変光星) であり、その順に 0–5 のインデックスが付けられる。

## 引用

本研究を利用する場合は、付随論文を引用していただきたい。

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

プレーンテキスト：Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990。

## ライセンス

本リポジトリは現在、明示的なオープンソースライセンスファイルを配布していないため、再利用、再配布、または改変の権利が一般的に許諾されていると想定すべきではない。上記の引用条件を超えてコードまたはデータを再利用したい場合は、許可を得るために著者 (Chenglong Yin, Sofia University) に連絡していただきたい。将来のリリースで正式なライセンスが追加される可能性がある。

## リンク

- **論文 (Astronomy & Astrophysics):** <https://doi.org/10.1051/0004-6361/202659990>
- **インタラクティブ Web コンパニオン:** <https://cbm-variable-stars.yinchenglong.com>
