<div align="center">

[English](README.md) · [中文](README.zh-CN.md) · [Čeština](README.cs.md) · [Български](README.bg.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · **Português**

</div>

# CBM Variable Stars

**Classificação interpretável de estrelas variáveis com Concept Bottleneck Models — cada previsão rastreada através de 12 conceitos estelares fisicamente significativos.**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## Visão geral

`cbm_variable_stars` é, tanto quanto sabemos, a primeira aplicação de **Concept Bottleneck Models (CBMs)** à classificação astronómica de estrelas variáveis. Classifica estrelas variáveis a partir da fotometria do Gaia DR3 em **6 classes** — RR Lyrae fundamental e de sobretom, Cefeidas clássicas, Delta Scuti / SX Phoenicis, binárias eclipsantes, e variáveis Mira / semirregulares de longo período de longa duração — encaminhando cada decisão através de **12 conceitos interpretáveis e inspecionáveis por astrónomos** em vez de um espaço de características opaco.

Com este pacote obtém:

- um pipeline completo e reprodutível, das características do Gaia DR3 (e do cruzamento de levantamentos com OGLE) até classificadores treinados;
- **8 variantes de modelo de destaque** que abrangem o espectro da interpretabilidade — desde um CBM linear totalmente transparente de 78 parâmetros até modelos de referência caixa-preta Random Forest e XGBoost;
- validação cruzada de 5 folds, ablação, intervenção e experiências de cruzamento de levantamentos, com as métricas, testes de significância, figuras e tabelas LaTeX que sustentam o artigo associado.

O artigo associado está publicado em *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), e está disponível uma demonstração web interativa em <https://cbm-variable-stars.yinchenglong.com>.

## Porquê concept bottleneck models

A maioria dos classificadores profundos mapeia as entradas brutas diretamente para um rótulo, não deixando qualquer registo inspecionável de *porque* uma dada estrela foi atribuída a uma dada classe. Em vez disso, um Concept Bottleneck Model força cada previsão a passar por uma camada estreita de **conceitos** significativos para o ser humano:

```
raw photometric features  →  12 physical concepts  →  class
```

Como toda a informação de classificação tem de fluir através do gargalo (bottleneck) de conceitos de 12 dimensões, cada previsão é rastreável a quantidades físicas que um astrónomo consegue ler e verificar — período, amplitude, parâmetros de forma de Fourier, cor, e por aí adiante. Torna também o modelo **interagível**: pode sobrescrever um conceito (por exemplo, fornecer um período corrigido) e observar a classe prevista a responder. Esta transparência tem um custo mensurável na exatidão bruta, e quantificar esse compromisso entre interpretabilidade e desempenho é um tema central deste trabalho.

## Destaques / principais resultados

- **Primeiro CBM para classificação de estrelas variáveis** — um gargalo de 12 conceitos fisicamente fundamentado sobre 6 classes de estrelas variáveis do Gaia DR3.
- **Interpretável por construção** — cada previsão é rastreável a 12 conceitos físicos nomeados, e os conceitos podem ser sobrescritos no momento da inferência para intervir numa previsão.
- **Exatidão forte e honesta** — o Hard CBM atinge **94.41% ± 0.36% de exatidão** (macro-F1 94.37% ± 0.38%, MCC 0.933) sob validação cruzada de 5 folds no conjunto de dados de destaque do Gaia DR3 com 18.000 fontes (3.000 exemplos equilibrados por classe).
- **O custo da interpretabilidade, medido** — os modelos de referência caixa-preta (Random Forest e XGBoost ≈ 99.8%) superam o transparente Hard CBM em cerca de 5 pontos percentuais, isolando o preço de um gargalo de conceitos imposto.
- **8 modelos ao longo do espectro de transparência** — Hard CBM, Hard CBM-Linear (78 parâmetros), Hard CBM-Calibrated, Soft CBM, CEM, um modelo de referência MLP, Random Forest e XGBoost.
- **Testado em cruzamento de levantamentos** — avaliado sob deslocamento de domínio Gaia → OGLE, juntamente com estudos de ablação e de intervenção em conceitos.

## Instalação

**Pré-requisitos**

- **Python ≥ 3.10** (declarado via `python_requires=">=3.10"` em `setup.py`; testado e validado em CPython 3.10–3.13).
- **Não** é necessária uma cadeia de ferramentas C/C++ — todas as dependências são distribuídas como wheels binários.
- **Não é necessária GPU.** Os modelos são pequenos (HardCBM ≈ 3K parâmetros, MLP ≈ 10K parâmetros) e a configuração de treino predefinida (batch size 256, ≤ 200 épocas com paragem antecipada) corre confortavelmente em CPU. Uma GPU CUDA apenas acelera as execuções completas com ~18.000 fontes.
- O acesso à Internet é necessário **apenas** para os passos opcionais de descarregamento de dados (consultas aos arquivos Gaia/OGLE via `astroquery`/`pyvo`); o treino sobre um conjunto de dados já construído funciona totalmente offline.

**Clone**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**Instalação**

```bash
# (recomendado) criar + ativar um ambiente virtual Python >=3.10
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# instalação editável (equivalente a `make install`)
pip install -e .
```

Em alternativa, instale o conjunto completo de dependências de runtime fixadas:

```bash
pip install -r requirements.txt
```

**Extras opcionais** (declarados em `setup.py`):

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (gráficos do espaço de conceitos t-SNE/UMAP)
pip install -e ".[explain]"    # shap>=0.43.0       (importância SHAP para modelos de referência)
pip install -e ".[dev]"        # pytest>=7.4.0      (executor de testes)
pip install -e ".[viz,explain,dev]"   # tudo
```

**Dependências principais** (instaladas automaticamente): `numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`.

> Para uma compilação CUDA explícita, instale o wheel do PyTorch correspondente **antes** de `pip install -e .`, por exemplo `pip install torch --index-url https://download.pytorch.org/whl/cu121`. O projeto não fixa uma versão de CUDA.

## Início rápido

O exemplo útil mais pequeno: carregar as características de conceitos já estandardizadas e distribuídas e executar uma passagem direta (forward pass) do HardCBM para obter os 12 valores de conceitos e uma previsão de 6 classes.

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

Isto carrega dados reais, instancia um modelo através do registo, e exercita a interface unificada `forward(x) -> {concepts, logits, probabilities}` partilhada por todas as variantes de modelo. Substitua `"hard_cbm"` por qualquer chave do registo (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`) para experimentar outras arquiteturas.

Para **treinar** em vez de usar pesos aleatórios, envolva as características no conjunto de dados e conduza a orquestração de treino:

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **Nota sobre o tamanho do conjunto de dados.** Os ficheiros parquet em `data/processed/` aqui distribuídos são uma divisão reduzida (`cv_pool` 2.550 + `test_in_domain` 450 + cruzamento de levantamentos 1.200 linhas). Os valores de destaque do artigo (HardCBM 94.41 ± 0.36% de exatidão, CV de 5 folds) provêm da tabela completa de **18.000 fontes** (`data/real/gaia_all_features.parquet`, 3.000 por classe), dividida em 15.300 CV + 2.700 teste. Veja as secções [Reproduzir os resultados](#reproduzir-os-resultados-pipeline-completo) e [Conjunto de dados](#conjunto-de-dados) para saber como reconstruir o conjunto de dados completo.

## Estrutura do repositório

Árvore anotada do código e dos dados distribuídos neste repositório (o pacote de importação é `cbm_variable_stars`).

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

> As 6 classes são `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR`; os 12 conceitos são `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` — ambos definidos de forma autoritativa em `cbm_variable_stars/shared/constants.py`.

## Reproduzir os resultados (pipeline completo)

O pipeline ponta-a-ponta vai da fotometria bruta do Gaia DR3 (e do OGLE) a modelos treinados, métricas validadas por validação cruzada e as figuras/tabelas do artigo. Cada etapa é um script numerado em `scripts/`, conduzido por um único ficheiro de configuração (`configs/default.yaml`). Executar o pipeline regenera um diretório local `results/`; os modelos treinados, métricas e figuras publicados encontram-se no artigo da *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)) e não nesta árvore de código.

### Pré-requisitos

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

As etapas 01–02 fazem chamadas de rede aos arquivos Gaia TAP e OGLE, pelo que necessitam de acesso à Internet. O treino (06) corre em CPU por predefinição; passe `--device cuda` para usar uma GPU.

### Passo a passo (os scripts efetivamente em disco)

Todos os comandos correm a partir da raiz do repositório; cada script lê `configs/default.yaml` por predefinição (sobreponha com `--config`).

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

Notas sobre as opções (flags) mais usadas:

- **06** requer `--data_path`; a sua lista `--models` predefinida é o subconjunto de 5 modelos `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`. Passe a lista completa de 8 modelos (acima) para reproduzir a comparação de destaque do artigo. Outras predefinições: `--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`.
- **05** opções: `--no-ogle` (apenas Gaia), `--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (predefinição `10dim`), `--verify-folds`.
- **07** opções: `--device cuda`, e `--skip-training`/`--skip-ablation` etc. para executar subconjuntos.
- **08** opções: `--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`.

### Etapa → script → saída

| Etapa | Script | Entradas principais | Saídas principais |
|------|--------|------------|-------------|
| 1. Descarregar Gaia | `scripts/01_download_gaia.py` | Arquivo Gaia TAP | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. Descarregar OGLE | `scripts/02_download_ogle.py` | Arquivo OGLE-IV | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. Extrair características | `scripts/03_extract_features.py` | curvas de luz brutas | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. Validar características | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, relatório de qualidade |
| 5. Construir conjunto de dados | `scripts/05_build_dataset.py` | características validadas | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. Treinar modelos | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | resultados de CV por modelo + checkpoints, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. Executar experiências | `scripts/07_run_experiments.py` | dados processados + modelos treinados | JSON de ablação / intervenção / cruzamento de levantamentos / curva de aprendizagem |
| 8. Gerar figuras | `scripts/08_generate_figures.py` | JSON de resultados | figuras do artigo (PDF) + tabelas LaTeX |

### Configuração

Todas as execuções são conduzidas por **`configs/default.yaml`** (carregado via OmegaConf; passe `--config <file>` a qualquer script para sobrepor). É a única fonte de verdade para: semente aleatória (`project.random_seed: 42`); esquema de divisão (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`, estratificado, StandardScaler); alvos de descarregamento por classe (`var_types.*`); parâmetros de extração de características (`features.*` — pesquisa de período, harmónicas de Fourier, deteção de aliases no período de precessão do Gaia de 63 dias, etc.); hiperparâmetros de treino (`training.*` — batch 256, lr 1e-3, máx. 200 épocas, patience 15, agendamento cosseno com reinício a quente); arquitetura por modelo (`models.*`); e grelhas de experiências (`experiments.*`). Existem dois ficheiros de configuração auxiliares: `configs/feature_config.yaml` e `configs/gaia_queries.yaml`.

### Notas de reprodutibilidade / âmbito dos dados

- **Semente e folds.** `random_seed=42` e a CV estratificada de 5 folds estão fixos em `configs/default.yaml`; `05_build_dataset.py` escreve índices de folds determinísticos em `data/processed/cv_folds.pkl`.
- **Conjunto de dados de destaque vs. divisão distribuída.** Os números de destaque do artigo (HardCBM 94.41% ± 0.36% de exatidão) provêm da matriz Gaia completa e equilibrada de **18.000 fontes** (3.000/classe; 15.300 CV + 2.700 teste). O `data/processed/cv_pool.parquet` distribuído *nesta árvore* é uma **divisão de demonstração reduzida** (2.550 linhas, 425/classe); executar sobre ele não reproduzirá a exatidão de destaque. Reexecutar as etapas de dados reconstrói o conjunto de dados completo a partir dos arquivos; a matriz de características completa também está disponível como `data/real/gaia_all_features.parquet` (18.000 linhas).
- **OGLE sob demanda.** `data/raw/ogle/` é distribuído vazio por desenho — a etapa 02 descarrega as curvas de luz do OGLE em tempo de execução (rede necessária para o teste de cruzamento de levantamentos).

> **Nota sobre o Makefile.** Os alvos do `make` espelham as fases acima (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`). Os alvos `data:` e `train:` submetidos estão ligeiramente desalinhados com os scripts atuais — `make data` referencia nomes de scripts mais antigos e `make train` omite o argumento `--data_path` requerido — pelo que os comandos explícitos `python scripts/0N_*.py` acima são as invocações canónicas e funcionais.

## Conjunto de dados

O repositório distribui um **conjunto de dados de características de estrelas variáveis do Gaia DR3** em `data/`, organizado como uma cascata reprodutível, da fotometria bruta até divisões normalizadas e prontas para validação cruzada. Todos os dados tabulares são armazenados em Apache Parquet; os índices de folds de validação cruzada e o scaler ajustado são pickle (`.pkl`); o mapa de rótulos é JSON.

O conjunto de dados abrange **6 classes de estrelas variáveis** descritas por **12 conceitos fisicamente significativos** (ver a tabela [Conceitos](#conceitos)). A tabela de estudo de destaque contém **18.000 fontes do Gaia DR3** equilibradas a **3.000 por classe** (`data/real/gaia_all_features.parquet`, em unidades físicas/não escaladas), dividida num pool de validação cruzada de 5 folds com 15.300 fontes e num conjunto de teste de retenção (hold-out) com 2.700 fontes.

Organização sob `data/`:

| Diretório | Conteúdo |
|---|---|
| `raw/gaia/epoch_photometry/` | Curvas de luz na banda G do Gaia DR3 por fonte, um Parquet por `<source_id>` (colunas `time, mag, mag_err`). |
| `raw/gaia/metadata/` | Metadados de fonte por classe e combinados (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`). |
| `raw/ogle/` | Vazio por desenho; as curvas de luz do cruzamento de levantamentos OGLE são descarregadas sob demanda. |
| `interim/` | Características extraídas antes da divisão (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`). |
| `processed/` | Divisões prontas para validação cruzada, normalizadas por StandardScaler (z-score): `cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE, fora de domínio), além de `cv_folds.pkl` (índices StratifiedKFold de 5 folds), `scaler.pkl`, e `label_mapping.json`. |
| `expanded/` | Variante aumentada maior (`gaia_expanded_features.parquet`, 30.000 linhas). |
| `real/` | A tabela de características de destaque com 18.000 fontes em unidades físicas (`gaia_all_features.parquet`) mais metadados brutos. |

Cada linha de características traz colunas de identificador/rótulo/qualidade (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`) seguidas das 12 colunas de conceitos. O `StandardScaler` global é ajustado apenas no pool de validação cruzada e aplicado a ambos os conjuntos de teste; `period_snr` é imputado pela mediana no momento da escala.

> **Nota.** Os ficheiros parquet em `data/processed/` distribuídos nesta árvore são uma divisão de demonstração reduzida (`cv_pool` 2.550 + `test_in_domain` 450 = 3.000 linhas; teste de cruzamento de levantamentos 1.200). A matriz completa de 18.000 fontes usada para os resultados de destaque publicados corresponde a `data/real/gaia_all_features.parquet`.

## Modelos

São comparadas oito variantes de modelo: seis redes neuronais e dois modelos de referência clássicos baseados em árvores. (O pacote neuronal distribui adicionalmente uma variante `EndToEndHardCBM` (`e2e_hard_cbm`) — um codificador de conceitos 1D-CNN sobre curvas de luz dobradas em fase — para além das oito variantes de destaque abaixo.)

| Modelo | Chave do registo | Descrição |
|---|---|---|
| **HardCBM** | `hard_cbm` | Hard Concept Bottleneck Model; as características de entrada servem como o gargalo de 12 conceitos que alimenta um preditor MLP (`12→64→32→6`). O modelo interpretável de referência. |
| **HardCBM-Linear** | `hard_cbm_linear` | Hard CBM com um único preditor `Linear(12, 6)` (78 parâmetros); os pesos lêem-se diretamente como contribuições de conceito para classe. Interpretabilidade máxima. |
| **HardCBM-Cal** | `hard_cbm_cal` | Hard CBM calibrado com 12 cabeças de calibração independentes que removem ruído dos conceitos extraídos antes de um preditor MLP; arquitetura principal para as experiências de intervenção. |
| **SoftCBM** | `soft_cbm` | Soft CBM com embeddings contínuos por conceito (gargalo de 48 dimensões); um gargalo mais largo troca interpretabilidade por maior exatidão. |
| **CEM** | `cem` | Concept Embedding Model (Espinosa Zarlenga et al. 2022); cada conceito é um par de embeddings positivo/negativo combinado pela sua ativação. |
| **MLP** | `mlp` | Modelo de referência perceptrão multicamada simples (`12→128→64→6`), sem gargalo; mede o custo de exatidão do gargalo. |
| **Random Forest** | `rf` | Modelo de referência clássico caixa-preta (500 árvores, pesos de classe equilibrados), alinhado com o classificador oficial do Gaia DR3 (Rimoldini et al. 2023). |
| **XGBoost** | `xgb` | Modelo de referência caixa-preta de árvores com gradient boosting; valores SHAP calculados para comparação com a importância de conceitos do CBM. |

### Métricas de destaque

Validação cruzada de 5 folds (`RANDOM_SEED=42`), macro-F1 como métrica primária, no conjunto de dados completo do Gaia DR3 com 18.000 fontes (15.300 CV + 2.700 teste; 3.000 por classe). Estes são os valores publicados no artigo da *Astronomy & Astrophysics* (Tabela 2 do artigo):

| Modelo | Exatidão (%) | Macro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

A diferença entre interpretabilidade e exatidão é visível diretamente: o transparente HardCBM (94.4%) fica atrás dos modelos de referência caixa-preta de árvores (≈ 99.8%) em cerca de 5 pontos percentuais, que é precisamente o custo de forçar cada decisão a passar por um gargalo físico de 12 conceitos.

## Conceitos

Os 12 conceitos físicos que formam o gargalo, em ordem fixa (definidos de forma autoritativa em `cbm_variable_stars/shared/constants.py`).

| # | Conceito | Unidade / Intervalo | Significado |
|---|---|---|---|
| 1 | `period` | dias | Período primário de pulsação/variabilidade. |
| 2 | `amplitude` | mag | Amplitude pico-a-pico da curva de luz. |
| 3 | `rise_fraction` | adimensional, [0, 1] | Fração do ciclo passada a aumentar de brilho. |
| 4 | `R21` | rácio adimensional | Rácio de amplitudes de Fourier A2/A1. |
| 5 | `R31` | rácio adimensional | Rácio de amplitudes de Fourier A3/A1. |
| 6 | `phi21` | radianos, [0, 2π) | Diferença de fase de Fourier φ2 − 2φ1. |
| 7 | `skewness` | adimensional | Assimetria (skewness) da distribuição de magnitudes. |
| 8 | `kurtosis` | adimensional | Curtose em excesso (de Fisher) da distribuição de magnitudes. |
| 9 | `stetson_K` | adimensional | Índice de variabilidade K de Stetson. |
| 10 | `period_snr` | adimensional | Significância do período, −log₁₀(probabilidade de falso alarme). |
| 11 | `color_bp_rp` | mag | Índice de cor BP − RP do Gaia. |
| 12 | `mean_mag` | mag | Magnitude média na banda G do Gaia. |

As seis classes são `RRAB` (RR Lyrae, modo fundamental), `RRC` (RR Lyrae, primeiro sobretom), `DCEP` (Cefeida clássica), `DSCT_SXPHE` (Delta Scuti / SX Phoenicis), `ECL` (binária eclipsante), e `MIRA_SR` (Mira / variável semirregular de longo período), indexadas de 0 a 5 nessa ordem.

## Citação

Se usar este trabalho, por favor cite o artigo associado:

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

Texto simples: Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990.

## Licença

Este repositório não distribui atualmente um ficheiro de licença de código aberto explícito, pelo que não deve ser assumida qualquer concessão geral de direitos de reutilização, redistribuição ou modificação. Se pretender reutilizar o código ou os dados para além dos termos de citação acima, por favor contacte o autor (Chenglong Yin, Sofia University) para acertar a autorização. Uma licença formal poderá ser adicionada numa versão futura.

## Ligações

- **Artigo (Astronomy & Astrophysics):** <https://doi.org/10.1051/0004-6361/202659990>
- **Companheiro web interativo:** <https://cbm-variable-stars.yinchenglong.com>
