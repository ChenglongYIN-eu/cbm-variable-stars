<div align="center">

[English](README.md) · [中文](README.zh-CN.md) · **Čeština** · [Български](README.bg.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Русский](README.ru.md) · [日本語](README.ja.md) · [Português](README.pt.md)

</div>

# CBM Variable Stars

**Interpretovatelná klasifikace proměnných hvězd pomocí Concept Bottleneck Models — každá predikce vysledovatelná prostřednictvím 12 fyzikálně smysluplných hvězdných konceptů.**

[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202659990-blue)](https://doi.org/10.1051/0004-6361/202659990)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/)

---

## Přehled

`cbm_variable_stars` je, pokud je nám známo, prvním použitím **Concept Bottleneck Models (CBMs)** na astronomickou klasifikaci proměnných hvězd. Klasifikuje proměnné hvězdy z fotometrie Gaia DR3 do **6 tříd** — RR Lyrae v základním a vyšším harmonickém módu, klasické cefeidy, Delta Scuti / SX Phoenicis, zákrytové dvojhvězdy a dlouhoperiodické proměnné typu Mira / semiregulární — tím, že každé rozhodnutí směruje přes **12 interpretovatelných konceptů, které může astronom prozkoumat**, namísto neprůhledného prostoru příznaků.

S tímto balíčkem získáte:

- kompletní, reprodukovatelný řetězec zpracování od příznaků Gaia DR3 (a křížového přehlídkového porovnání s OGLE) k natrénovaným klasifikátorům;
- **8 hlavních variant modelu** pokrývajících celé spektrum interpretovatelnosti — od plně transparentního lineárního CBM se 78 parametry až po black-box základní modely Random Forest a XGBoost;
- 5násobnou křížovou validaci, ablační, intervenční a křížové přehlídkové experimenty, spolu s metrikami, testy významnosti, obrázky a tabulkami LaTeX, které dokládají doprovodný článek.

Doprovodný článek je publikován v časopise *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)) a interaktivní webové demo je dostupné na <https://cbm-variable-stars.yinchenglong.com>.

## Proč concept bottleneck models

Většina hlubokých klasifikátorů mapuje surové vstupy přímo na štítek, aniž by ponechala jakýkoli prozkoumatelný záznam o tom, *proč* byla daná hvězda přiřazena k dané třídě. Concept Bottleneck Model místo toho nutí každou predikci projít úzkou vrstvou pro člověka smysluplných **konceptů**:

```
raw photometric features  →  12 physical concepts  →  class
```

Protože veškerá klasifikační informace musí protéct 12rozměrným konceptovým hrdlem, je každá predikce vysledovatelná k fyzikálním veličinám, které dokáže astronom přečíst a ověřit — perioda, amplituda, parametry Fourierova tvaru, barva a tak dále. Díky tomu je model také **interaktivní**: můžete přepsat koncept (například dodat opravenou periodu) a sledovat, jak na to predikovaná třída zareaguje. Tato transparentnost má měřitelnou cenu v hrubé přesnosti a kvantifikace tohoto kompromisu mezi interpretovatelností a výkonem je ústředním tématem této práce.

## Hlavní body / klíčové výsledky

- **První CBM pro klasifikaci proměnných hvězd** — fyzikálně podložené konceptové hrdlo o 12 konceptech nad 6 třídami proměnných hvězd z Gaia DR3.
- **Interpretovatelný už ze své konstrukce** — každá predikce je vysledovatelná k 12 pojmenovaným fyzikálním konceptům a koncepty lze v době inference přepsat a zasáhnout tak do predikce.
- **Silná, poctivá přesnost** — Hard CBM dosahuje **přesnosti 94.41% ± 0.36%** (makro-F1 94.37% ± 0.38%, MCC 0.933) při 5násobné křížové validaci na hlavním souboru dat Gaia DR3 s 18 000 zdroji (3 000 vyvážených příkladů na třídu).
- **Změřená cena interpretovatelnosti** — black-box základní modely (Random Forest a XGBoost ≈ 99.8%) překonávají transparentní Hard CBM zhruba o 5 procentních bodů, čímž izolují cenu vynuceného konceptového hrdla.
- **8 modelů napříč spektrem transparentnosti** — Hard CBM, Hard CBM-Linear (78 parametrů), Hard CBM-Calibrated, Soft CBM, CEM, základní model MLP, Random Forest a XGBoost.
- **Otestováno napříč přehlídkami** — vyhodnoceno za doménového posunu Gaia → OGLE, vedle ablačních studií a studií intervence do konceptů.

## Instalace

**Předpoklady**

- **Python ≥ 3.10** (deklarováno přes `python_requires=">=3.10"` v `setup.py`; ověřeně funkční na CPython 3.10–3.13).
- Sada nástrojů C/C++ **není** vyžadována — všechny závislosti se distribuují jako binární wheely.
- **GPU není vyžadováno.** Modely jsou malé (HardCBM ≈ 3K parametrů, MLP ≈ 10K parametrů) a výchozí trénovací konfigurace (velikost dávky 256, ≤ 200 epoch s předčasným ukončením) běží pohodlně na CPU. CUDA GPU pouze urychluje plné běhy s ~18 000 zdroji.
- Přístup k internetu je potřeba **pouze** pro volitelné kroky stahování dat (dotazy na archivy Gaia/OGLE přes `astroquery`/`pyvo`); trénování na již sestaveném souboru dat funguje plně offline.

**Klonování**

```bash
git clone <repo-url> cbm_variable_stars
cd cbm_variable_stars
```

**Instalace**

```bash
# (doporučeno) vytvořte + aktivujte virtuální prostředí Pythonu >=3.10
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editovatelná instalace (ekvivalent `make install`)
pip install -e .
```

Případně nainstalujte plnou sadu připnutých běhových závislostí:

```bash
pip install -r requirements.txt
```

**Volitelné doplňky** (deklarované v `setup.py`):

```bash
pip install -e ".[viz]"        # umap-learn>=0.5.4  (t-SNE/UMAP concept-space plots)
pip install -e ".[explain]"    # shap>=0.43.0       (SHAP importance for baselines)
pip install -e ".[dev]"        # pytest>=7.4.0      (test runner)
pip install -e ".[viz,explain,dev]"   # everything
```

**Základní závislosti** (stahované automaticky): `numpy>=1.24`, `pandas>=2.1`, `scipy>=1.11`, `scikit-learn>=1.3`, `torch>=2.1`, `xgboost>=2.0`, `astropy>=6.0`, `astroquery>=0.4.7`, `pyvo>=1.5`, `pyarrow>=14.0`, `pyyaml>=6.0`, `omegaconf>=2.3`, `matplotlib>=3.8`, `seaborn>=0.13`, `loguru>=0.7`, `tqdm>=4.66`, `requests>=2.31`.

> Pro explicitní sestavení s CUDA nainstalujte odpovídající wheel PyTorch **před** `pip install -e .`, např. `pip install torch --index-url https://download.pytorch.org/whl/cu121`. Projekt nepřipíná konkrétní verzi CUDA.

## Rychlý start

Nejmenší užitečný příklad: načtěte dodané, již standardizované konceptové příznaky a spusťte dopředný průchod HardCBM, abyste získali 12 konceptových hodnot a predikci do 6 tříd.

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

Tím se načtou reálná data, instancuje model přes registr a procvičí se sjednocené rozhraní `forward(x) -> {concepts, logits, probabilities}` sdílené každou variantou modelu. Zaměňte `"hard_cbm"` za libovolný klíč registru (`hard_cbm_linear`, `hard_cbm_cal`, `e2e_hard_cbm`, `soft_cbm`, `cem`, `mlp`) a vyzkoušejte jiné architektury.

Pro **trénování** namísto použití náhodných vah zabalte příznaky do datasetu a řiďte orchestraci trénování:

```python
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

ds     = VariableStarDataset(df[CONCEPT_NAMES].values, df["label_name"].values)
loader = create_dataloader(ds, batch_size=256, shuffle=True)
# then drive training via cbm_variable_stars.training.trainer.train_cbm(...)
```

> **Poznámka k velikosti souboru dat.** Parquety v `data/processed/` zde dodané jsou redukované rozdělení (`cv_pool` 2 550 + `test_in_domain` 450 + křížová přehlídka 1 200 řádků). Hlavní hodnoty z článku (HardCBM přesnost 94.41 ± 0.36%, 5násobná CV) pocházejí z plné tabulky s **18 000 zdroji** (`data/real/gaia_all_features.parquet`, 3 000 na třídu), rozdělené na 15 300 CV + 2 700 test. Jak znovu sestavit plný soubor dat, viz sekce [Reprodukce výsledků](#reprodukce-výsledků-plný-řetězec-zpracování) a [Soubor dat](#soubor-dat).

## Struktura repozitáře

Komentovaný strom kódu a dodaných dat v tomto repozitáři (importní balíček je `cbm_variable_stars`).

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

> 6 tříd je `RRAB, RRC, DCEP, DSCT_SXPHE, ECL, MIRA_SR`; 12 konceptů je `period, amplitude, rise_fraction, R21, R31, phi21, skewness, kurtosis, stetson_K, period_snr, color_bp_rp, mean_mag` — obojí je závazně definováno v `cbm_variable_stars/shared/constants.py`.

## Reprodukce výsledků (plný řetězec zpracování)

Komplexní řetězec zpracování vede od surové fotometrie Gaia DR3 (a OGLE) k natrénovaným modelům, křížově validovaným metrikám a obrázkům/tabulkám článku. Každá fáze je číslovaný skript pod `scripts/`, řízený jediným konfiguračním souborem (`configs/default.yaml`). Spuštění řetězce regeneruje lokální adresář `results/`; publikované natrénované modely, metriky a obrázky se nacházejí v článku *Astronomy & Astrophysics* (DOI [10.1051/0004-6361/202659990](https://doi.org/10.1051/0004-6361/202659990)), nikoli v tomto stromu kódu.

### Předpoklady

```bash
pip install -e .                  # Python >= 3.10; installs the cbm_variable_stars package
pip install -e ".[viz,explain]"   # optional: umap-learn (embeddings), shap (concept importance)
```

Fáze 01–02 provádějí síťová volání na archivy Gaia TAP a OGLE, takže potřebují přístup k internetu. Trénování (06) běží ve výchozím nastavení na CPU; pro GPU předejte `--device cuda`.

### Krok za krokem (skutečné skripty na disku)

Všechny příkazy se spouštějí z kořene repozitáře; každý skript ve výchozím nastavení čte `configs/default.yaml` (přepište pomocí `--config`).

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

Poznámky k nejpoužívanějším přepínačům:

- **06** vyžaduje `--data_path`; jeho výchozí seznam `--models` je 5modelová podmnožina `hard_cbm hard_cbm_linear hard_cbm_cal mlp rf`. Pro reprodukci hlavního srovnání z článku předejte plný seznam 8 modelů (výše). Další výchozí hodnoty: `--output_dir results`, `--seed 42`, `--max_epochs 200`, `--patience 15`, `--device cpu`.
- **05** možnosti: `--no-ogle` (pouze Gaia), `--ogle-mode {10dim,12dim_with_match,12dim_fill_median}` (výchozí `10dim`), `--verify-folds`.
- **07** možnosti: `--device cuda` a `--skip-training`/`--skip-ablation` atd. pro spuštění podmnožin.
- **08** možnosti: `--figures-only`, `--tables-only`, `--results-dir`, `--output-dir`.

### Krok → skript → výstup

| Krok | Skript | Klíčové vstupy | Klíčové výstupy |
|------|--------|------------|-------------|
| 1. Stažení Gaia | `scripts/01_download_gaia.py` | archiv Gaia TAP | `data/raw/gaia/metadata/*.parquet`, `data/raw/gaia/epoch_photometry/<source_id>.parquet` |
| 2. Stažení OGLE | `scripts/02_download_ogle.py` | archiv OGLE-IV | `data/raw/ogle/metadata/`, `data/raw/ogle/light_curves/` |
| 3. Extrakce příznaků | `scripts/03_extract_features.py` | surové světelné křivky | `data/interim/gaia_features_raw.parquet`, `ogle_features_raw.parquet` |
| 4. Validace příznaků | `scripts/04_validate_features.py` | `data/interim/*_raw.parquet` | `data/interim/*_features_validated.parquet`, zpráva o kvalitě |
| 5. Sestavení souboru dat | `scripts/05_build_dataset.py` | validované příznaky | `data/processed/{cv_pool,test_in_domain,test_cross_survey}.parquet`, `scaler.pkl`, `cv_folds.pkl`, `label_mapping.json` |
| 6. Trénování modelů | `scripts/06_train_models.py` | `data/processed/cv_pool.parquet` (+ OGLE) | výsledky CV pro jednotlivé modely + checkpointy, `comparison_table.{csv,tex}`, `significance_tests.json` |
| 7. Spuštění experimentů | `scripts/07_run_experiments.py` | zpracovaná data + natrénované modely | JSON ablace / intervence / křížové přehlídky / učicí křivky |
| 8. Generování obrázků | `scripts/08_generate_figures.py` | JSON s výsledky | obrázky článku (PDF) + tabulky LaTeX |

### Konfigurace

Všechny běhy jsou řízeny souborem **`configs/default.yaml`** (načítaným přes OmegaConf; pro přepsání předejte libovolnému skriptu `--config <file>`). Je jediným zdrojem pravdy pro: náhodné semínko (`project.random_seed: 42`); schéma rozdělení (`dataset.test_in_domain_ratio: 0.15`, `n_cv_folds: 5`, stratifikované, StandardScaler); cílové počty stahování na třídu (`var_types.*`); parametry extrakce příznaků (`features.*` — hledání periody, Fourierovy harmonické, detekce aliasů na 63denní precesní periodě Gaia atd.); trénovací hyperparametry (`training.*` — dávka 256, lr 1e-3, max 200 epoch, patience 15, kosinový rozvrh s teplými restarty); architekturu jednotlivých modelů (`models.*`); a experimentální mřížky (`experiments.*`). Existují dvě pomocné konfigurace: `configs/feature_config.yaml` a `configs/gaia_queries.yaml`.

### Poznámky k reprodukovatelnosti / rozsahu dat

- **Semínko a foldy.** `random_seed=42` a 5násobná stratifikovaná CV jsou pevně dány v `configs/default.yaml`; `05_build_dataset.py` zapisuje deterministické indexy foldů do `data/processed/cv_folds.pkl`.
- **Hlavní soubor dat vs. dodané rozdělení.** Hlavní čísla z článku (HardCBM přesnost 94.41% ± 0.36%) pocházejí z plné vyvážené matice Gaia s **18 000 zdroji** (3 000/třídu; 15 300 CV + 2 700 test). Soubor `data/processed/cv_pool.parquet` dodaný *v tomto stromu* je **redukované demonstrační rozdělení** (2 550 řádků, 425/třídu); běh nad ním hlavní přesnost nereprodukuje. Opětovné spuštění datových fází znovu sestaví plný soubor dat z archivů; plná matice příznaků je dostupná také jako `data/real/gaia_all_features.parquet` (18 000 řádků).
- **OGLE na vyžádání.** `data/raw/ogle/` je záměrně dodán prázdný — krok 02 stahuje světelné křivky OGLE za běhu (pro křížový přehlídkový test je vyžadována síť).

> **Poznámka k Makefile.** Cíle `make` zrcadlí výše uvedené fáze (`make install`, `make data`, `make train`, `make experiments`, `make figures`, `make all`). Zakomitované cíle `data:` a `train:` jsou mírně mimo soulad s aktuálními skripty — `make data` odkazuje na starší názvy skriptů a `make train` vynechává povinný argument `--data_path` — takže explicitní příkazy `python scripts/0N_*.py` výše jsou kanonickými, funkčními voláními.

## Soubor dat

Repozitář dodává **soubor dat příznaků proměnných hvězd Gaia DR3** pod `data/`, organizovaný jako reprodukovatelná kaskáda od surové fotometrie po normalizovaná rozdělení připravená ke křížové validaci. Všechna tabulková data jsou uložena jako Apache Parquet; indexy foldů křížové validace a natrénovaný škálovač jsou pickle (`.pkl`); mapa štítků je JSON.

Soubor dat pokrývá **6 tříd proměnných hvězd** popsaných **12 fyzikálně smysluplnými koncepty** (viz tabulka [Koncepty](#koncepty)). Hlavní studijní tabulka obsahuje **18 000 zdrojů Gaia DR3** vyvážených na **3 000 na třídu** (`data/real/gaia_all_features.parquet`, ve fyzikálních/neškálovaných jednotkách), rozdělených do 5násobného křížově validačního poolu s 15 300 zdroji a testovací množiny stranou s 2 700 zdroji.

Organizace pod `data/`:

| Adresář | Obsah |
|---|---|
| `raw/gaia/epoch_photometry/` | Světelné křivky Gaia DR3 v pásmu G pro jednotlivé zdroje, jeden Parquet na `<source_id>` (sloupce `time, mag, mag_err`). |
| `raw/gaia/metadata/` | Metadata zdrojů pro jednotlivé třídy i kombinovaná (`source_id, best_class_name, best_class_score, phot_g_mean_mag, bp_rp, parallax`). |
| `raw/ogle/` | Záměrně prázdný; světelné křivky OGLE pro křížovou přehlídku se stahují na vyžádání. |
| `interim/` | Extrahované příznaky před rozdělením (`gaia_features_raw.parquet`, `ogle_features_raw.parquet`). |
| `processed/` | Rozdělení připravená ke křížové validaci, normalizovaná StandardScalerem (z-skóre): `cv_pool.parquet`, `test_in_domain.parquet`, `test_cross_survey.parquet` (OGLE, mimo doménu), plus `cv_folds.pkl` (indexy 5násobného StratifiedKFold), `scaler.pkl` a `label_mapping.json`. |
| `expanded/` | Větší augmentovaná varianta (`gaia_expanded_features.parquet`, 30 000 řádků). |
| `real/` | Hlavní tabulka příznaků s 18 000 zdroji ve fyzikálních jednotkách (`gaia_all_features.parquet`) plus surová metadata. |

Každý řádek příznaků nese sloupce identifikátor/štítek/kvalita (`source_id, label, label_name, source, n_obs, quality_flag, alias_flag`) následované 12 konceptovými sloupci. Globální `StandardScaler` je natrénován pouze na křížově validačním poolu a aplikován na obě testovací množiny; `period_snr` je v době škálování imputován mediánem.

> **Poznámka.** Parquety v `data/processed/` dodané v tomto stromu jsou redukované demonstrační rozdělení (`cv_pool` 2 550 + `test_in_domain` 450 = 3 000 řádků; křížový přehlídkový test 1 200). Plná matice s 18 000 zdroji použitá pro publikované hlavní výsledky odpovídá `data/real/gaia_all_features.parquet`.

## Modely

Porovnává se osm variant modelu: šest neuronových sítí a dva klasické stromové základní modely. (Neuronový balíček navíc dodává variantu `EndToEndHardCBM` (`e2e_hard_cbm`) — konceptový enkodér s 1D-CNN nad fázově složenými světelnými křivkami — nad rámec osmi hlavních variant níže.)

| Model | Klíč registru | Popis |
|---|---|---|
| **HardCBM** | `hard_cbm` | Hard Concept Bottleneck Model; vstupní příznaky slouží jako konceptové hrdlo o 12 konceptech napájející MLP prediktor (`12→64→32→6`). Referenční interpretovatelný model. |
| **HardCBM-Linear** | `hard_cbm_linear` | Hard CBM s jediným prediktorem `Linear(12, 6)` (78 parametrů); váhy se čtou přímo jako příspěvky konceptu k třídě. Maximálně interpretovatelný. |
| **HardCBM-Cal** | `hard_cbm_cal` | Kalibrovaný Hard CBM s 12 nezávislými kalibračními hlavami, které odšumují extrahované koncepty před MLP prediktorem; primární architektura pro intervenční experimenty. |
| **SoftCBM** | `soft_cbm` | Soft CBM se spojitými embeddingy pro jednotlivé koncepty (48rozměrné hrdlo); širší hrdlo vyměňuje interpretovatelnost za vyšší přesnost. |
| **CEM** | `cem` | Concept Embedding Model (Espinosa Zarlenga et al. 2022); každý koncept je dvojice pozitivního/negativního embeddingu míchaná jeho aktivací. |
| **MLP** | `mlp` | Prostý vícevrstvý perceptron jako základní model (`12→128→64→6`), bez hrdla; měří cenu hrdla v přesnosti. |
| **Random Forest** | `rf` | Klasický black-box základní model (500 stromů, vyvážené váhy tříd), sladěný s oficiálním klasifikátorem Gaia DR3 (Rimoldini et al. 2023). |
| **XGBoost** | `xgb` | Black-box základní model s gradientně zesilovanými stromy; hodnoty SHAP počítány pro srovnání s konceptovou důležitostí CBM. |

### Hlavní metriky

5násobná křížová validace (`RANDOM_SEED=42`), makro-F1 jako primární metrika, na plném souboru dat Gaia DR3 s 18 000 zdroji (15 300 CV + 2 700 test; 3 000 na třídu). Jsou to publikované hodnoty z článku *Astronomy & Astrophysics* (Tabulka 2 článku):

| Model | Přesnost (%) | Makro-F1 (%) | MCC |
|---|---|---|---|
| XGBoost | 99.81 ± 0.11 | 99.81 ± 0.11 | 0.998 |
| Random Forest | 99.79 ± 0.09 | 99.79 ± 0.09 | 0.998 |
| SoftCBM | 99.12 ± 0.29 | 99.12 ± 0.29 | 0.989 |
| CEM | 97.13 ± 0.36 | 97.13 ± 0.37 | 0.965 |
| HardCBM-Cal | 96.97 ± 0.47 | 96.96 ± 0.47 | 0.964 |
| MLP | 95.84 ± 0.29 | 95.83 ± 0.29 | 0.950 |
| **HardCBM** | **94.41 ± 0.36** | **94.37 ± 0.38** | **0.933** |
| HardCBM-Linear | 90.67 ± 0.85 | 90.59 ± 0.87 | 0.888 |

Rozdíl mezi interpretovatelností a přesností je viditelný přímo: transparentní HardCBM (94.4%) zaostává za black-box stromovými základními modely (≈ 99.8%) zhruba o 5 procentních bodů, což je přesně cena za vynucení každého rozhodnutí přes fyzikální hrdlo o 12 konceptech.

## Koncepty

12 fyzikálních konceptů tvořících hrdlo, v pevném pořadí (závazně definováno v `cbm_variable_stars/shared/constants.py`).

| # | Koncept | Jednotka / Rozsah | Význam |
|---|---|---|---|
| 1 | `period` | dny | Primární perioda pulzace/proměnnosti. |
| 2 | `amplitude` | mag | Amplituda světelné křivky špička–špička. |
| 3 | `rise_fraction` | bezrozměrné, [0, 1] | Podíl cyklu stráveného nárůstem jasnosti. |
| 4 | `R21` | bezrozměrný poměr | Poměr Fourierových amplitud A2/A1. |
| 5 | `R31` | bezrozměrný poměr | Poměr Fourierových amplitud A3/A1. |
| 6 | `phi21` | radiány, [0, 2π) | Fourierův fázový rozdíl φ2 − 2φ1. |
| 7 | `skewness` | bezrozměrné | Šikmost rozdělení magnitud. |
| 8 | `kurtosis` | bezrozměrné | Nadbytečná (Fisherova) špičatost rozdělení magnitud. |
| 9 | `stetson_K` | bezrozměrné | Stetsonův index proměnnosti K. |
| 10 | `period_snr` | bezrozměrné | Významnost periody, −log₁₀(pravděpodobnost falešného poplachu). |
| 11 | `color_bp_rp` | mag | Barevný index Gaia BP − RP. |
| 12 | `mean_mag` | mag | Střední magnituda Gaia v pásmu G. |

Šest tříd je `RRAB` (RR Lyrae, základní mód), `RRC` (RR Lyrae, první vyšší harmonický mód), `DCEP` (klasická cefeida), `DSCT_SXPHE` (Delta Scuti / SX Phoenicis), `ECL` (zákrytová dvojhvězda) a `MIRA_SR` (Mira / semiregulární dlouhoperiodická proměnná), indexované 0–5 v tomto pořadí.

## Citace

Pokud tuto práci použijete, citujte prosím doprovodný článek:

```bibtex
@article{Yin2026CBM,
  author  = {Yin, Chenglong},
  title   = {Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  doi     = {10.1051/0004-6361/202659990}
}
```

Prostý text: Yin, C. 2026, *Concept Bottleneck Models for Interpretable Variable Star Classification with Gaia DR3*, Astronomy & Astrophysics, DOI: 10.1051/0004-6361/202659990.

## Licence

Tento repozitář v současné době nedodává explicitní soubor s open-source licencí, takže by se neměl předpokládat žádný obecný udělený nárok na opětovné použití, redistribuci ani úpravu. Pokud byste chtěli znovu použít kód nebo data nad rámec výše uvedených citačních podmínek, kontaktujte prosím autora (Chenglong Yin, Sofia University) za účelem sjednání povolení. Formální licence může být přidána v budoucím vydání.

## Odkazy

- **Článek (Astronomy & Astrophysics):** <https://doi.org/10.1051/0004-6361/202659990>
- **Interaktivní webový doprovod:** <https://cbm-variable-stars.yinchenglong.com>
