"""
CBM变星分类项目 -- 全局共享常量定义

本文件是三个管线（数据、模型、实验）的唯一权威概念定义来源。
任何代码中涉及概念名称、类别名称、物理先验范围等常量，
必须且只能从本文件导入，严禁硬编码。
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

# ============================================================
# 1. 概念名称定义（统一snake_case）
# ============================================================

CONCEPT_NAMES_12: List[str] = [
    "period",           # C1:  主周期 (天)
    "amplitude",        # C2:  光变振幅 (mag)
    "rise_fraction",    # C3:  上升时间比 (无量纲, [0,1])
    "R21",              # C4:  Fourier振幅比 A2/A1 (无量纲)
    "R31",              # C5:  Fourier振幅比 A3/A1 (无量纲)
    "phi21",            # C6:  Fourier相位差 phi2-2*phi1 (弧度, [0, 2*pi))
    "skewness",         # C7:  星等分布偏度 (无量纲)
    "kurtosis",         # C8:  星等分布超额峰度 (无量纲, Fisher定义)
    "stetson_K",        # C9:  Stetson K指数 (无量纲)
    "period_snr",       # C10: 周期信噪比 = -log10(FAP), 越大越显著
    "color_bp_rp",      # C11: Gaia BP-RP颜色指数 (mag)
    "mean_mag",         # C12: 平均G波段星等 (mag)
]

CONCEPT_NAMES_20: List[str] = CONCEPT_NAMES_12 + [
    "R41",              # C13: Fourier振幅比 A4/A1
    "R51",              # C14: Fourier振幅比 A5/A1
    "phi31",            # C15: Fourier相位差 phi3-3*phi1 (弧度, [0, 2*pi))
    "phi41",            # C16: Fourier相位差 phi4-4*phi1 (弧度, [0, 2*pi))
    "mag_std",          # C17: 星等标准差 (mag)
    "iqr",              # C18: 星等四分位距 (mag)
    "eta",              # C19: von Neumann eta统计量 (无量纲)
    "percent_beyond_1std",  # C20: 超出1sigma的数据点比例 (无量纲, [0,1])
]

# Aliases for model pipeline compatibility
CONCEPT_NAMES = CONCEPT_NAMES_12  # Compat alias for CONCEPT_NAMES_12

NUM_CONCEPTS: int = 12
NUM_CLASSES: int = 6

# ============================================================
# 2. 变星类别定义
# ============================================================

CLASS_NAMES: List[str] = [
    "RRAB",         # RR Lyrae 基频模式
    "RRC",          # RR Lyrae 一阶泛音
    "DCEP",         # 经典造父变星
    "DSCT_SXPHE",   # Delta Scuti / SX Phoenicis
    "ECL",          # 食双星
    "MIRA_SR",      # Mira / 半规则变星
]

N_CLASSES: int = NUM_CLASSES  # Compat alias for NUM_CLASSES

LABEL_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
LABEL_MAP: Dict[str, int] = LABEL_TO_IDX

# Gaia DR3中的对应标签名
GAIA_LABEL_MAP: Dict[str, str] = {
    "RRAB":       "RR",          # vari_rrlyrae.best_classification = 'RRab'
    "RRC":        "RR",          # vari_rrlyrae.best_classification = 'RRc'
    "DCEP":       "CEP",         # vari_cepheid.type_best_classification = 'DCEP'
    "DSCT_SXPHE": "DSCT|GDOR|SXPHE",  # vari_classifier_result.best_class_name
    "ECL":        "ECL",         # vari_classifier_result.best_class_name
    "MIRA_SR":    "LPV",         # vari_classifier_result.best_class_name
}

# OGLE子类型到本研究6类的映射
OGLE_SUBTYPE_MAP: Dict[str, Optional[str]] = {
    "RRab":       "RRAB",
    "RRc":        "RRC",
    "RRd":        None,
    "RRe":        None,
    "DCEP_F":     "DCEP",
    "DCEP_1O":    "DCEP",
    "DCEP_F/1O":  "DCEP",
    "T2CEP":      None,
    "ACEP":       None,
    "DSCT":       "DSCT_SXPHE",
    "SXPhe":      "DSCT_SXPHE",
    "EC":         "ECL",
    "ESD":        "ECL",
    "ED":         "ECL",
    "Mira":       "MIRA_SR",
    "SRV":        "MIRA_SR",
    "OSARG":      "MIRA_SR",
}

# ============================================================
# 3. 概念分组（物理意义分组，用于消融实验和可视化）
# ============================================================

CONCEPT_GROUPS: Dict[str, List[str]] = {
    "timing": ["period", "rise_fraction", "period_snr"],
    "fourier": ["R21", "R31", "phi21"],
    "amplitude": ["amplitude"],
    "statistics": ["skewness", "kurtosis", "stetson_K"],
    "photometric": ["color_bp_rp", "mean_mag"],
}

MINIMAL_CONCEPTS: List[str] = ["period", "amplitude", "R21", "phi21"]
CONCEPTS_NO_COLOR: List[str] = [c for c in CONCEPT_NAMES_12 if c != "color_bp_rp"]
CONCEPTS_CROSS_SURVEY_10: List[str] = [
    c for c in CONCEPT_NAMES_12 if c not in ("color_bp_rp", "mean_mag")
]

# ============================================================
# 4. 物理先验范围
# ============================================================

PHYSICAL_PRIOR_RANGES: Dict[str, Tuple[float, float]] = {
    "period":         (0.02,    1000.0),
    "amplitude":      (0.005,   10.0),
    "rise_fraction":  (0.0,     1.0),
    "R21":            (0.0,     1.0),
    "R31":            (0.0,     0.6),
    "phi21":          (0.0,     6.2832),
    "skewness":       (-5.0,    5.0),
    "kurtosis":       (-3.0,    20.0),
    "stetson_K":      (0.0,     5.0),
    "period_snr":     (0.0,     300.0),
    "color_bp_rp":    (-0.5,    5.0),
    "mean_mag":       (3.0,     22.0),
}

# ============================================================
# 5. 数据划分参数
# ============================================================

RANDOM_SEED: int = 42
N_CV_FOLDS: int = 5
HOLDOUT_TEST_RATIO: float = 0.15
TEST_IN_DOMAIN_RATIO: float = HOLDOUT_TEST_RATIO  # Alias used by dataset/builder.py

LEARNING_CURVE_SAMPLE_SIZES: List[int] = [
    500, 1000, 2000, 4000, 6000, 8000, 10000, 12000
]

# ============================================================
# 6. Gaia扫描周期混叠相关常量
# ============================================================

GAIA_PRECESSION_PERIOD_DAYS: float = 63.0
GAIA_ALIAS_FREQUENCIES: List[float] = [
    1.0 / GAIA_PRECESSION_PERIOD_DAYS,
    2.0 / GAIA_PRECESSION_PERIOD_DAYS,
    3.0 / GAIA_PRECESSION_PERIOD_DAYS,
    1.0,
    1.0 + 1.0 / GAIA_PRECESSION_PERIOD_DAYS,
    1.0 - 1.0 / GAIA_PRECESSION_PERIOD_DAYS,
]
ALIAS_FREQUENCY_TOLERANCE: float = 0.002

# ============================================================
# 7. 默认超参数
# ============================================================

DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_MAX_EPOCHS: int = 200
DEFAULT_PATIENCE: int = 15

# ============================================================
# 8. OGLE FTP路径映射
# ============================================================

OGLE_FTP_PATHS: Dict[str, Dict[str, str]] = {
    "RRAB": {
        "blg": "blg/rrlyr/",
        "lmc": "lmc/rrlyr/",
        "smc": "smc/rrlyr/",
    },
    "RRC": {
        "blg": "blg/rrlyr/",
        "lmc": "lmc/rrlyr/",
        "smc": "smc/rrlyr/",
    },
    "DCEP": {
        "blg": "blg/cep/",
        "lmc": "lmc/cep/",
        "smc": "smc/cep/",
    },
    "DSCT_SXPHE": {
        "blg": "blg/dsct/",
    },
    "ECL": {
        "blg": "blg/ecl/",
        "lmc": "lmc/ecl/",
        "smc": "smc/ecl/",
    },
    "MIRA_SR": {
        "blg": "blg/lpv/",
        "lmc": "lmc/lpv/",
        "smc": "smc/lpv/",
    },
}
