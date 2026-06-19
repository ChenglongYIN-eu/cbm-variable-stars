# cbm_variable_stars/dataset/builder.py
"""
Dataset construction module.

Implements the corrected S2 data split scheme:
    Gaia data:
        15% hold-out  -> test_in_domain (fixed, never used in training)
        85%           -> cv_pool
                         -> 5-fold StratifiedKFold CV (each fold: 80% train + 20% val)
    OGLE:           -> test_cross_survey (always out-of-domain test set)

Scaler fitting:
    Global scaler: fit on entire cv_pool, applied to test_in_domain and OGLE
    Fold-level scaler: fit on each fold's train subset (handled by training loop)
"""

from __future__ import annotations
from pathlib import Path
import json
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from omegaconf import DictConfig

from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES_12,
    RANDOM_SEED,
    TEST_IN_DOMAIN_RATIO,
    N_CV_FOLDS,
    LABEL_TO_IDX,
)
from cbm_variable_stars.dataset.transforms import fit_scaler, transform_features


def build_datasets(
    gaia_features: pd.DataFrame,
    ogle_features: Optional[pd.DataFrame],
    cfg: DictConfig,
    output_dir: str | Path = "data/processed/",
) -> dict[str, pd.DataFrame]:
    """
    Build complete train/test datasets with the corrected S2 split scheme.

    Parameters
    ----------
    gaia_features : pd.DataFrame
        Gaia feature table (quality-controlled), with CONCEPT_NAMES_12 +
        label + label_name columns
    ogle_features : pd.DataFrame or None
        OGLE feature table (optional; used as cross-survey test set)
    cfg : DictConfig
        Configuration
    output_dir : str or Path
        Output directory for saved datasets

    Returns
    -------
    dict[str, pd.DataFrame]
        {
            "cv_pool":          85% Gaia data for 5-fold CV,
            "test_in_domain":   15% Gaia hold-out test set,
            "test_cross_survey": OGLE cross-survey test set (if provided),
        }

    Data split scheme (corrected S2)
    ----------------------------------
    Old scheme: 70/15/15 three-way split + 5-fold CV (conflicting)
    New scheme:
        Gaia data:
          |-- 15% hold-out --> test_in_domain (fixed, not in any training)
          |-- 85% ----------> cv_pool
                               |-> 5-fold StratifiedKFold
                                   (each fold: 80% train + 20% val)
        OGLE: always out-of-domain test set

    Scaler:
        Global scaler: fit on cv_pool, applied to test_in_domain and OGLE
        Fold-level scaler: fit per fold (handled by training loop)

    Example
    -------
    >>> datasets = build_datasets(gaia_features, ogle_features, cfg)
    >>> cv_pool = datasets["cv_pool"]
    >>> test_id = datasets["test_in_domain"]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get random seed from config or use default
    seed = getattr(getattr(cfg, "project", {}), "random_seed", RANDOM_SEED)
    if hasattr(cfg, "project") and hasattr(cfg.project, "random_seed"):
        seed = cfg.project.random_seed

    # ---- Step 1: Quality filtering ----
    if "quality_flag" in gaia_features.columns:
        gaia_clean = gaia_features[gaia_features["quality_flag"] <= 1].copy()
        n_removed = len(gaia_features) - len(gaia_clean)
        logger.info(
            f"[builder] Quality filter: removed {n_removed} bad sources, "
            f"kept {len(gaia_clean)}"
        )
    else:
        gaia_clean = gaia_features.copy()
        logger.debug("[builder] No quality_flag column; using all sources")

    if len(gaia_clean) == 0:
        raise ValueError("No valid Gaia sources after quality filtering")

    # Ensure label column exists
    if "label" not in gaia_clean.columns:
        if "label_name" in gaia_clean.columns:
            gaia_clean["label"] = gaia_clean["label_name"].map(LABEL_TO_IDX).fillna(-1).astype(int)
        else:
            raise ValueError("gaia_features must contain 'label' or 'label_name' column")

    # ---- Step 2: Stratified 15% hold-out + 85% CV pool ----
    y = gaia_clean["label"].values

    cv_pool_df, test_df = train_test_split(
        gaia_clean,
        test_size=TEST_IN_DOMAIN_RATIO,   # 0.15
        random_state=seed,
        stratify=y,
    )

    cv_pool_df = cv_pool_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info("=" * 50)
    logger.info("[builder] Data split (corrected S2):")
    logger.info(
        f"  CV pool:       {len(cv_pool_df)} ({100.0*len(cv_pool_df)/len(gaia_clean):.1f}%)"
    )
    logger.info(
        f"  Hold-out test: {len(test_df)} ({100.0*len(test_df)/len(gaia_clean):.1f}%)"
    )

    for split_name, split_df in [("CV pool", cv_pool_df), ("Hold-out test", test_df)]:
        if "label_name" in split_df.columns:
            dist = split_df["label_name"].value_counts().to_dict()
            logger.info(f"  {split_name} class distribution: {dist}")

    # ---- Step 3: Fit global scaler on cv_pool ----
    scaler_path = output_dir / "scaler.pkl"
    scaler = fit_scaler(
        train_features=cv_pool_df,
        method="standard",
        save_path=scaler_path,
    )

    # Load medians for NaN imputation
    with open(scaler_path, "rb") as f:
        scaler_data = pickle.load(f)
    medians = scaler_data["medians"]

    # Transform both splits
    cv_pool_scaled = transform_features(cv_pool_df, scaler, medians)
    test_scaled = transform_features(test_df, scaler, medians)

    # ---- Step 4: Generate 5-fold CV indices ----
    skf = StratifiedKFold(
        n_splits=N_CV_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    available_concepts = [c for c in CONCEPT_NAMES_12 if c in cv_pool_scaled.columns]
    cv_labels = cv_pool_scaled["label"].values

    cv_folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(cv_pool_scaled[available_concepts], cv_labels)
    ):
        cv_folds.append((train_idx, val_idx))
        logger.info(
            f"  CV fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}"
        )

    with open(output_dir / "cv_folds.pkl", "wb") as f:
        pickle.dump(cv_folds, f)
    logger.info(f"[builder] Saved {N_CV_FOLDS}-fold CV indices: {output_dir / 'cv_folds.pkl'}")

    # ---- Step 5: OGLE cross-survey test set ----
    datasets = {
        "cv_pool": cv_pool_scaled,
        "test_in_domain": test_scaled,
    }

    if ogle_features is not None and len(ogle_features) > 0:
        ogle_clean = ogle_features.copy()
        if "quality_flag" in ogle_clean.columns:
            ogle_clean = ogle_clean[ogle_clean["quality_flag"] <= 1].copy()
            logger.info(
                f"[builder] OGLE quality filter: kept {len(ogle_clean)}/{len(ogle_features)}"
            )

        if len(ogle_clean) > 0:
            ogle_scaled = transform_features(ogle_clean, scaler, medians)
            datasets["test_cross_survey"] = ogle_scaled
            logger.info(f"[builder] Cross-survey test (OGLE): {len(ogle_scaled)} sources")
        else:
            logger.warning("[builder] No OGLE sources passed quality filter")

    # ---- Step 6: Save datasets ----
    for name, df_data in datasets.items():
        path = output_dir / f"{name}.parquet"
        df_data.to_parquet(path, index=False)
        logger.info(f"[builder] Saved: {path} ({len(df_data)} rows)")

    # Save label mapping
    label_map_path = output_dir / "label_mapping.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_TO_IDX, f, indent=2)
    logger.info(f"[builder] Saved label mapping: {label_map_path}")

    logger.info("=" * 50)
    logger.info("[builder] Dataset build complete")

    return datasets


def get_cv_fold(
    cv_pool_df: pd.DataFrame,
    fold_idx: int,
    cv_folds_path: str | Path = "data/processed/cv_folds.pkl",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve a specific cross-validation fold.

    Parameters
    ----------
    cv_pool_df : pd.DataFrame
        Full CV pool data (already standardized)
    fold_idx : int
        Fold index (0 to N_CV_FOLDS-1, i.e., 0-4 for 5-fold CV)
    cv_folds_path : str or Path
        Path to saved CV indices pickle file

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (fold_train, fold_val)
        Both are subsets of cv_pool_df.

    Example
    -------
    >>> cv_pool = pd.read_parquet("data/processed/cv_pool.parquet")
    >>> for k in range(5):
    ...     fold_train, fold_val = get_cv_fold(cv_pool, k)
    ...     model.fit(fold_train)
    ...     metrics = evaluate(model, fold_val)
    """
    cv_folds_path = Path(cv_folds_path)
    if not cv_folds_path.exists():
        raise FileNotFoundError(f"CV folds file not found: {cv_folds_path}")

    with open(cv_folds_path, "rb") as f:
        cv_folds = pickle.load(f)

    n_folds = len(cv_folds)
    if fold_idx < 0 or fold_idx >= n_folds:
        raise IndexError(
            f"fold_idx={fold_idx} out of range [0, {n_folds-1}]"
        )

    train_idx, val_idx = cv_folds[fold_idx]
    fold_train = cv_pool_df.iloc[train_idx].copy().reset_index(drop=True)
    fold_val = cv_pool_df.iloc[val_idx].copy().reset_index(drop=True)

    logger.debug(
        f"[builder] CV fold {fold_idx}: "
        f"train={len(fold_train)}, val={len(fold_val)}"
    )

    return fold_train, fold_val
