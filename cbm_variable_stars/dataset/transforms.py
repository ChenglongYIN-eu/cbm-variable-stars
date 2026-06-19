# cbm_variable_stars/dataset/transforms.py
"""
Feature standardization module.

Project-wide convention (correction S3):
    Only StandardScaler is supported. RobustScaler is explicitly rejected
    to maintain consistency across the training, evaluation, and baseline pipelines.

Scaler fitting strategy (correction S2):
    Global scaler: fit on cv_pool (85% of Gaia data)
    Applied to: test_in_domain, OGLE cross-survey test set
    Fold-level scaler: fit per fold's train subset (training loop's responsibility)
"""

from __future__ import annotations
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12


def fit_scaler(
    train_features: pd.DataFrame,
    method: str = "standard",
    save_path: Optional[str | Path] = None,
) -> StandardScaler:
    """
    Fit a StandardScaler on training features.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature table, must contain CONCEPT_NAMES_12 columns
    method : str
        Only "standard" is supported (project-wide convention S3).
        Raises ValueError for any other value.
    save_path : str or Path, optional
        If provided, save the fitted scaler and medians to this .pkl file.

    Returns
    -------
    StandardScaler
        Fitted scikit-learn StandardScaler

    Raises
    ------
    ValueError
        If method != "standard"

    Saved pickle format
    -------------------
    {
        "scaler": StandardScaler instance,
        "medians": {concept_name: median_value, ...},
        "concept_names": CONCEPT_NAMES_12,
    }
    The medians are used for NaN imputation before scaling.

    Example
    -------
    >>> scaler = fit_scaler(train_df, save_path="data/processed/scaler.pkl")
    >>> print(f"period: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    """
    if method != "standard":
        raise ValueError(
            f"Project uses StandardScaler only. Got method='{method}'. "
            f"See shared convention S3. "
            f"RobustScaler and other scalers are not supported."
        )

    scaler = StandardScaler()

    # Get available concept columns
    available = [c for c in CONCEPT_NAMES_12 if c in train_features.columns]
    if not available:
        raise ValueError(
            f"No concept columns found in train_features. "
            f"Expected columns from CONCEPT_NAMES_12: {CONCEPT_NAMES_12}"
        )

    if len(available) < len(CONCEPT_NAMES_12):
        missing = [c for c in CONCEPT_NAMES_12 if c not in available]
        logger.warning(
            f"[scaler] Missing concept columns: {missing}. "
            f"Fitting on {len(available)} available concepts."
        )

    X = train_features[available].copy()

    # Compute medians for NaN imputation
    medians = X.median()

    # Fill NaN with medians before fitting
    X_filled = X.fillna(medians)

    scaler.fit(X_filled)

    logger.info(f"[scaler] StandardScaler fitted on {len(X_filled)} samples, {len(available)} concepts:")
    for i, feat in enumerate(available):
        logger.info(
            f"  {feat}: mean={scaler.mean_[i]:.4f}, std={scaler.scale_[i]:.4f}, "
            f"median={float(medians[feat]):.4f}"
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "scaler": scaler,
                    "medians": medians.to_dict(),
                    "concept_names": available,
                },
                f,
            )
        logger.info(f"[scaler] Scaler saved: {save_path}")

    return scaler


def transform_features(
    features_df: pd.DataFrame,
    scaler: StandardScaler,
    medians: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Apply a fitted StandardScaler to feature columns.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature table to transform
    scaler : StandardScaler
        Fitted StandardScaler (from fit_scaler())
    medians : dict, optional
        Training set medians for NaN imputation {concept_name: value}.
        If None, NaN values are filled with 0 after scaling.

    Returns
    -------
    pd.DataFrame
        Copy of input with concept columns standardized.
        Non-concept columns (metadata) are preserved unchanged.

    Notes
    -----
    - NaN values in concept columns are filled with training medians
      BEFORE scaling (so they become ~0 after scaling).
    - If a concept column is missing from features_df, it is not modified.
    - The transform is applied only to columns present in features_df
      that match CONCEPT_NAMES_12.

    Example
    -------
    >>> test_scaled = transform_features(test_df, scaler, medians)
    >>> train_scaled = transform_features(train_df, scaler, medians)
    """
    df = features_df.copy()

    # Get concept columns that are in this DataFrame
    available_concepts = [c for c in CONCEPT_NAMES_12 if c in df.columns]

    if not available_concepts:
        logger.warning(
            "[transform] No concept columns to transform. "
            f"DataFrame columns: {list(df.columns)}"
        )
        return df

    # Fill NaN with medians
    if medians is not None:
        for feat in available_concepts:
            if feat in medians and pd.api.types.is_numeric_dtype(df[feat]):
                df[feat] = df[feat].fillna(medians[feat])

    # Apply scaling
    # The scaler was fit on CONCEPT_NAMES_12 (or subset); we need to match columns
    try:
        # Get the columns the scaler was fitted on
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
        else:
            # Assume same order as available_concepts if no feature names stored
            scaler_cols = available_concepts[:len(scaler.mean_)]

        # Find intersection
        transform_cols = [c for c in scaler_cols if c in df.columns]

        if not transform_cols:
            logger.warning(
                "[transform] No overlap between scaler columns and DataFrame columns"
            )
            return df

        # Build array for transform
        X = df[transform_cols].values.astype(float)

        # Replace any remaining NaN with 0 (should be rare after median fill)
        n_remaining_nan = np.isnan(X).sum()
        if n_remaining_nan > 0:
            logger.warning(
                f"[transform] {n_remaining_nan} NaN values remain after median "
                f"imputation, filling with 0.0"
            )
        X = np.nan_to_num(X, nan=0.0)

        # Get corresponding scaler parameters
        if len(transform_cols) == len(scaler_cols):
            X_scaled = scaler.transform(X)
        else:
            # Partial columns: manually apply scaling
            scaler_col_idx = {col: i for i, col in enumerate(scaler_cols)}
            X_scaled = np.zeros_like(X)
            for j, col in enumerate(transform_cols):
                if col in scaler_col_idx:
                    idx = scaler_col_idx[col]
                    mean = scaler.mean_[idx]
                    scale = scaler.scale_[idx]
                    if scale > 0:
                        X_scaled[:, j] = (X[:, j] - mean) / scale
                    else:
                        X_scaled[:, j] = X[:, j] - mean

        df[transform_cols] = X_scaled

    except Exception as e:
        logger.error(f"[transform] Scaling failed: {e}")
        # Return unscaled data rather than crashing
        return features_df.copy()

    return df


def load_scaler(
    scaler_path: str | Path,
) -> tuple[StandardScaler, dict[str, float]]:
    """
    Load a saved scaler from disk.

    Parameters
    ----------
    scaler_path : str or Path
        Path to .pkl file saved by fit_scaler()

    Returns
    -------
    tuple[StandardScaler, dict[str, float]]
        (scaler, medians)

    Example
    -------
    >>> scaler, medians = load_scaler("data/processed/scaler.pkl")
    >>> test_scaled = transform_features(test_df, scaler, medians)
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(scaler_path, "rb") as f:
        data = pickle.load(f)

    scaler = data["scaler"]
    medians = data.get("medians", {})

    logger.info(f"[scaler] Loaded scaler from {scaler_path}")
    return scaler, medians
