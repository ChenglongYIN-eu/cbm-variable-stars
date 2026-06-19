# cbm_variable_stars/features/quality.py
"""
Feature quality control module.

Validates extracted features, applies sigma clipping to remove outliers,
and assigns quality flags for downstream filtering.

Quality flag definitions:
    0 = Good (all features within physical bounds, no excessive NaN)
    1 = Acceptable (minor issues, still usable)
    2 = Bad (too many NaN or out-of-physical-range values, exclude from training)
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd

from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES_12,
    PHYSICAL_PRIOR_RANGES,
)


def validate_features(
    features_df: pd.DataFrame,
    sigma_clip: float = 3.0,
    max_nan_fraction: float = 0.3,
    strict_physical_bounds: bool = False,
) -> pd.DataFrame:
    """
    Validate extracted features and add a quality_flag column.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature table with CONCEPT_NAMES_12 columns plus metadata
    sigma_clip : float
        Sigma clipping threshold for outlier detection (default: 3.0).
        Features more than sigma_clip standard deviations from the
        mean are flagged.
    max_nan_fraction : float
        Maximum allowed fraction of NaN values across concept columns
        for a single source (default: 0.3 = 30%).
        Sources exceeding this are flagged as bad (quality_flag=2).
    strict_physical_bounds : bool
        If True, flag as bad (quality_flag=2) any source with a concept
        value outside PHYSICAL_PRIOR_RANGES. Default False (flag=1).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added column:
        - quality_flag: int (0=Good, 1=Acceptable, 2=Bad)

    Quality flag assignment logic
    ------------------------------
    1. Count NaN across concept columns for each source.
       If nan_fraction > max_nan_fraction -> flag=2 (Bad)
    2. Check physical bounds for each concept.
       If out-of-range -> flag=max(current_flag, 1 or 2)
    3. Sigma clipping within each class (if label_name available).
       If any concept is >sigma_clip sigma away from class mean -> flag=max(flag, 1)

    Notes
    -----
    This function modifies the DataFrame in place and returns it.
    Sources with quality_flag=2 are typically excluded from model training
    but may be retained for diagnostic purposes.

    Example
    -------
    >>> features_df = validate_features(features_df, sigma_clip=3.0)
    >>> n_good = (features_df["quality_flag"] == 0).sum()
    >>> print(f"Good sources: {n_good}/{len(features_df)}")
    """
    df = features_df.copy()
    n_total = len(df)

    if n_total == 0:
        logger.warning("[quality] Empty features DataFrame")
        df["quality_flag"] = 2
        return df

    # Available concept columns
    available_concepts = [c for c in CONCEPT_NAMES_12 if c in df.columns]
    if not available_concepts:
        logger.warning(
            f"[quality] No concept columns found in DataFrame. "
            f"Columns: {list(df.columns)}"
        )
        df["quality_flag"] = 1
        return df

    # Initialize quality flags
    df["quality_flag"] = 0

    # ---- Step 1: NaN fraction check ----
    n_nan = df[available_concepts].isna().sum(axis=1)
    nan_fraction = n_nan / len(available_concepts)
    bad_nan_mask = nan_fraction > max_nan_fraction
    df.loc[bad_nan_mask, "quality_flag"] = 2

    n_bad_nan = int(bad_nan_mask.sum())
    if n_bad_nan > 0:
        logger.info(
            f"[quality] {n_bad_nan}/{n_total} sources flagged bad "
            f"(>{max_nan_fraction:.0%} NaN concepts)"
        )

    # ---- Step 2: Physical bounds check ----
    for concept in available_concepts:
        if concept not in PHYSICAL_PRIOR_RANGES:
            continue

        vmin, vmax = PHYSICAL_PRIOR_RANGES[concept]
        col_data = df[concept]
        valid_mask = col_data.notna()

        oob_mask = valid_mask & ((col_data < vmin) | (col_data > vmax))
        n_oob = int(oob_mask.sum())

        if n_oob > 0:
            logger.debug(
                f"[quality] '{concept}': {n_oob} values outside [{vmin}, {vmax}]"
            )
            flag_level = 2 if strict_physical_bounds else 1
            # Only upgrade flag (never downgrade)
            df.loc[oob_mask & (df["quality_flag"] < flag_level), "quality_flag"] = flag_level

    # ---- Step 3: Sigma clipping within class ----
    if "label_name" in df.columns and sigma_clip > 0:
        for class_name in df["label_name"].unique():
            if pd.isna(class_name):
                continue

            class_mask = df["label_name"] == class_name
            class_df = df.loc[class_mask, available_concepts]

            if class_mask.sum() < 5:
                continue

            for concept in available_concepts:
                col = class_df[concept].dropna()
                if len(col) < 5:
                    continue

                mean_val = float(col.mean())
                std_val = float(col.std(ddof=1))

                if std_val <= 0:
                    continue

                # Find outliers
                outlier_mask = (
                    class_mask
                    & df[concept].notna()
                    & (np.abs(df[concept] - mean_val) > sigma_clip * std_val)
                )
                n_outliers = int(outlier_mask.sum())

                if n_outliers > 0:
                    logger.debug(
                        f"[quality] '{concept}' [{class_name}]: "
                        f"{n_outliers} sigma-clipped outliers"
                    )
                    # Only upgrade to 1 (not 2) for sigma clip
                    df.loc[outlier_mask & (df["quality_flag"] == 0), "quality_flag"] = 1

    # ---- Summary ----
    flag_counts = df["quality_flag"].value_counts().sort_index()
    logger.info(
        f"[quality] Feature validation summary:\n"
        f"  Good (0): {flag_counts.get(0, 0)}\n"
        f"  Acceptable (1): {flag_counts.get(1, 0)}\n"
        f"  Bad (2): {flag_counts.get(2, 0)}\n"
        f"  Total: {n_total}"
    )

    return df
