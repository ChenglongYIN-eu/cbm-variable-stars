"""
Per-class median imputation utilities.

[M9 FIX] Extracted from scripts/pipeline_utils.py into the package so that
cross_val.py (and any other in-package code) can import without depending
on the scripts/ directory, which breaks when the package is pip-installed
or run from a different working directory.
"""

from __future__ import annotations

import numpy as np

from cbm_variable_stars.shared.logger import logger


def compute_imputation_stats(features: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute per-class median imputation statistics on training data only.

    MUST be called ONLY on training/CV data to prevent data leakage.

    Returns:
        dict with 'per_class_medians' and 'global_medians' per feature column.
    """
    assert features.ndim == 2
    stats: dict = {"per_class_medians": {}, "global_medians": {}}
    for j in range(features.shape[1]):
        col = features[:, j]
        class_meds: dict = {}
        for cls in np.unique(labels):
            vals = col[(labels == cls) & ~np.isnan(col)]
            if len(vals) > 0:
                class_meds[int(cls)] = float(np.median(vals))
        stats["per_class_medians"][j] = class_meds
        valid = col[~np.isnan(col)]
        stats["global_medians"][j] = float(np.median(valid)) if len(valid) > 0 else 0.0
    return stats


def apply_imputation(
    features: np.ndarray,
    labels: np.ndarray,
    stats: dict,
    use_class_labels: bool = True,
) -> np.ndarray:
    """
    Apply pre-computed imputation statistics to fill NaN values.

    Args:
        features:         Feature matrix with potential NaN values.
        labels:           Label array (used for per-class routing if use_class_labels=True).
        stats:            Imputation statistics from compute_imputation_stats().
        use_class_labels: If True, use per-class medians (appropriate for training data
                          where labels are known). If False, use only global medians
                          (appropriate for test data to prevent label leakage).
    """
    features = features.copy()
    for j in range(features.shape[1]):
        if np.any(np.isnan(features[:, j])):
            if use_class_labels:
                for cls in np.unique(labels):
                    mask = (labels == cls) & np.isnan(features[:, j])
                    if np.any(mask):
                        med = stats["per_class_medians"].get(j, {}).get(
                            int(cls), stats["global_medians"].get(j, 0.0)
                        )
                        features[mask, j] = med
            remaining = np.isnan(features[:, j])
            if np.any(remaining):
                features[remaining, j] = stats["global_medians"].get(j, 0.0)
    return features
