"""
Data split utilities for variable star classification.

[Fix S2] Unified split scheme:
    Full Gaia data (~19,500)
        -> In-domain test set (15%, ~2,925, hold-out, never seen during training)
        -> CV subset (85%, ~16,575) -> 5-fold CV

OGLE data (~10,000) always used as out-of-domain test set, not participating in any split.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import List, Tuple, Dict

from cbm_variable_stars.shared.constants import (
    HOLDOUT_TEST_RATIO, N_CV_FOLDS, RANDOM_SEED,
)


def create_full_split(
    labels: np.ndarray,
    test_ratio: float = HOLDOUT_TEST_RATIO,  # 0.15 [Fix S2]
    random_seed: int = RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    """
    Create train+CV subset / in-domain test set split.

    [Fix S2] Unified split scheme:
        Full Gaia data (~19,500) -> In-domain test set (15%, ~2,925, hold-out)
                                 -> CV subset (85%, ~16,575) -> 5-fold CV

    OGLE data (~10,000) always used as out-of-domain test set, not participating
    in any split.

    Args:
        labels:      All labels, shape (N,)
        test_ratio:  Hold-out test set ratio (default 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        dict with keys:
            "cv_indices":   Indices for CV subset (~16,575 -> enters 5-fold CV)
            "test_indices": Indices for in-domain hold-out test set (~2,925)
    """
    indices = np.arange(len(labels))
    cv_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_seed,
    )

    return {
        "cv_indices": cv_idx,      # ~16,575 -> enters 5-fold CV
        "test_indices": test_idx,  # ~2,925  -> in-domain hold-out test set
    }


def create_cv_splits(
    labels: np.ndarray,
    n_folds: int = N_CV_FOLDS,
    random_seed: int = RANDOM_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified k-fold cross-validation splits.

    Stratification ensures each fold has the same class proportion as
    the full dataset.

    Args:
        labels:      Label array, shape (N,)
        n_folds:     Number of folds (default 5)
        random_seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_seed,
    )

    splits = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        splits.append((train_idx, val_idx))

    return splits
