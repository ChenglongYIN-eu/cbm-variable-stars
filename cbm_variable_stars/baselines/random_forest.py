"""
Random Forest baseline for variable star classification.

Hyperparameters aligned with Gaia DR3 official classifier (Rimoldini et al. 2023):
    n_estimators=500, max_features="sqrt", class_weight="balanced"
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional

from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES, CLASS_NAMES, N_CV_FOLDS, RANDOM_SEED,
)


def create_rf_model(
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    max_features: str = "sqrt",
    random_seed: int = RANDOM_SEED,
) -> RandomForestClassifier:
    """
    Create a Random Forest classifier with recommended settings.

    Settings aligned with Gaia DR3 official classifier (Rimoldini et al. 2023).

    Args:
        n_estimators:      Number of trees (default 500)
        max_depth:         Maximum tree depth (None = unlimited)
        min_samples_split: Minimum samples to split internal node (default 10)
        min_samples_leaf:  Minimum samples at leaf node (default 5)
        max_features:      Features per split (default "sqrt")
        random_seed:       Random seed

    Returns:
        Configured RandomForestClassifier instance (not yet fitted).
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
        oob_score=True,
    )


def train_rf(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    random_seed: int = RANDOM_SEED,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Train Random Forest and return (model, metrics).

    Single-split training (no CV). Use train_random_forest() for full 5-fold CV.

    Args:
        features:     Feature matrix, shape (N, 12)
        labels:       Label array, shape (N,)
        n_estimators: Number of trees
        max_depth:    Maximum tree depth
        random_seed:  Random seed
        **kwargs:     Additional RandomForestClassifier kwargs

    Returns:
        dict with keys:
            "model":              Fitted RF model
            "accuracy":           OOB accuracy estimate
            "macro_f1":           OOB macro F1 (approximated)
            "feature_importance": Feature importance array (12,)
            "oob_score":          Out-of-bag score
    """
    rf = create_rf_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_seed=random_seed,
        **{k: v for k, v in kwargs.items()
           if k in ("min_samples_split", "min_samples_leaf", "max_features")},
    )
    rf.fit(features, labels)

    y_oob_pred = rf.oob_decision_function_.argmax(axis=1)
    accuracy = float(accuracy_score(labels, y_oob_pred))
    macro_f1 = float(f1_score(labels, y_oob_pred, average="macro", zero_division=0))

    return {
        "model": rf,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "feature_importance": rf.feature_importances_.tolist(),
        "oob_score": float(rf.oob_score_),
    }


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = N_CV_FOLDS,
    random_seed: int = RANDOM_SEED,
    output_dir: str = "results/rf",
) -> Dict[str, Any]:
    """
    Random Forest baseline -- complete 5-fold cross-validation.

    Hyperparameters aligned with Gaia DR3 official classifier:
        n_estimators=500, max_features="sqrt", class_weight="balanced"

    Args:
        features:    Pre-standardized feature matrix, shape (N, 12)
        labels:      Label array, shape (N,)
        n_folds:     Number of CV folds (default 5)
        random_seed: Random seed for reproducibility
        output_dir:  Directory to save the last fold model

    Returns:
        dict with keys:
            "fold_results":   List of per-fold result dicts
            "aggregated":     Aggregated metrics (mean +/- std across folds)
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_seed,
    )

    fold_results: List[Dict[str, Any]] = []
    feature_importances_all: List[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"RF Fold {fold_idx + 1}/{n_folds}")

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_seed + fold_idx,
            n_jobs=-1,
            oob_score=True,
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
        feat_imp = rf.feature_importances_
        feature_importances_all.append(feat_imp)

        fold_result: Dict[str, Any] = {
            "fold": fold_idx,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "per_class_f1": per_class_f1.tolist(),
            "oob_score": float(rf.oob_score_),
            "feature_importance": feat_imp.tolist(),
            "predictions": y_pred.tolist(),
            "true_labels": y_val.tolist(),
        }
        fold_results.append(fold_result)

        print(
            f"  Acc: {accuracy:.4f} | F1: {macro_f1:.4f} | "
            f"OOB: {rf.oob_score_:.4f}"
        )

        # Save model from last fold
        if fold_idx == n_folds - 1:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(rf, out / "rf_last_fold.joblib")
            print(f"  Saved model to: {out / 'rf_last_fold.joblib'}")

    fi_array = np.array(feature_importances_all)
    aggregated: Dict[str, Any] = {
        "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_results])),
        "accuracy_std": float(np.std([r["accuracy"] for r in fold_results], ddof=1)),
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in fold_results])),
        "macro_f1_std": float(np.std([r["macro_f1"] for r in fold_results], ddof=1)),
        "feature_importance_mean": fi_array.mean(axis=0).tolist(),
        "feature_importance_std": fi_array.std(axis=0, ddof=1).tolist(),
        "feature_names": CONCEPT_NAMES,
    }

    print(f"\nRF CV Summary:")
    print(f"  Accuracy: {aggregated['accuracy_mean']:.4f} +/- {aggregated['accuracy_std']:.4f}")
    print(f"  Macro F1: {aggregated['macro_f1_mean']:.4f} +/- {aggregated['macro_f1_std']:.4f}")

    return {
        "fold_results": fold_results,
        "aggregated": aggregated,
    }
