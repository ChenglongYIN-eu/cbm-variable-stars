"""
XGBoost baseline for variable star classification.

Uses sample_weight to simulate class_weight="balanced".
Computes SHAP values on the last fold for comparison with CBM concept importance.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, List, Optional

from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES, CLASS_NAMES, N_CV_FOLDS, RANDOM_SEED,
)


def create_xgb_model(
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_classes: int = 6,
    random_seed: int = RANDOM_SEED,
    **kwargs: Any,
):
    """
    Create an XGBoost multiclass classifier with recommended settings.

    Args:
        n_estimators:  Number of boosting rounds (default 300)
        max_depth:     Maximum tree depth (default 6)
        learning_rate: Step size shrinkage (default 0.1)
        n_classes:     Number of output classes (default 6)
        random_seed:   Random seed
        **kwargs:      Additional XGBClassifier kwargs

    Returns:
        Configured XGBClassifier instance (not yet fitted).
    """
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=kwargs.get("subsample", 0.8),
        colsample_bytree=kwargs.get("colsample_bytree", 0.8),
        reg_alpha=kwargs.get("reg_alpha", 0.1),
        reg_lambda=kwargs.get("reg_lambda", 1.0),
        min_child_weight=kwargs.get("min_child_weight", 5),
        gamma=kwargs.get("gamma", 0.1),
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=random_seed,
        n_jobs=-1,
        verbosity=0,
    )


def train_xgb(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_seed: int = RANDOM_SEED,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Train XGBoost and return (model, metrics). Single-split training.

    Use train_xgboost() for full 5-fold CV.

    Args:
        features:      Feature matrix, shape (N, 12)
        labels:        Label array, shape (N,)
        n_estimators:  Number of boosting rounds
        max_depth:     Maximum tree depth
        learning_rate: Step size shrinkage
        random_seed:   Random seed
        **kwargs:      Additional XGBClassifier kwargs

    Returns:
        dict with keys:
            "model":              Fitted XGBClassifier
            "feature_importance": Feature importance array (12,)
    """
    n_classes = len(np.unique(labels))
    model = create_xgb_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_classes=n_classes,
        random_seed=random_seed,
        **kwargs,
    )

    # Compute sample weights for class balance
    unique_classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    class_weight_map = {
        c: n_total / (n_classes * cnt)
        for c, cnt in zip(unique_classes, counts)
    }
    sample_weights = np.array([class_weight_map[l] for l in labels])

    model.fit(features, labels, sample_weight=sample_weights)

    return {
        "model": model,
        "feature_importance": model.feature_importances_.tolist(),
    }


def train_xgboost(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = N_CV_FOLDS,
    random_seed: int = RANDOM_SEED,
    output_dir: str = "results/xgb",
    compute_shap: bool = True,
) -> Dict[str, Any]:
    """
    XGBoost baseline -- complete 5-fold cross-validation.

    Uses sample_weight to simulate class_weight="balanced".
    Computes SHAP values on last fold for comparison with CBM concept importance.

    Args:
        features:     Pre-standardized feature matrix, shape (N, 12)
        labels:       Label array, shape (N,)
        n_folds:      Number of CV folds (default 5)
        random_seed:  Random seed for reproducibility
        output_dir:   Directory to save SHAP values
        compute_shap: Whether to compute SHAP values on last fold

    Returns:
        dict with keys:
            "fold_results": List of per-fold result dicts
            "aggregated":   Aggregated metrics (mean +/- std across folds)
            "shap_values":  SHAP values from last fold (if compute_shap=True)
    """
    import xgboost as xgb

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_seed,
    )

    fold_results: List[Dict[str, Any]] = []
    shap_values_last: Any = None

    # Pre-compute sample weights for class balance
    unique_classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    n_classes = len(unique_classes)
    class_weight_map = {
        c: n_total / (n_classes * cnt)
        for c, cnt in zip(unique_classes, counts)
    }
    sample_weights = np.array([class_weight_map[l] for l in labels])

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"XGBoost Fold {fold_idx + 1}/{n_folds}")

        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        w_train = sample_weights[train_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            gamma=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=random_seed + fold_idx,
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        per_class_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
        feat_imp = model.feature_importances_

        # Compute SHAP values on the last fold for interpretability comparison
        if compute_shap and fold_idx == n_folds - 1:
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values_last = explainer.shap_values(X_val)
            except ImportError:
                print("  Warning: shap not installed; skipping SHAP computation.")

        best_iter = getattr(model, "best_iteration", 300)

        fold_result: Dict[str, Any] = {
            "fold": fold_idx,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "per_class_f1": per_class_f1.tolist(),
            "feature_importance": feat_imp.tolist(),
            "best_iteration": best_iter,
            "predictions": y_pred.tolist(),
            "true_labels": y_val.tolist(),
        }
        fold_results.append(fold_result)

        print(f"  Acc: {accuracy:.4f} | F1: {macro_f1:.4f}")

    # Save SHAP values
    if shap_values_last is not None:
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "shap_values.npy", shap_values_last)
        print(f"  SHAP values saved to: {out / 'shap_values.npy'}")

    aggregated: Dict[str, Any] = {
        "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_results])),
        "accuracy_std": float(np.std([r["accuracy"] for r in fold_results], ddof=1)),
        "macro_f1_mean": float(np.mean([r["macro_f1"] for r in fold_results])),
        "macro_f1_std": float(np.std([r["macro_f1"] for r in fold_results], ddof=1)),
        "feature_names": CONCEPT_NAMES,
    }

    print(f"\nXGBoost CV Summary:")
    print(f"  Accuracy: {aggregated['accuracy_mean']:.4f} +/- {aggregated['accuracy_std']:.4f}")
    print(f"  Macro F1: {aggregated['macro_f1_mean']:.4f} +/- {aggregated['macro_f1_std']:.4f}")

    return {
        "fold_results": fold_results,
        "aggregated": aggregated,
        "shap_values": shap_values_last,
    }
