"""
Classification metrics for variable star classification evaluation.

Provides comprehensive metrics: accuracy, macro/weighted F1, per-class
metrics, confusion matrix, MCC, Cohen's Kappa, AUC-ROC, and specificity.
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from typing import Dict, List, Any, Union, Optional

from cbm_variable_stars.shared.constants import CLASS_NAMES

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Metric selection:
        1. Accuracy:        Overall accuracy
        2. Macro F1:        Primary evaluation metric (equal weight per class)
        3. Weighted F1:     Reflects actual deployment performance
        4. Per-class F1/Precision/Recall: Identifies underperforming classes
        5. Confusion matrix: Shows specific misclassification patterns
        6. MCC:             Matthews Correlation Coefficient (imbalance-robust)
        7. Cohen's Kappa:   Inter-rater agreement beyond chance
        8. AUC-ROC:         One-vs-Rest macro/weighted (requires probabilities)
        9. Per-class Specificity/Sensitivity

    Args:
        y_true:        Ground truth labels
        y_pred:        Predicted labels
        class_names:   List of class names for per-class metrics
        y_pred_proba:  Predicted class probabilities, shape (N, n_classes).
                       Optional; when None, AUC-ROC fields are returned as None.

    Returns:
        dict with keys:
            "accuracy":                Overall accuracy
            "macro_f1":                Macro-averaged F1
            "weighted_f1":             Weighted-averaged F1
            "macro_precision":         Macro-averaged precision
            "macro_recall":            Macro-averaged recall
            "per_class":               {class_name: {f1, precision, recall, support}}
            "confusion_matrix":        Raw confusion matrix as nested list
            "classification_report":   sklearn classification report dict
            "mcc":                     Matthews Correlation Coefficient
            "cohen_kappa":             Cohen's Kappa score
            "auc_roc_macro":           Macro AUC-ROC (None if no probabilities)
            "auc_roc_weighted":        Weighted AUC-ROC (None if no probabilities)
            "per_class_specificity":   {class_name: specificity}
            "per_class_sensitivity":   {class_name: sensitivity (= recall)}
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    all_labels = list(range(len(class_names)))
    per_class_f1 = f1_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # --- Matthews Correlation Coefficient ---
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    # --- Cohen's Kappa ---
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = 0.0

    # --- AUC-ROC (One-vs-Rest) ---
    auc_roc_macro: Optional[float] = None
    auc_roc_weighted: Optional[float] = None

    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba)
        try:
            auc_roc_macro = float(
                roc_auc_score(
                    y_true,
                    y_pred_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=all_labels,
                )
            )
        except (ValueError, IndexError) as e:
            logger.warning("Could not compute macro AUC-ROC: %s", e)
            auc_roc_macro = None

        try:
            auc_roc_weighted = float(
                roc_auc_score(
                    y_true,
                    y_pred_proba,
                    multi_class="ovr",
                    average="weighted",
                    labels=all_labels,
                )
            )
        except (ValueError, IndexError) as e:
            logger.warning("Could not compute weighted AUC-ROC: %s", e)
            auc_roc_weighted = None

    # --- Per-class Specificity & Sensitivity from confusion matrix ---
    n_classes = len(class_names)
    per_class_specificity: Dict[str, float] = {}
    per_class_sensitivity: Dict[str, float] = {}

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        per_class_specificity[name] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        per_class_sensitivity[name] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "per_class": {
            name: {
                "f1": float(per_class_f1[i]),
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "support": int(np.sum(y_true == i)),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "auc_roc_macro": auc_roc_macro,
        "auc_roc_weighted": auc_roc_weighted,
        "per_class_specificity": per_class_specificity,
        "per_class_sensitivity": per_class_sensitivity,
    }


def compute_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    normalize: str = "true",
) -> Dict[str, Any]:
    """
    Compute a (optionally normalized) confusion matrix.

    Args:
        y_true:      Ground truth labels
        y_pred:      Predicted labels
        class_names: List of class names
        normalize:   Normalization strategy passed to sklearn.
                     'true' = normalize by true class counts (row-wise).
                     'pred' = normalize by predicted class counts (col-wise).
                     'all'  = normalize by total samples.
                     None   = raw counts.

    Returns:
        dict with keys:
            "matrix":      Confusion matrix (nested list)
            "normalized":  Whether matrix is normalized
            "class_names": Class name list
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    all_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize=normalize)

    return {
        "matrix": cm.tolist(),
        "normalized": normalize is not None,
        "normalize_mode": normalize,
        "class_names": class_names,
    }
