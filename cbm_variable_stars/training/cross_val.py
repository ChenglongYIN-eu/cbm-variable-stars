"""
5-fold cross-validation outer loop for CBM variable star classification.

Per-fold standardization to prevent information leakage (Fix C3).
Each fold fits a StandardScaler on training data and transforms validation data.

Per-fold imputation to prevent imputation data leakage (Fix C4).
When raw_features is provided, imputation statistics are computed on the
training fold only and applied to both training and validation folds.
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from sklearn.preprocessing import StandardScaler

from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
from cbm_variable_stars.data.lightcurve_dataset import LightCurveDataset
from cbm_variable_stars.data.splits import create_cv_splits
from cbm_variable_stars.models import create_model
from cbm_variable_stars.losses.cbm_loss import (
    CBMJointLoss, CBMSequentialLoss, CBMIndependentLoss, compute_class_weights,
)
from cbm_variable_stars.training.trainer import Trainer
from cbm_variable_stars.shared.constants import N_CV_FOLDS, RANDOM_SEED


def _default_per_fold_imputation(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Default per-fold imputation using pipeline_utils per-class medians.

    Computes imputation statistics on the training fold only, then applies
    them to both training and validation folds. Training data uses per-class
    medians; validation data uses global medians (no label leakage).
    Rows that still contain NaN after imputation are removed.

    Returns:
        (imputed_train_features, train_labels, imputed_val_features, val_labels)
        Labels arrays may be shorter than input if NaN rows were removed.
    """
    from cbm_variable_stars.shared.imputation import compute_imputation_stats, apply_imputation

    stats = compute_imputation_stats(train_features, train_labels)
    train_imputed = apply_imputation(train_features, train_labels, stats, use_class_labels=True)
    val_imputed = apply_imputation(val_features, val_labels, stats, use_class_labels=False)

    # Remove rows that still contain NaN after imputation
    train_valid = ~np.any(np.isnan(train_imputed), axis=1)
    val_valid = ~np.any(np.isnan(val_imputed), axis=1)

    return (
        train_imputed[train_valid], train_labels[train_valid],
        val_imputed[val_valid], val_labels[val_valid],
    )


def run_cross_validation(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str = "hard_cbm",
    model_kwargs: Optional[Dict] = None,
    n_folds: int = N_CV_FOLDS,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 15,
    random_seed: int = RANDOM_SEED,
    output_dir: str = "results",
    concept_gt: Optional[np.ndarray] = None,
    device: str = "cpu",
    training_mode: str = "joint",
    save_predictions: bool = True,
    raw_features: Optional[np.ndarray] = None,
    imputation_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Complete 5-fold cross-validation pipeline.

    Per-fold standardization (Fix C3): fits a StandardScaler on each training
    fold and transforms the validation fold, preventing information leakage.

    Per-fold imputation (Fix C4): when ``raw_features`` is provided, imputation
    statistics (per-class medians) are computed on each training fold only and
    applied to both training and validation folds, eliminating the ~20% data
    leakage that occurred when imputation was done on the full CV pool.

    Args:
        features:        Imputed (but NOT pre-standardized) feature matrix, shape (N, 12).
                         Used when raw_features is None (backward-compatible path).
        labels:          Label array, shape (N,)
        model_name:      Model registry key (e.g., "hard_cbm", "hard_cbm_linear")
        model_kwargs:    Additional model constructor kwargs (e.g., hidden_dims)
        n_folds:         Number of CV folds (default 5)
        batch_size:      Training batch size (default 256)
        learning_rate:   Learning rate (default 1e-3)
        weight_decay:    Weight decay (default 1e-4)
        max_epochs:      Maximum training epochs per fold (default 200)
        patience:        Early stopping patience (default 15)
        random_seed:     Random seed for split reproducibility
        output_dir:      Root directory for saving results
        concept_gt:      Optional concept ground truth for Plan B, shape (N, 12)
        device:          Training device ('cpu' or 'cuda')
        training_mode:   Training mode: "joint" (default), "sequential", or "independent"
        save_predictions: Whether to save per-fold predictions in JSON (default True)
        raw_features:    Optional raw feature matrix with NaN values, shape (N, 12).
                         When provided, per-fold imputation is performed inside
                         each fold to prevent data leakage.
        imputation_fn:   Optional callable with signature
                         (train_features, train_labels, val_features, val_labels)
                         -> (imputed_train, train_labels, imputed_val, val_labels).
                         Defaults to _default_per_fold_imputation.

    Returns:
        dict with keys:
            "fold_results":  List of per-fold result dicts
            "aggregated":    Aggregated metrics (mean +/- std across folds)
    """
    if model_kwargs is None:
        model_kwargs = {}

    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    splits = create_cv_splits(labels, n_folds, random_seed)

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'#'*60}")
        print(f"# FOLD {fold_idx + 1}/{n_folds}")
        print(f"# Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"{'#'*60}")

        # === Per-fold imputation (Fix C4: prevent imputation data leakage) ===
        if raw_features is not None:
            impute = imputation_fn if imputation_fn is not None else _default_per_fold_imputation

            n_train_before = len(train_idx)
            n_val_before = len(val_idx)

            (
                fold_train_features, fold_train_labels,
                fold_val_features, fold_val_labels,
            ) = impute(
                raw_features[train_idx], labels[train_idx],
                raw_features[val_idx], labels[val_idx],
            )

            # Synchronize concept_gt with any rows removed during imputation.
            # We detect removed rows by comparing output length to input length;
            # if they differ, we re-derive valid masks using the same imputation
            # logic so that concept_gt indices stay aligned.
            if concept_gt is not None:
                if (len(fold_train_labels) < n_train_before
                        or len(fold_val_labels) < n_val_before):
                    # Re-derive valid masks: apply imputation and check for residual NaNs
                    from cbm_variable_stars.shared.imputation import (
                        compute_imputation_stats as _cis,
                        apply_imputation as _ai,
                    )
                    _stats = _cis(raw_features[train_idx], labels[train_idx])
                    _tr_imp = _ai(raw_features[train_idx], labels[train_idx],
                                  _stats, use_class_labels=True)
                    _vl_imp = _ai(raw_features[val_idx], labels[val_idx],
                                  _stats, use_class_labels=False)
                    _train_keep = ~np.any(np.isnan(_tr_imp), axis=1)
                    _val_keep = ~np.any(np.isnan(_vl_imp), axis=1)
                    fold_train_concept_gt = concept_gt[train_idx][_train_keep]
                    fold_val_concept_gt = concept_gt[val_idx][_val_keep]
                else:
                    fold_train_concept_gt = concept_gt[train_idx]
                    fold_val_concept_gt = concept_gt[val_idx]
            else:
                fold_train_concept_gt = None
                fold_val_concept_gt = None

            print(f"  Per-fold imputation: train {n_train_before}->{len(fold_train_labels)}, "
                  f"val {n_val_before}->{len(fold_val_labels)}")
        else:
            # Backward-compatible path: features are already imputed
            fold_train_features = features[train_idx]
            fold_train_labels = labels[train_idx]
            fold_val_features = features[val_idx]
            fold_val_labels = labels[val_idx]
            fold_train_concept_gt = concept_gt[train_idx] if concept_gt is not None else None
            fold_val_concept_gt = concept_gt[val_idx] if concept_gt is not None else None

        # === Per-fold standardization (Fix C3: prevent val fold information leakage) ===
        fold_scaler = StandardScaler()
        train_features_scaled = fold_scaler.fit_transform(fold_train_features)
        val_features_scaled = fold_scaler.transform(fold_val_features)

        # === Datasets ===
        train_dataset = VariableStarDataset(
            features=train_features_scaled,
            labels=fold_train_labels,
            concept_gt=fold_train_concept_gt,
        )

        val_dataset = VariableStarDataset(
            features=val_features_scaled,
            labels=fold_val_labels,
            concept_gt=fold_val_concept_gt,
        )

        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, device=device,
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, device=device,
        )

        # === Model ===
        model = create_model(model_name, **model_kwargs)

        # === Class weights (computed from training fold only) ===
        _num_classes = model_kwargs.get("num_classes", None)
        class_weights = compute_class_weights(
            torch.tensor(fold_train_labels, dtype=torch.long),
            **({} if _num_classes is None else {"num_classes": _num_classes}),
        )

        # === Loss function (selected by training_mode) ===
        use_concept_loss = model_name in ("hard_cbm_cal",)

        if training_mode == "sequential":
            loss_fn = CBMSequentialLoss(
                class_weights=class_weights,
            )
            # Stage 1: concept learning
            loss_fn.set_stage(1)
            trainer_s1 = Trainer(
                model=model,
                loss_fn=loss_fn,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_epochs=max_epochs // 2,
                patience=patience,
                device=device,
                log_dir=str(output_path / "logs"),
                checkpoint_dir=str(output_path / "checkpoints"),
            )
            trainer_s1.fit(train_loader, val_loader, fold_id=fold_idx)

            # Stage 2: classification learning -- freeze concept calibrators
            loss_fn.set_stage(2)
            for name, param in model.named_parameters():
                if "concept_calibrator" in name:
                    param.requires_grad = False
            trainer_s2 = Trainer(
                model=model,
                loss_fn=loss_fn,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_epochs=max_epochs - max_epochs // 2,
                patience=patience,
                device=device,
                log_dir=str(output_path / "logs"),
                checkpoint_dir=str(output_path / "checkpoints"),
            )
            train_result = trainer_s2.fit(train_loader, val_loader, fold_id=fold_idx)

            # Unfreeze for next fold
            for name, param in model.named_parameters():
                if "concept_calibrator" in name:
                    param.requires_grad = True

            # Use the stage-2 trainer for final validation below
            trainer = trainer_s2

        elif training_mode == "independent":
            loss_fn = CBMIndependentLoss(
                class_weights=class_weights,
            )
            trainer = Trainer(
                model=model,
                loss_fn=loss_fn,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
                log_dir=str(output_path / "logs"),
                checkpoint_dir=str(output_path / "checkpoints"),
            )
            train_result = trainer.fit(train_loader, val_loader, fold_id=fold_idx)

        else:
            # Default: joint training
            loss_fn = CBMJointLoss(
                alpha=1.0 if use_concept_loss else 0.0,
                beta=1.0,
                class_weights=class_weights,
                use_concept_loss=use_concept_loss,
            )
            trainer = Trainer(
                model=model,
                loss_fn=loss_fn,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
                log_dir=str(output_path / "logs"),
                checkpoint_dir=str(output_path / "checkpoints"),
            )
            train_result = trainer.fit(train_loader, val_loader, fold_id=fold_idx)

        # === Final evaluation on validation fold ===
        final_metrics = trainer.validate(val_loader)

        # Collect predictions
        all_preds: List[int] = []
        all_labels_fold: List[int] = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                features_batch = batch["features"].to(device)
                output = model(features_batch)
                all_preds.extend(output["logits"].argmax(1).cpu().tolist())
                all_labels_fold.extend(batch["label"].tolist())

        fold_result: Dict[str, Any] = {
            "fold": fold_idx,
            "best_epoch": train_result["best_epoch"],
            "training_time": train_result["training_time"],
            "metrics": final_metrics,
            "predictions": all_preds,
            "true_labels": all_labels_fold,
            "val_indices": val_idx.tolist(),
        }
        fold_results.append(fold_result)

    # === Aggregate results ===
    accuracies = [r["metrics"]["val_accuracy"] for r in fold_results]
    macro_f1s = [r["metrics"]["val_macro_f1"] for r in fold_results]

    per_class_f1_matrix = [r["metrics"]["val_per_class_f1"] for r in fold_results]

    aggregated: Dict[str, Any] = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
        "per_class_f1_mean": np.mean(per_class_f1_matrix, axis=0).tolist(),
        "per_class_f1_std": np.std(per_class_f1_matrix, axis=0, ddof=1).tolist(),
    }

    # Save results (optionally include per-fold predictions)
    if save_predictions:
        fold_results_to_save = fold_results
    else:
        fold_results_to_save = [
            {k: v for k, v in r.items() if k != "predictions"}
            for r in fold_results
        ]

    results_to_save: Dict[str, Any] = {
        "model_name": model_name,
        "n_folds": n_folds,
        "training_mode": training_mode,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
        },
        "fold_results": fold_results_to_save,
        "aggregated": aggregated,
    }

    with open(output_path / "cv_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary ({model_name}):")
    print(
        f"  Accuracy: {aggregated['accuracy_mean']:.4f} "
        f"+/- {aggregated['accuracy_std']:.4f}"
    )
    print(
        f"  Macro F1: {aggregated['macro_f1_mean']:.4f} "
        f"+/- {aggregated['macro_f1_std']:.4f}"
    )
    print(f"{'='*60}")

    return {
        "fold_results": fold_results,
        "aggregated": aggregated,
    }


def run_e2e_cross_validation(
    raw_features: np.ndarray,
    labels: np.ndarray,
    model_kwargs: Optional[Dict] = None,
    n_folds: int = N_CV_FOLDS,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 15,
    random_seed: int = RANDOM_SEED,
    output_dir: str = "results",
    device: str = "cpu",
    concept_loss_alpha: float = 1.0,
    noise_level: float = 0.03,
    save_predictions: bool = True,
) -> Dict[str, Any]:
    """
    5-fold cross-validation for the end-to-end CBM (EndToEndHardCBM).

    Unlike run_cross_validation, this function:
    1. Synthesizes phase-folded light curves from raw features.
    2. Standardizes concept_gt per fold for concept loss.
    3. Uses CBMJointLoss with concept supervision.

    Args:
        raw_features:       Raw physical features, shape (N, 12). NOT standardized.
        labels:             Integer-encoded labels, shape (N,).
        model_kwargs:       EndToEndHardCBM constructor kwargs.
        n_folds:            Number of CV folds (default 5).
        batch_size:         Training batch size.
        learning_rate:      Learning rate.
        weight_decay:       Weight decay.
        max_epochs:         Max epochs per fold.
        patience:           Early stopping patience.
        random_seed:        Random seed.
        output_dir:         Output directory.
        device:             Training device.
        concept_loss_alpha: Weight for concept loss (default 1.0).
        noise_level:        Noise level for light curve synthesis.
        save_predictions:   Whether to save per-fold predictions.

    Returns:
        dict with keys: "fold_results", "aggregated"
    """
    from sklearn.preprocessing import StandardScaler

    model_name = "e2e_hard_cbm"
    if model_kwargs is None:
        model_kwargs = {}

    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    splits = create_cv_splits(labels, n_folds, random_seed)

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'#'*60}")
        print(f"# E2E FOLD {fold_idx + 1}/{n_folds}")
        print(f"# Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"{'#'*60}")

        fold_train_raw = raw_features[train_idx]
        fold_val_raw = raw_features[val_idx]
        fold_train_labels = labels[train_idx]
        fold_val_labels = labels[val_idx]

        # Per-fold concept standardization
        concept_scaler = StandardScaler()
        train_concept_gt = concept_scaler.fit_transform(fold_train_raw)
        val_concept_gt = concept_scaler.transform(fold_val_raw)

        # Create LightCurve datasets
        train_dataset = LightCurveDataset(
            raw_features=fold_train_raw,
            labels=fold_train_labels,
            concept_gt=train_concept_gt,
            noise_level=noise_level,
            augment=True,
        )
        val_dataset = LightCurveDataset(
            raw_features=fold_val_raw,
            labels=fold_val_labels,
            concept_gt=val_concept_gt,
            noise_level=noise_level,
            augment=False,
        )

        train_loader = create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, device=device,
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, device=device,
        )

        # Model
        model = create_model(model_name, **model_kwargs)

        # Class weights
        class_weights = compute_class_weights(
            torch.tensor(fold_train_labels, dtype=torch.long)
        )

        # Loss with concept supervision
        loss_fn = CBMJointLoss(
            alpha=concept_loss_alpha,
            beta=1.0,
            class_weights=class_weights,
            use_concept_loss=True,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            log_dir=str(output_path / "logs"),
            checkpoint_dir=str(output_path / "checkpoints"),
        )

        train_result = trainer.fit(train_loader, val_loader, fold_id=fold_idx)

        # Final evaluation
        final_metrics = trainer.validate(val_loader)

        # Collect predictions
        all_preds: List[int] = []
        all_labels_fold: List[int] = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                features_batch = batch["features"].to(device)
                output = model(features_batch)
                all_preds.extend(output["logits"].argmax(1).cpu().tolist())
                all_labels_fold.extend(batch["label"].tolist())

        fold_result: Dict[str, Any] = {
            "fold": fold_idx,
            "best_epoch": train_result["best_epoch"],
            "training_time": train_result["training_time"],
            "metrics": final_metrics,
            "predictions": all_preds,
            "true_labels": all_labels_fold,
            "val_indices": val_idx.tolist(),
        }
        fold_results.append(fold_result)

    # Aggregate
    accuracies = [r["metrics"]["val_accuracy"] for r in fold_results]
    macro_f1s = [r["metrics"]["val_macro_f1"] for r in fold_results]
    per_class_f1_matrix = [r["metrics"]["val_per_class_f1"] for r in fold_results]

    aggregated: Dict[str, Any] = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
        "per_class_f1_mean": np.mean(per_class_f1_matrix, axis=0).tolist(),
        "per_class_f1_std": np.std(per_class_f1_matrix, axis=0, ddof=1).tolist(),
    }

    # Save results
    fold_results_to_save = fold_results if save_predictions else [
        {k: v for k, v in r.items() if k != "predictions"} for r in fold_results
    ]
    results_to_save: Dict[str, Any] = {
        "model_name": model_name,
        "n_folds": n_folds,
        "training_mode": "e2e_joint",
        "concept_loss_alpha": concept_loss_alpha,
        "noise_level": noise_level,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
        },
        "fold_results": fold_results_to_save,
        "aggregated": aggregated,
    }

    with open(output_path / "cv_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"E2E Cross-Validation Summary ({model_name}):")
    print(
        f"  Accuracy: {aggregated['accuracy_mean']:.4f} "
        f"+/- {aggregated['accuracy_std']:.4f}"
    )
    print(
        f"  Macro F1: {aggregated['macro_f1_mean']:.4f} "
        f"+/- {aggregated['macro_f1_std']:.4f}"
    )
    print(f"{'='*60}")

    return {
        "fold_results": fold_results,
        "aggregated": aggregated,
    }
