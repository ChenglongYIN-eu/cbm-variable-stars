"""
Multi-model parallel cross-validation using CUDA Streams.

When device='cuda', trains all CBM models simultaneously on each CV fold
using ParallelTrainer. When device='cpu', falls back to serial
run_cross_validation calls.

Output format is identical to run_cross_validation:
    Dict[model_name, {"fold_results": [...], "aggregated": {...}}]
"""

import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data import DataLoader
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
from cbm_variable_stars.data.splits import create_cv_splits
from cbm_variable_stars.models import create_model
from cbm_variable_stars.losses.cbm_loss import CBMJointLoss, compute_class_weights
from cbm_variable_stars.shared.constants import N_CV_FOLDS, RANDOM_SEED, N_CLASSES
from cbm_variable_stars.training.parallel_trainer import (
    ParallelTrainer, ModelSlot, create_model_slot,
)


def run_parallel_cross_validation(
    features: np.ndarray,
    labels: np.ndarray,
    model_names: List[str],
    model_kwargs_list: Optional[List[Dict]] = None,
    n_folds: int = N_CV_FOLDS,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 15,
    random_seed: int = RANDOM_SEED,
    output_dir: str = "results",
    concept_gt: Optional[np.ndarray] = None,
    device: str = "cuda",
    save_predictions: bool = True,
    raw_features: Optional[np.ndarray] = None,
    imputation_fn: Optional[Callable] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Multi-model parallel cross-validation.

    When device starts with 'cuda', all models in model_names are trained
    simultaneously on each fold using CUDA Streams (ParallelTrainer).
    When device='cpu', falls back to serial run_cross_validation.

    Per-fold imputation (Fix C4): when ``raw_features`` is provided, imputation
    statistics are computed on each training fold only and applied to both
    training and validation folds, eliminating imputation data leakage.

    Only supports training_mode="joint" (the default and most common mode).
    Models requiring sequential training (e.g., hard_cbm_cal with
    training_mode="sequential") should be trained separately.

    Args:
        features:          Imputed (NOT pre-standardized) feature matrix (N, 12).
                           Used when raw_features is None (backward-compatible path).
        labels:            Label array (N,)
        model_names:       List of model registry keys to train in parallel
        model_kwargs_list: Optional per-model kwargs (length must match model_names)
        n_folds:           Number of CV folds
        batch_size:        Training batch size
        learning_rate:     Learning rate for all models
        weight_decay:      Weight decay for all models
        max_epochs:        Maximum epochs per fold
        patience:          Early stopping patience
        random_seed:       Random seed for split reproducibility
        output_dir:        Root directory for saving results
        concept_gt:        Optional concept ground truth (N, 12)
        device:            'cuda' for parallel, 'cpu' for serial fallback
        save_predictions:  Whether to save per-fold predictions
        raw_features:      Optional raw feature matrix with NaN values (N, 12).
                           When provided, per-fold imputation is performed inside
                           each fold to prevent data leakage.
        imputation_fn:     Optional callable with signature
                           (train_features, train_labels, val_features, val_labels)
                           -> (imputed_train, train_labels, imputed_val, val_labels).
                           Defaults to cross_val._default_per_fold_imputation.

    Returns:
        Dict mapping model_name -> {"fold_results": [...], "aggregated": {...}}
        Same format as run_cross_validation.
    """
    # --- Input validation ---
    if not model_names:
        return {}
    if len(set(model_names)) != len(model_names):
        raise ValueError(f"Duplicate model names: {model_names}")
    if model_kwargs_list is not None and len(model_kwargs_list) != len(model_names):
        raise ValueError(
            f"model_kwargs_list length ({len(model_kwargs_list)}) "
            f"must match model_names length ({len(model_names)})"
        )

    # CPU fallback: serial execution
    if not device.startswith("cuda"):
        from cbm_variable_stars.training.cross_val import run_cross_validation
        all_results = {}
        for i, model_name in enumerate(model_names):
            mkw = (model_kwargs_list[i] if model_kwargs_list else None) or {}
            all_results[model_name] = run_cross_validation(
                features=features, labels=labels,
                model_name=model_name, model_kwargs=mkw,
                n_folds=n_folds, batch_size=batch_size,
                learning_rate=learning_rate, weight_decay=weight_decay,
                max_epochs=max_epochs, patience=patience,
                random_seed=random_seed, output_dir=output_dir,
                concept_gt=concept_gt, device=device,
                save_predictions=save_predictions,
                training_mode="joint",
                raw_features=raw_features,
                imputation_fn=imputation_fn,
            )
        return all_results

    # === CUDA parallel path ===
    if model_kwargs_list is None:
        model_kwargs_list = [{} for _ in model_names]

    print(f"\n{'='*70}")
    print(f"  PARALLEL CROSS-VALIDATION ({len(model_names)} models x {n_folds} folds)")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    total_start = time.time()

    splits = create_cv_splits(labels, n_folds, random_seed)

    # Per-model accumulators: fold_results for each model
    all_fold_results: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in model_names
    }

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'#'*60}")
        print(f"# FOLD {fold_idx + 1}/{n_folds}")
        print(f"# Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"{'#'*60}")

        # === Per-fold imputation (Fix C4: prevent imputation data leakage) ===
        if raw_features is not None:
            from cbm_variable_stars.training.cross_val import _default_per_fold_imputation
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

            # Synchronize concept_gt with any rows removed during imputation
            if concept_gt is not None:
                if (len(fold_train_labels) < n_train_before
                        or len(fold_val_labels) < n_val_before):
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

        # === Per-fold standardization (Fix C3) ===
        fold_scaler = StandardScaler()
        train_features_scaled = fold_scaler.fit_transform(fold_train_features)
        val_features_scaled = fold_scaler.transform(fold_val_features)

        # === Shared datasets and data loaders ===
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

        # === Class weights (from training fold) ===
        class_weights = compute_class_weights(
            torch.tensor(fold_train_labels, dtype=torch.long)
        )

        # === Create model slots (one per model, each with its own stream) ===
        slots: List[ModelSlot] = []
        for i, model_name in enumerate(model_names):
            mkw = model_kwargs_list[i] or {}
            model = create_model(model_name, **mkw)

            use_concept_loss = model_name in ("hard_cbm_cal",)
            loss_fn = CBMJointLoss(
                alpha=1.0 if use_concept_loss else 0.0,
                beta=1.0,
                class_weights=class_weights.clone(),
                use_concept_loss=use_concept_loss,
            )

            slot = create_model_slot(
                name=model_name,
                model=model,
                loss_fn=loss_fn,
                device=device,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            slots.append(slot)

        # === Run parallel training ===
        output_path = Path(output_dir)
        trainer = ParallelTrainer(
            slots=slots,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            log_dir=str(output_path / "parallel_logs"),
            checkpoint_dir=str(output_path / "parallel_checkpoints"),
        )

        train_results = trainer.fit(train_loader, val_loader, fold_id=fold_idx)

        # === Collect per-model fold results (single forward pass) ===
        for slot in slots:
            model_name = slot.name
            slot.model.eval()

            # Single validation pass: get metrics AND predictions together
            final_metrics, all_preds, all_labels_fold = _validate_and_predict(
                slot.model, val_loader, slot.loss_fn, device
            )

            tr = train_results[model_name]
            fold_result: Dict[str, Any] = {
                "fold": fold_idx,
                "best_epoch": tr["best_epoch"],
                "training_time": tr["training_time"],
                "metrics": final_metrics,
                "predictions": all_preds,
                "true_labels": all_labels_fold,
                "val_indices": val_idx.tolist(),
            }
            all_fold_results[model_name].append(fold_result)

        # Clean up GPU memory between folds
        del slots, trainer, train_loader, val_loader
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start

    # === Aggregate results per model (same format as run_cross_validation) ===
    all_results: Dict[str, Dict[str, Any]] = {}
    for model_name in model_names:
        fold_results = all_fold_results[model_name]

        accuracies = [r["metrics"]["val_accuracy"] for r in fold_results]
        macro_f1s = [r["metrics"]["val_macro_f1"] for r in fold_results]
        per_class_f1_matrix = [
            r["metrics"]["val_per_class_f1"] for r in fold_results
        ]

        aggregated: Dict[str, Any] = {
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies, ddof=1)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
            "per_class_f1_mean": np.mean(per_class_f1_matrix, axis=0).tolist(),
            "per_class_f1_std": np.std(per_class_f1_matrix, axis=0, ddof=1).tolist(),
        }

        # Save per-model results JSON
        model_output_path = Path(output_dir) / model_name
        model_output_path.mkdir(parents=True, exist_ok=True)

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
            "training_mode": "joint",
            "parallel": True,
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

        with open(model_output_path / "cv_results.json", "w") as f:
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

        all_results[model_name] = {
            "fold_results": fold_results,
            "aggregated": aggregated,
        }

    print(f"\nTotal parallel CV time: {total_elapsed:.1f}s "
          f"(avg {total_elapsed/len(model_names):.1f}s per model)")

    return all_results


@torch.no_grad()
def _validate_and_predict(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
) -> Tuple[Dict[str, Any], List[int], List[int]]:
    """Validate a single model and collect predictions in one forward pass.

    Returns:
        (metrics_dict, predictions_list, true_labels_list)
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in val_loader:
        features = batch["features"].to(device)
        lab = batch["label"].to(device)

        output = model(features)

        loss_kwargs: Dict[str, Any] = {
            "model_output": output,
            "targets": lab,
        }
        concept_gt = batch.get("concept_gt")
        if concept_gt is not None:
            loss_kwargs["concept_targets"] = concept_gt.to(device)
        has_ogle = batch.get("has_ogle_match")
        if has_ogle is not None:
            loss_kwargs["has_concept_gt"] = has_ogle.to(device)

        loss_dict = loss_fn(**loss_kwargs)
        total_loss += loss_dict["total_loss"].item() * features.size(0)
        all_preds.append(output["logits"].argmax(dim=1).cpu())
        all_labels.append(lab.cpu())

    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()

    all_label_ids = list(range(N_CLASSES))
    accuracy = accuracy_score(all_labels_np, all_preds_np)
    macro_f1 = f1_score(
        all_labels_np, all_preds_np, labels=all_label_ids,
        average="macro", zero_division=0,
    )
    per_class_f1 = f1_score(
        all_labels_np, all_preds_np, labels=all_label_ids,
        average=None, zero_division=0,
    )

    metrics = {
        "val_loss": total_loss / len(all_labels_np),
        "val_accuracy": float(accuracy),
        "val_macro_f1": float(macro_f1),
        "val_per_class_f1": per_class_f1.tolist(),
    }

    return metrics, all_preds_np.tolist(), all_labels_np.tolist()
