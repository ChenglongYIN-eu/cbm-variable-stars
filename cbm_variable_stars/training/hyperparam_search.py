"""
Hyperparameter search utilities for CBM variable star classification.

Provides:
    - grid_search:            Grid search over param_grid using cross-validation
    - get_default_param_grid: Return model-specific default search grids
    - HYPERPARAM_GRID:        Full hyperparameter search space
    - DEFAULT_HYPERPARAMS:    Recommended default values

Recommended two-stage search strategy:
    Stage 1: Coarse search (single fold)
        - Fixed hidden_dims=[64,32], batch_norm=True
        - Search: lr x weight_decay x dropout x batch_size
        - 3 x 4 x 4 x 4 = 192 combinations
        - Each ~30s -> Total ~1.6 hours

    Stage 2: Fine search (all 5 folds)
        - Take top-5 configs from Stage 1
        - Run full 5-fold CV
        - 5 x 5 folds x ~30s = ~12.5 minutes

    Total hyperparameter search time: ~2 hours (CPU)
"""

import numpy as np
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler

from cbm_variable_stars.shared.constants import N_CV_FOLDS, RANDOM_SEED, DEFAULT_PATIENCE


# ===== Full hyperparameter search space =====
HYPERPARAM_GRID: Dict[str, List[Any]] = {
    # Model architecture
    "hidden_dims": [[64, 32], [32, 16], [128, 64], [64]],
    "dropout_rate": [0.1, 0.2, 0.3, 0.5],
    "use_batch_norm": [True, False],

    # Training
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0, 1e-5, 1e-4, 1e-3],
    "batch_size": [64, 128, 256, 512],

    # Loss function (Plan B only)
    "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
    "label_smoothing": [0.0, 0.05, 0.1],

    # LR scheduler
    "scheduler_T0": [30, 50, 80],
}

# ===== Recommended default values =====
DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "hidden_dims": [64, 32],
    "dropout_rate": 0.3,
    "use_batch_norm": True,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 256,
    "alpha": 0.0,
    "label_smoothing": 0.05,
    "scheduler_T0": 50,
    "max_epochs": 200,
    "patience": DEFAULT_PATIENCE,
}


def get_default_param_grid(model_name: str) -> Dict[str, List[Any]]:
    """
    Return the default search grid for a specific model type.

    The grid is designed for Stage 1 coarse search (single fold).
    Plan B has additional alpha search; Plan A fixes alpha=0.

    Args:
        model_name: Model registry key (e.g., 'hard_cbm', 'hard_cbm_cal')

    Returns:
        Dict mapping hyperparameter names to lists of values to search.
    """
    # Base grid: training hyperparams (shared by all models)
    base_grid: Dict[str, List[Any]] = {
        "learning_rate": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0, 1e-5, 1e-4, 1e-3],
        "dropout_rate": [0.1, 0.3, 0.5],
        "batch_size": [128, 256, 512],
    }

    if model_name in ("hard_cbm_cal",):
        # Plan B: also search alpha (concept loss weight)
        base_grid["alpha"] = [0.1, 0.5, 1.0, 2.0, 5.0]
        base_grid["label_smoothing"] = [0.0, 0.05, 0.1]
    elif model_name in ("hard_cbm", "hard_cbm_linear"):
        # Plan A: also search label predictor complexity
        if model_name == "hard_cbm":
            base_grid["hidden_dims"] = [[32], [64, 32], [128, 64], [128, 64, 32]]
        # Plan A always has alpha=0 (no concept loss)
    elif model_name in ("soft_cbm",):
        base_grid["hidden_dims"] = [[64, 32], [128, 64]]
        base_grid["concept_embed_dim"] = [4, 8]
    elif model_name in ("cem",):
        base_grid["hidden_dims"] = [[64, 32], [128, 64]]
        base_grid["concept_embed_dim"] = [4, 8]
    elif model_name in ("mlp",):
        base_grid["hidden_dims"] = [[64, 32], [128, 64], [256, 128, 64]]

    return base_grid


def grid_search(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_folds: int = 1,
    max_epochs: int = 100,
    patience: int = 15,
    random_seed: int = RANDOM_SEED,
    output_dir: str = "results/hyperparam_search",
    concept_gt: Optional[np.ndarray] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Grid search over hyperparameters using cross-validation.

    For Stage 1 (coarse search): use n_folds=1 (single validation fold).
    For Stage 2 (fine search): use n_folds=5 (full CV on top-K configs).

    Args:
        features:    Imputed (but NOT pre-standardized) feature matrix, shape (N, 12)
        labels:      Label array, shape (N,)
        model_name:  Model registry key
        param_grid:  Dict of param_name -> list of values to search.
                     If None, uses get_default_param_grid(model_name).
        n_folds:     Number of CV folds for each config evaluation
        max_epochs:  Max training epochs per config (use reduced for Stage 1)
        patience:    Early stopping patience
        random_seed: Random seed
        output_dir:  Directory for saving search results
        concept_gt:  Optional concept ground truth (Plan B)
        device:      Training device
        verbose:     Whether to print progress

    Returns:
        dict with keys:
            "best_params":   Best hyperparameter configuration
            "best_score":    Best validation macro F1
            "all_results":   List of (params, score) for all configurations
            "search_summary": Summary statistics of the search
    """
    import torch
    from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
    from cbm_variable_stars.data.splits import create_cv_splits
    from cbm_variable_stars.models import create_model
    from cbm_variable_stars.losses.cbm_loss import CBMJointLoss, compute_class_weights
    from cbm_variable_stars.training.trainer import Trainer

    if param_grid is None:
        param_grid = get_default_param_grid(model_name)

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    n_configs = len(all_combinations)

    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nGrid search: {model_name}")
        print(f"  Configurations: {n_configs}")
        print(f"  Folds per config: {n_folds}")
        print(f"  Total training runs: {n_configs * n_folds}")

    all_results: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_params: Dict[str, Any] = {}

    splits = create_cv_splits(labels, max(n_folds, N_CV_FOLDS), random_seed)
    # Use only first n_folds splits for Stage 1
    search_splits = splits[:n_folds]

    for config_idx, values in enumerate(all_combinations):
        params = dict(zip(param_names, values))

        if verbose and config_idx % 10 == 0:
            print(f"  Config {config_idx + 1}/{n_configs}: {params}")

        fold_scores: List[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(search_splits):
            # Separate model kwargs from training kwargs
            model_kwarg_keys = {"hidden_dims", "dropout_rate", "use_batch_norm",
                                "concept_embed_dim", "calibrator_hidden"}
            model_kwargs = {k: v for k, v in params.items() if k in model_kwarg_keys}
            batch_size = params.get("batch_size", 256)
            lr = params.get("learning_rate", 1e-3)
            wd = params.get("weight_decay", 1e-4)
            alpha = params.get("alpha", 0.0)
            label_smoothing = params.get("label_smoothing", 0.05)

            # Per-fold standardization (consistent with cross_val.py)
            fold_scaler = StandardScaler()
            train_features_scaled = fold_scaler.fit_transform(features[train_idx])
            val_features_scaled = fold_scaler.transform(features[val_idx])

            train_dataset = VariableStarDataset(
                features=train_features_scaled,
                labels=labels[train_idx],
                concept_gt=(concept_gt[train_idx] if concept_gt is not None else None),
            )
            val_dataset = VariableStarDataset(
                features=val_features_scaled,
                labels=labels[val_idx],
                concept_gt=(concept_gt[val_idx] if concept_gt is not None else None),
            )

            train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

            model = create_model(model_name, **model_kwargs)

            class_weights = compute_class_weights(
                torch.tensor(labels[train_idx], dtype=torch.long)
            )
            use_concept_loss = model_name in ("hard_cbm_cal",)
            loss_fn = CBMJointLoss(
                alpha=alpha if use_concept_loss else 0.0,
                beta=1.0,
                class_weights=class_weights,
                use_concept_loss=use_concept_loss,
                label_smoothing=label_smoothing,
            )

            trainer = Trainer(
                model=model,
                loss_fn=loss_fn,
                learning_rate=lr,
                weight_decay=wd,
                max_epochs=max_epochs,
                patience=patience,
                device=device,
                log_dir=str(output_path / "search_logs"),
                checkpoint_dir=str(output_path / "search_checkpoints"),
            )

            trainer.fit(train_loader, val_loader, fold_id=config_idx * n_folds + fold_idx)
            val_metrics = trainer.validate(val_loader)
            fold_scores.append(val_metrics["val_macro_f1"])

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        result: Dict[str, Any] = {
            "config_idx": config_idx,
            "params": params,
            "mean_macro_f1": mean_score,
            "std_macro_f1": std_score,
            "fold_scores": fold_scores,
        }
        all_results.append(result)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            if verbose:
                print(
                    f"    New best: macro_f1={best_score:.4f} "
                    f"+/- {std_score:.4f} | params={params}"
                )

    # Sort results by mean score
    all_results.sort(key=lambda r: r["mean_macro_f1"], reverse=True)

    search_summary: Dict[str, Any] = {
        "model_name": model_name,
        "n_configs": n_configs,
        "n_folds": n_folds,
        "best_score": best_score,
        "best_params": best_params,
        "top5_results": all_results[:5],
    }

    import json
    with open(output_path / "search_results.json", "w") as f:
        json.dump(search_summary, f, indent=2, default=str)

    if verbose:
        print(f"\nSearch complete. Best macro_f1: {best_score:.4f}")
        print(f"Best params: {best_params}")

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": all_results,
        "search_summary": search_summary,
    }
