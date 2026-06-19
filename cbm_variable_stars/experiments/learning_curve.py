"""
CBM Variable Star Classification -- Learning Curve Experiment

S2 fix: Maximum sample size = 12,000 (safe upper bound below single-fold
training set size of ~13,260).

Trains with progressively larger subsets to measure how performance scales
with training data volume.  Helps identify the data efficiency ceiling and
whether more data collection would be worthwhile.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_NAMES_12,
    LEARNING_CURVE_SAMPLE_SIZES,
    N_CLASSES,
    RANDOM_SEED,
)
from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.reproducibility import set_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(
    model_name: str,
    n_concepts: int,
    concept_names: List[str],
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "n_concepts": n_concepts,
        "concept_names": concept_names,
    }


def _stratified_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified random sub-sample of size `n_samples` from (features, labels).

    If n_samples >= len(features), returns the full dataset unchanged.

    Parameters
    ----------
    features : np.ndarray, shape (N, n_concepts)
    labels : np.ndarray, shape (N,)
    n_samples : int
    random_seed : int

    Returns
    -------
    (features_sub, labels_sub) both of length n_samples.
    """
    if n_samples >= len(features):
        return features, labels

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_samples,
        random_state=random_seed,
    )
    train_idx, _ = next(sss.split(features, labels))
    return features[train_idx], labels[train_idx]


def _run_single_training(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    model_name: str,
    n_concepts: int,
    cfg: Any,
    seed: int,
) -> Dict[str, Any]:
    """
    Train a model on a given (features_train, labels_train) subset and
    evaluate on (features_val, labels_val).

    Returns accuracy and macro F1 on the validation split.
    """
    from cbm_variable_stars.models import create_model
    from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
    from cbm_variable_stars.losses.cbm_loss import CBMJointLoss, compute_class_weights
    from cbm_variable_stars.training.trainer import Trainer
    import torch

    set_global_seed(seed)

    hp = getattr(cfg, "training", None) or {}
    batch_size = getattr(hp, "batch_size", 256)
    lr = getattr(hp, "learning_rate", 1e-3)
    wd = getattr(hp, "weight_decay", 1e-4)
    max_epochs = getattr(hp, "max_epochs", 200)
    patience = getattr(hp, "patience", 15)

    model = create_model(model_name, num_concepts=n_concepts, num_classes=N_CLASSES)

    train_dataset = VariableStarDataset(
        features=features_train,
        labels=labels_train,
    )
    val_dataset = VariableStarDataset(
        features=features_val,
        labels=labels_val,
    )
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weights(
        torch.tensor(labels_train, dtype=torch.long)
    )
    loss_fn = CBMJointLoss(
        alpha=0.0,
        beta=1.0,
        class_weights=class_weights,
        use_concept_loss=False,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        learning_rate=lr,
        weight_decay=wd,
        max_epochs=max_epochs,
        patience=patience,
        log_dir="/tmp/cbm_lc_logs",
        checkpoint_dir="/tmp/cbm_lc_checkpoints",
    )

    t0 = time.time()
    train_result = trainer.fit(train_loader, val_loader, fold_id=0)
    elapsed = time.time() - t0

    val_metrics = trainer.validate(val_loader)

    from sklearn.metrics import f1_score

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch["features"])
            all_preds.extend(out["logits"].argmax(1).tolist())
            all_labels.extend(batch["label"].tolist())

    macro_f1 = float(
        f1_score(all_labels, all_preds, labels=list(range(N_CLASSES)),
                 average="macro", zero_division=0)
    )

    return {
        "accuracy": float(val_metrics.get("val_accuracy", 0.0)),
        "macro_f1": macro_f1,
        "training_time_sec": elapsed,
        "best_epoch": train_result.get("best_epoch", 0),
    }


# ---------------------------------------------------------------------------
# run_learning_curve
# ---------------------------------------------------------------------------

def run_learning_curve(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str = "hard_cbm",
    sample_sizes: Optional[List[int]] = None,
    n_repeats: int = 3,
    cfg: Any = None,
    output_dir: Optional[Path] = None,
    val_fraction: float = 0.2,
) -> Dict[str, Any]:
    """
    Train with different sample sizes and record performance.

    For each sample size `n`:
        - Stratified sub-sample n training examples from the CV pool.
        - Hold out `val_fraction` of the *entire* pool as fixed validation.
        - Repeat `n_repeats` times with different random seeds.
        - Record mean and std of accuracy and macro F1.

    Parameters
    ----------
    features : np.ndarray
        Full feature matrix (CV pool), shape (N, n_concepts).
        Column order must match CONCEPT_NAMES_12.
    labels : np.ndarray
        Integer class labels, shape (N,).
    model_name : str
        Model name as registered in models.__init__.create_model().
    sample_sizes : list of int or None
        Training set sizes to evaluate.
        Defaults to LEARNING_CURVE_SAMPLE_SIZES =
        [500, 1000, 2000, 4000, 6000, 8000, 10000, 12000].
    n_repeats : int
        Number of independent repeats per sample size.  Default 3.
    cfg : DictConfig or None
        Project configuration for training hyper-parameters.
    output_dir : Path or None
        If provided, results are saved here.
    val_fraction : float
        Fraction of data reserved as the fixed validation set.

    Returns
    -------
    dict
        {
          "_metadata": {...},
          "sample_sizes": [...],
          "n_repeats": int,
          "results": {
              500: {
                  "mean_accuracy": float,
                  "std_accuracy": float,
                  "mean_f1": float,
                  "std_f1": float,
                  "all_accuracies": [...],
                  "all_f1s": [...],
                  "mean_training_time": float,
              },
              ...
          }
        }

    Notes on S2 fix
    ---------------
    Maximum sample size is capped at 12,000 (LEARNING_CURVE_SAMPLE_SIZES max).
    This is safely below the single-fold training set size of ~13,260.
    """
    if sample_sizes is None:
        sample_sizes = LEARNING_CURVE_SAMPLE_SIZES  # [500,1000,...,12000]

    if cfg is None:
        # Minimal stub to avoid AttributeError
        class _Cfg:
            class training:
                batch_size = 256
                learning_rate = 1e-3
                weight_decay = 1e-4
                max_epochs = 100  # shorter for LC experiment
                patience = 15
            class project:
                random_seed = RANDOM_SEED

        cfg = _Cfg()

    seed = getattr(getattr(cfg, "project", cfg), "random_seed", RANDOM_SEED)
    n_concepts = features.shape[1]

    logger.info("=" * 60)
    logger.info(f"Learning Curve Experiment: model={model_name}")
    logger.info(f"  Sample sizes: {sample_sizes}")
    logger.info(f"  n_repeats: {n_repeats}")
    logger.info(f"  n_concepts: {n_concepts}")
    logger.info(f"  Total dataset: {len(features)} samples")
    logger.info("=" * 60)

    # Create a fixed held-out validation set
    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=seed,
    )
    train_pool_idx, val_idx = next(sss_val.split(features, labels))

    features_pool = features[train_pool_idx]
    labels_pool = labels[train_pool_idx]
    features_val = features[val_idx]
    labels_val = labels[val_idx]

    logger.info(
        f"  Training pool: {len(features_pool)} | "
        f"Validation (fixed): {len(features_val)}"
    )

    t_total = time.time()
    results: Dict[str, Any] = {}

    for n in sample_sizes:
        if n > len(features_pool):
            logger.warning(
                f"  Sample size {n} > training pool size {len(features_pool)}. "
                f"Skipping."
            )
            continue

        logger.info(f"\n  Sample size n={n}:")
        repeat_accs: List[float] = []
        repeat_f1s: List[float] = []
        repeat_times: List[float] = []

        for rep in range(n_repeats):
            rep_seed = seed + rep * 1000 + n
            set_global_seed(rep_seed)

            # Sub-sample n training examples
            features_sub, labels_sub = _stratified_subsample(
                features_pool, labels_pool, n_samples=n, random_seed=rep_seed
            )

            try:
                metrics = _run_single_training(
                    features_train=features_sub,
                    labels_train=labels_sub,
                    features_val=features_val,
                    labels_val=labels_val,
                    model_name=model_name,
                    n_concepts=n_concepts,
                    cfg=cfg,
                    seed=rep_seed,
                )
                repeat_accs.append(metrics["accuracy"])
                repeat_f1s.append(metrics["macro_f1"])
                repeat_times.append(metrics["training_time_sec"])

                logger.info(
                    f"    rep {rep+1}/{n_repeats}: "
                    f"acc={metrics['accuracy']:.4f}, "
                    f"f1={metrics['macro_f1']:.4f}, "
                    f"time={metrics['training_time_sec']:.1f}s"
                )
            except Exception as e:
                logger.error(f"    rep {rep+1}/{n_repeats} failed: {e}")
                repeat_accs.append(float("nan"))
                repeat_f1s.append(float("nan"))
                repeat_times.append(float("nan"))

        results[n] = {
            "n_samples": n,
            "mean_accuracy": float(np.nanmean(repeat_accs)),
            "std_accuracy": float(np.nanstd(repeat_accs, ddof=1)),
            "mean_f1": float(np.nanmean(repeat_f1s)),
            "std_f1": float(np.nanstd(repeat_f1s, ddof=1)),
            "mean_training_time": float(np.nanmean(repeat_times)),
            "all_accuracies": repeat_accs,
            "all_f1s": repeat_f1s,
        }

        logger.info(
            f"  n={n}: "
            f"acc={results[n]['mean_accuracy']:.4f}±{results[n]['std_accuracy']:.4f}, "
            f"f1={results[n]['mean_f1']:.4f}±{results[n]['std_f1']:.4f}"
        )

    elapsed = time.time() - t_total

    concept_names = (
        CONCEPT_NAMES_12[:n_concepts]
        if n_concepts <= 12
        else CONCEPT_NAMES_12
    )

    output = {
        "_metadata": _make_metadata(
            model_name=model_name,
            n_concepts=n_concepts,
            concept_names=concept_names,
            random_seed=seed,
        ),
        "model_name": model_name,
        "sample_sizes": [n for n in sample_sizes if n in results],
        "n_repeats": n_repeats,
        "val_fraction": val_fraction,
        "n_val_samples": int(len(features_val)),
        "results": {str(n): v for n, v in results.items()},
        "elapsed_seconds": elapsed,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "learning_curve.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"\nLearning curve results saved to {out_path}")

    logger.info(f"\nLearning curve experiment complete in {elapsed:.1f}s")
    return output


# ---------------------------------------------------------------------------
# Convenience: run with multiple models for comparison
# ---------------------------------------------------------------------------

def run_learning_curve_comparison(
    features: np.ndarray,
    labels: np.ndarray,
    model_names: Optional[List[str]] = None,
    sample_sizes: Optional[List[int]] = None,
    n_repeats: int = 3,
    cfg: Any = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run learning curves for multiple models and compare.

    Parameters
    ----------
    features : np.ndarray
    labels : np.ndarray
    model_names : list of str or None
        Default: ["hard_cbm", "rf_baseline", "xgb_baseline"]
    sample_sizes : list of int or None
    n_repeats : int
    cfg : DictConfig or None
    output_dir : Path or None

    Returns
    -------
    dict
        {model_name: learning_curve_result}
    """
    if model_names is None:
        model_names = ["hard_cbm", "rf_baseline", "xgb_baseline"]

    output_dir = Path(output_dir) if output_dir else None
    all_results: Dict[str, Any] = {}

    for model_name in model_names:
        logger.info(f"\nLearning curve for model: {model_name}")
        sub_dir = output_dir / model_name if output_dir else None
        try:
            result = run_learning_curve(
                features=features,
                labels=labels,
                model_name=model_name,
                sample_sizes=sample_sizes,
                n_repeats=n_repeats,
                cfg=cfg,
                output_dir=sub_dir,
            )
            all_results[model_name] = result
        except Exception as e:
            logger.error(f"Learning curve for {model_name} failed: {e}")
            all_results[model_name] = {"error": str(e)}

    return all_results
