"""
Core training pipeline for CBM variable star classification.

Provides:
    - Trainer:          Full training class with early stopping, checkpointing,
                        and CSV logging.
    - train_cbm:        Top-level wrapper function for CBM model training.
    - evaluate_model:   Full evaluation on a data loader.
    - train_baseline:   Baseline (RF/XGBoost) training wrapper.
    - evaluate_baseline: Baseline model evaluation.

Optimizer:   AdamW (decoupled weight decay, robust to LR choice)
LR schedule: CosineAnnealingWarmRestarts (T_0=50, T_mult=1, eta_min=1e-6)
Early stop:  patience=15, monitor val_macro_f1
Grad clip:   max_norm=5.0
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import time
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

from cbm_variable_stars.shared.constants import CLASS_NAMES
from cbm_variable_stars.evaluation.metrics import compute_all_metrics


class Trainer:
    """
    CBM model trainer.

    Optimizer: AdamW (decoupled weight decay, robust to LR selection)
    LR schedule: CosineAnnealingWarmRestarts (T_0=50, T_mult=1, eta_min=1e-6)
    Early stop: patience=15, monitor val_macro_f1
    Gradient clipping: max_norm=5.0

    Args:
        model:           Neural network model to train
        loss_fn:         Loss function (CBMJointLoss or similar)
        learning_rate:   Initial learning rate (default 1e-3)
        weight_decay:    L2 regularization weight (default 1e-4)
        max_epochs:      Maximum training epochs (default 200)
        patience:        Early stopping patience (default 15)
        device:          Training device ('cpu' or 'cuda')
        log_dir:         Directory for training CSV logs
        checkpoint_dir:  Directory for model checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 15,
        device: str = "cpu",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience

        self.lr = learning_rate
        self.weight_decay = weight_decay

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=1,
            eta_min=1e-6,
        )

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: List[Dict[str, Any]] = []

    def train_one_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for a single epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            dict with keys: train_loss, train_concept_loss, train_cls_loss, train_accuracy
        """
        self.model.train()
        total_loss = 0.0
        concept_loss_sum = 0.0
        cls_loss_sum = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            # Optional concept GT and OGLE match mask
            concept_gt = batch.get("concept_gt")
            if concept_gt is not None:
                concept_gt = concept_gt.to(self.device)
            has_ogle = batch.get("has_ogle_match")
            if has_ogle is not None:
                has_ogle = has_ogle.to(self.device)

            output = self.model(features)

            # Pass appropriate kwargs to loss function
            loss_kwargs: Dict[str, Any] = {
                "model_output": output,
                "targets": labels,
            }
            if concept_gt is not None:
                loss_kwargs["concept_targets"] = concept_gt
            if has_ogle is not None:
                loss_kwargs["has_concept_gt"] = has_ogle

            loss_dict = self.loss_fn(**loss_kwargs)
            loss = loss_dict["total_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=5.0
            )
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)
            concept_loss_sum += loss_dict["concept_loss"].item() * features.size(0)
            cls_loss_sum += loss_dict["classification_loss"].item() * features.size(0)

            preds = output["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            "train_loss": total_loss / total,
            "train_concept_loss": concept_loss_sum / total,
            "train_cls_loss": cls_loss_sum / total,
            "train_accuracy": correct / total,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Evaluate on validation set. Returns comprehensive classification metrics.

        Args:
            val_loader: Validation data loader.

        Returns:
            dict with keys: val_loss, val_accuracy, val_macro_f1, val_per_class_f1
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for batch in val_loader:
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(features)

            concept_gt = batch.get("concept_gt")
            if concept_gt is not None:
                concept_gt = concept_gt.to(self.device)

            has_ogle = batch.get("has_ogle_match")
            if has_ogle is not None:
                has_ogle = has_ogle.to(self.device)

            loss_kwargs_val: Dict[str, Any] = {
                "model_output": output,
                "targets": labels,
            }
            if concept_gt is not None:
                loss_kwargs_val["concept_targets"] = concept_gt
            if has_ogle is not None:
                loss_kwargs_val["has_concept_gt"] = has_ogle

            loss_dict = self.loss_fn(**loss_kwargs_val)

            total_loss += loss_dict["total_loss"].item() * features.size(0)
            all_preds.append(output["logits"].argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels).numpy()

        from sklearn.metrics import accuracy_score, f1_score

        # Infer number of classes from model output or observed labels
        n_classes_observed = max(int(all_labels_np.max()), int(all_preds_np.max())) + 1
        if hasattr(self.model, 'num_classes'):
            n_classes_observed = max(n_classes_observed, self.model.num_classes)
        all_labels_list = list(range(n_classes_observed))
        accuracy = accuracy_score(all_labels_np, all_preds_np)
        macro_f1 = f1_score(all_labels_np, all_preds_np, labels=all_labels_list, average="macro", zero_division=0)
        per_class_f1 = f1_score(all_labels_np, all_preds_np, labels=all_labels_list, average=None, zero_division=0)

        return {
            "val_loss": total_loss / len(all_labels_np),
            "val_accuracy": float(accuracy),
            "val_macro_f1": float(macro_f1),
            "val_per_class_f1": per_class_f1.tolist(),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping, checkpointing, and CSV logging.

        Args:
            train_loader: Training data loader.
            val_loader:   Validation data loader.
            fold_id:      Current fold index (used for file naming).

        Returns:
            dict with keys:
                "best_epoch":     Epoch with best validation macro F1
                "best_metric":    Best validation macro F1 value
                "training_time":  Total training time in seconds
                "total_epochs":   Number of epochs completed
                "history":        List of per-epoch metric dicts
        """
        model_name = self.model.__class__.__name__
        n_params = sum(p.numel() for p in self.model.parameters())

        print(f"\n{'='*60}")
        print(f"Fold {fold_id} | Model: {model_name} | Params: {n_params:,}")
        print(f"{'='*60}")

        start_time = time.time()
        epoch = 0

        for epoch in range(self.max_epochs):
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record: Dict[str, Any] = {
                "epoch": epoch,
                "lr": current_lr,
                **train_metrics,
                **{k: v for k, v in val_metrics.items() if k != "val_per_class_f1"},
            }
            self.history.append(epoch_record)

            if epoch % 10 == 0 or epoch == self.max_epochs - 1:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val F1: {val_metrics['val_macro_f1']:.4f} | "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

            current_metric = val_metrics["val_macro_f1"]

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.patience_counter = 0

                checkpoint_path = (
                    self.checkpoint_dir / f"best_model_fold{fold_id}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_metric": self.best_metric,
                        "val_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(
                    f"\nEarly stopping at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch}, "
                    f"Best macro_f1: {self.best_metric:.4f}"
                )
                break

        elapsed = time.time() - start_time
        print(f"Training time: {elapsed:.1f}s")

        # Save training log
        if self.history:
            log_path = self.log_dir / f"training_log_fold{fold_id}.csv"
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                for record in self.history:
                    row = {
                        k: (json.dumps(v) if isinstance(v, list) else v)
                        for k, v in record.items()
                    }
                    writer.writerow(row)

        # Load best model checkpoint
        checkpoint_path = self.checkpoint_dir / f"best_model_fold{fold_id}.pt"
        if checkpoint_path.exists():
            best_ckpt = torch.load(
                checkpoint_path,
                weights_only=False,
                map_location=self.device,
            )
            self.model.load_state_dict(best_ckpt["model_state_dict"])

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "training_time": elapsed,
            "total_epochs": epoch + 1,
            "history": self.history,
        }

    def reset(self) -> None:
        """
        Reset trainer state for the next CV fold.

        Resets metrics, patience counter, history, re-initializes model weights,
        and rebuilds optimizer and scheduler so they don't carry stale state
        (e.g. momentum buffers, LR schedule position) from the previous fold.
        """
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = []
        if hasattr(self.model, "_init_weights"):
            self.model._init_weights()
        # Reset BatchNorm running statistics to prevent cross-fold leakage
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.reset_running_stats()
        # Rebuild optimizer and scheduler to clear stale state from previous fold
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=1,
            eta_min=1e-6,
        )


# ============================================================
# [Fix M9] Complete core function definitions
# ============================================================

def train_cbm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 15,
    device: str = "cpu",
    log_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
    fold_id: int = 0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Top-level wrapper for training a CBM model.

    Provides a clean interface for experiment pipelines and scripts.
    Internally creates a Trainer and executes the full training loop.

    Args:
        model:          Model instance to train
        train_loader:   Training data loader
        val_loader:     Validation data loader
        loss_fn:        Loss function instance
        learning_rate:  Learning rate (default 1e-3)
        weight_decay:   L2 regularization weight (default 1e-4)
        max_epochs:     Maximum training epochs (default 200)
        patience:       Early stopping patience (default 15)
        device:         Training device ('cpu' or 'cuda')
        log_dir:        Directory for training logs
        checkpoint_dir: Directory for model checkpoints
        fold_id:        Current fold index

    Returns:
        Tuple of (trained_model, result_dict).
        result_dict contains: best_epoch, best_metric, training_time, history
    """
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
    )

    result = trainer.fit(train_loader, val_loader, fold_id=fold_id)

    return model, result


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on a given dataset with comprehensive metrics.

    Called by experiment pipelines to evaluate on in-domain test set and
    out-of-domain test set (OGLE).

    Args:
        model:       Trained model instance
        data_loader: Data loader for evaluation
        device:      Evaluation device
        class_names: List of class name strings

    Returns:
        dict with keys:
            "accuracy", "macro_f1", "weighted_f1",
            "macro_precision", "macro_recall",
            "per_class": {class_name: {f1, precision, recall, support}},
            "confusion_matrix": [[...]],
            "all_preds": [...],
            "all_labels": [...],
            "all_probs": ndarray (N, 6),
            "all_concepts": ndarray (N, 12)
    """
    if class_names is None:
        class_names = CLASS_NAMES

    model = model.to(device)
    model.eval()

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    all_concepts: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device)
            labels = batch["label"]

            output = model(features)

            preds = output["logits"].argmax(dim=1).cpu()
            probs = output["probabilities"].cpu()
            concepts = output["concepts"].cpu()

            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
            all_concepts.append(concepts)

    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    all_probs_np = torch.cat(all_probs).numpy()
    all_concepts_np = torch.cat(all_concepts).numpy()

    # Compute comprehensive metrics
    metrics = compute_all_metrics(all_labels_np, all_preds_np, class_names,
                                   y_pred_proba=all_probs_np)

    # Attach raw data for downstream statistical tests
    metrics["all_preds"] = all_preds_np.tolist()
    metrics["all_labels"] = all_labels_np.tolist()
    metrics["all_probs"] = all_probs_np.tolist()
    metrics["all_concepts"] = all_concepts_np.tolist()

    return metrics


def train_baseline(
    model_type: str,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    random_seed: int = 42,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a baseline model (RF or XGBoost) with a unified interface.

    Called by experiment pipelines for fair comparison with CBM models.

    Args:
        model_type:     "rf" or "xgb"
        features_train: Training features, shape (N_train, 12)
        labels_train:   Training labels, shape (N_train,)
        features_val:   Validation features, shape (N_val, 12)
        labels_val:     Validation labels, shape (N_val,)
        random_seed:    Random seed for reproducibility
        **kwargs:       Additional model hyperparameters
                        (n_estimators, max_depth, xgb_lr, etc.)

    Returns:
        Tuple of (trained_model, metrics_dict).
        metrics_dict contains: accuracy, macro_f1, per_class_f1,
                               predictions, feature_importance
    """
    from sklearn.metrics import accuracy_score, f1_score

    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 10),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
            oob_score=True,
        )
    elif model_type == "xgb":
        import xgboost as xgb
        n_classes = len(np.unique(labels_train))
        model = xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 300),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("xgb_lr", 0.1),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            gamma=0.1,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=random_seed,
            n_jobs=-1,
            verbosity=0,
        )
    else:
        raise ValueError(
            f"Unknown baseline type: '{model_type}'. Choose from: 'rf', 'xgb'."
        )

    model.fit(features_train, labels_train)

    y_pred = model.predict(features_val)
    all_label_ids = list(range(len(np.unique(np.concatenate([labels_train, labels_val])))))
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(labels_val, y_pred)),
        "macro_f1": float(f1_score(labels_val, y_pred, labels=all_label_ids, average="macro", zero_division=0)),
        "per_class_f1": f1_score(
            labels_val, y_pred, labels=all_label_ids, average=None, zero_division=0
        ).tolist(),
        "predictions": y_pred.tolist(),
        "feature_importance": model.feature_importances_.tolist(),
    }

    return model, metrics


def evaluate_baseline(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained baseline model (sklearn/XGBoost).

    Args:
        model:       Trained sklearn model instance
        features:    Test feature matrix, shape (N, 12)
        labels:      Test labels, shape (N,)
        class_names: List of class name strings

    Returns:
        Comprehensive metrics dict (same format as compute_all_metrics).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    y_pred = model.predict(features)
    return compute_all_metrics(labels, y_pred, class_names)
