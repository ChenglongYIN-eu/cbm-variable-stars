"""
Training callbacks for CBM variable star classification.

Provides:
    - EarlyStopping:      Monitor a metric, save best model, stop when no improvement
    - CheckpointCallback: Save model state to disk at configurable intervals
    - TrainingLogger:     Log training metrics to CSV
"""

import csv
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional


class EarlyStopping:
    """
    Early stopping callback -- monitors a validation metric and stops training
    when no improvement is seen for `patience` consecutive epochs.

    Optionally saves the best model checkpoint.

    Note: Trainer uses its own inline early stopping logic. This class is
    provided as a standalone utility for custom training loops.

    Args:
        patience:      Number of epochs without improvement before stopping
        monitor:       Metric name to monitor (e.g., 'val_macro_f1')
        mode:          'max' to maximize metric, 'min' to minimize
        min_delta:     Minimum change to qualify as an improvement
        save_best:     Whether to save the best model checkpoint
        checkpoint_dir: Directory for saving best model
        verbose:       Whether to print messages on improvement
    """

    def __init__(
        self,
        patience: int = 15,
        monitor: str = "val_macro_f1",
        mode: str = "max",
        min_delta: float = 0.0,
        save_best: bool = True,
        checkpoint_dir: str = "checkpoints",
        verbose: bool = True,
    ) -> None:
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.save_best = save_best
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose

        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

        if save_best:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _is_improvement(self, current: float) -> bool:
        """Check if current metric value is an improvement over the best."""
        if self.mode == "max":
            return current > self.best_value + self.min_delta
        else:
            return current < self.best_value - self.min_delta

    def step(
        self,
        metrics: Dict[str, Any],
        model: nn.Module,
        epoch: int,
        fold_id: int = 0,
    ) -> bool:
        """
        Execute one step of early stopping check.

        Args:
            metrics:  Dict containing the monitored metric.
            model:    Model to save if improved.
            epoch:    Current epoch index.
            fold_id:  Fold index for checkpoint naming.

        Returns:
            True if training should stop, False otherwise.
        """
        current = metrics.get(self.monitor)
        if current is None:
            return False

        if self._is_improvement(current):
            if self.verbose:
                print(
                    f"  EarlyStopping: {self.monitor} improved "
                    f"{self.best_value:.4f} -> {current:.4f} "
                    f"(epoch {epoch})"
                )
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0

            if self.save_best:
                ckpt_path = self.checkpoint_dir / f"best_model_fold{fold_id}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_metric": self.best_value,
                        "monitor": self.monitor,
                    },
                    ckpt_path,
                )
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(
                    f"  EarlyStopping: no improvement for {self.counter} epochs "
                    f"(patience={self.patience})"
                )

        if self.counter >= self.patience:
            self.should_stop = True
            if self.verbose:
                print(
                    f"\n  EarlyStopping triggered at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch}, "
                    f"Best {self.monitor}: {self.best_value:.4f}"
                )

        return self.should_stop

    def reset(self) -> None:
        """Reset state for next fold."""
        self.best_value = float("-inf") if self.mode == "max" else float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False


class CheckpointCallback:
    """
    Checkpoint callback -- saves model state at configurable intervals.

    Can save every N epochs or only the best model.

    Args:
        checkpoint_dir: Directory for saving checkpoints
        save_every:     Save every N epochs (None = only save best)
        keep_last_n:    Keep only last N regular checkpoints (0 = keep all)
        verbose:        Whether to print checkpoint save messages
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_every: Optional[int] = None,
        keep_last_n: int = 0,
        verbose: bool = False,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.verbose = verbose
        self._saved_paths: List[Path] = []

    def step(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, Any],
        fold_id: int = 0,
    ) -> Optional[Path]:
        """
        Save checkpoint if conditions are met.

        Args:
            model:    Model to checkpoint.
            epoch:    Current epoch index.
            metrics:  Current epoch metrics (saved with checkpoint).
            fold_id:  Fold index for file naming.

        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        if self.save_every is None or (epoch + 1) % self.save_every != 0:
            return None

        ckpt_path = self.checkpoint_dir / f"checkpoint_fold{fold_id}_epoch{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            },
            ckpt_path,
        )
        self._saved_paths.append(ckpt_path)

        if self.verbose:
            print(f"  Checkpoint saved: {ckpt_path}")

        # Prune old checkpoints
        if self.keep_last_n > 0 and len(self._saved_paths) > self.keep_last_n:
            old_path = self._saved_paths.pop(0)
            if old_path.exists():
                old_path.unlink()

        return ckpt_path

    def save_best(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, Any],
        fold_id: int = 0,
    ) -> Path:
        """
        Save best model checkpoint (called externally when improvement detected).

        Args:
            model:    Best model to save.
            epoch:    Best epoch index.
            metrics:  Best epoch metrics.
            fold_id:  Fold index for file naming.

        Returns:
            Path to saved checkpoint.
        """
        ckpt_path = self.checkpoint_dir / f"best_model_fold{fold_id}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            },
            ckpt_path,
        )
        return ckpt_path

    def load_best(
        self,
        model: nn.Module,
        fold_id: int = 0,
        device: str = "cpu",
    ) -> nn.Module:
        """
        Load the best model checkpoint.

        Args:
            model:    Model to load weights into.
            fold_id:  Fold index for file naming.
            device:   Device to map weights to.

        Returns:
            Model with best weights loaded.
        """
        ckpt_path = self.checkpoint_dir / f"best_model_fold{fold_id}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        return model


class TrainingLogger:
    """
    CSV training logger -- records per-epoch metrics to a CSV file.

    Args:
        log_dir:    Directory for CSV log files
        model_name: Model name (used in file naming)
        fold_id:    Fold index (used in file naming)
    """

    def __init__(
        self,
        log_dir: str = "logs",
        model_name: str = "model",
        fold_id: int = 0,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"training_log_{model_name}_fold{fold_id}.csv"
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._header_written = False

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log one epoch's metrics.

        Args:
            metrics: Dict of metric names to values.
                     List values are JSON-serialized automatically.
        """
        # Flatten list values to JSON strings for CSV compatibility
        row = {
            k: (json.dumps(v) if isinstance(v, list) else v)
            for k, v in metrics.items()
        }

        if not self._header_written:
            self._file = open(self.log_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
            self._header_written = True

        if self._writer is not None:
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __del__(self) -> None:
        self.close()
