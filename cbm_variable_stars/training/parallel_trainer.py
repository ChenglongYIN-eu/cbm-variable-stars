"""
CUDA Streams parallel trainer for training multiple CBM models simultaneously.

Key insight: individual CBM models (78-12K parameters) are too small to saturate
a modern GPU. By training 6 models concurrently on separate CUDA streams, we can
utilize 12-24 out of 26 SMs on an RTX 5060.

All models share the same DataLoader and batch transfer (one CPU→GPU copy per batch),
while each model has its own stream, optimizer, scheduler, and early stopping state.

Performance note: .item() and .cpu() calls are kept OUTSIDE stream contexts to
avoid implicit synchronization that would serialize parallel execution. Metrics
are accumulated as GPU tensors and only converted to Python floats after
torch.cuda.synchronize().
"""

import time
import csv
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from sklearn.metrics import accuracy_score, f1_score
from cbm_variable_stars.shared.constants import N_CLASSES


@dataclass
class ModelSlot:
    """Complete training context for one model in the parallel trainer."""
    name: str
    model: nn.Module
    optimizer: AdamW
    scheduler: CosineAnnealingWarmRestarts
    loss_fn: nn.Module
    stream: torch.cuda.Stream
    best_metric: float = -1.0  # init to -1 so even F1=0.0 triggers first checkpoint
    best_epoch: int = 0
    patience_counter: int = 0
    done: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)


def create_model_slot(
    name: str,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> ModelSlot:
    """Create a ModelSlot with its own CUDA stream, optimizer, and scheduler.

    Requires a CUDA device — the parallel path guards this at a higher level.
    """
    model = model.to(device)
    # Move loss_fn (nn.Module) to device so class_weights etc. are on GPU
    loss_fn = loss_fn.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-6,
    )
    stream = torch.cuda.Stream(device=device)

    return ModelSlot(
        name=name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        stream=stream,
    )


class ParallelTrainer:
    """
    Train N models in parallel using CUDA Streams.

    Each model gets its own stream, optimizer, scheduler, and early stopping.
    All models share the same DataLoader — batch is transferred to GPU once
    on the default stream, then each model's stream waits for that transfer
    before running its own forward/backward/step.

    Args:
        slots:          List of ModelSlot instances (one per model)
        max_epochs:     Maximum training epochs
        patience:       Early stopping patience (per model)
        device:         CUDA device string
        log_dir:        Directory for per-model CSV training logs
        checkpoint_dir: Directory for per-model checkpoints
    """

    def __init__(
        self,
        slots: List[ModelSlot],
        max_epochs: int = 200,
        patience: int = 15,
        device: str = "cuda",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.slots = slots
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _active_slots(self) -> List[ModelSlot]:
        """Return slots that haven't triggered early stopping."""
        return [s for s in self.slots if not s.done]

    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Train all active models for one epoch in parallel.

        For each batch:
          1. Default stream: transfer batch to GPU (one copy for all models)
          2. Each active slot's stream: forward → loss → backward → step
          3. Synchronize all streams at end of batch
          4. Extract metrics from GPU tensors AFTER sync (avoids implicit sync)

        Returns:
            Dict mapping model_name -> {train_loss, train_concept_loss,
                                         train_cls_loss, train_accuracy}
        """
        active = self._active_slots()
        if not active:
            return {}

        for slot in active:
            slot.model.train()

        # GPU tensor accumulators — avoid .item() inside stream context
        accum = {
            slot.name: {
                "loss": torch.zeros(1, device=self.device),
                "concept_loss": torch.zeros(1, device=self.device),
                "cls_loss": torch.zeros(1, device=self.device),
                "correct": torch.zeros(1, dtype=torch.long, device=self.device),
                "total": 0,
            }
            for slot in active
        }

        default_stream = torch.cuda.default_stream(self.device)

        for batch in train_loader:
            # Step 1: Transfer batch on default stream (shared, one copy)
            features = batch["features"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            bs = features.size(0)  # CPU-side shape query, no sync needed

            concept_gt = batch.get("concept_gt")
            if concept_gt is not None:
                concept_gt = concept_gt.to(self.device, non_blocking=True)
            has_ogle = batch.get("has_ogle_match")
            if has_ogle is not None:
                has_ogle = has_ogle.to(self.device, non_blocking=True)

            # Step 2: Each model processes the batch on its own stream
            for slot in active:
                with torch.cuda.stream(slot.stream):
                    # Wait for data transfer to complete
                    slot.stream.wait_stream(default_stream)

                    output = slot.model(features)

                    loss_kwargs: Dict[str, Any] = {
                        "model_output": output,
                        "targets": labels,
                    }
                    if concept_gt is not None:
                        loss_kwargs["concept_targets"] = concept_gt
                    if has_ogle is not None:
                        loss_kwargs["has_concept_gt"] = has_ogle

                    loss_dict = slot.loss_fn(**loss_kwargs)
                    loss = loss_dict["total_loss"]

                    slot.optimizer.zero_grad()
                    loss.backward()
                    # clip_grad_norm_ stays on GPU in modern PyTorch (no .item())
                    torch.nn.utils.clip_grad_norm_(
                        slot.model.parameters(), max_norm=5.0
                    )
                    slot.optimizer.step()

                    # Accumulate on GPU tensors — NO .item() here
                    a = accum[slot.name]
                    a["loss"] += loss.detach() * bs
                    a["concept_loss"] += loss_dict["concept_loss"].detach() * bs
                    a["cls_loss"] += loss_dict["classification_loss"].detach() * bs
                    preds = output["logits"].argmax(dim=1)
                    a["correct"] += (preds == labels).sum()
                    a["total"] += bs

            # Step 3: Synchronize all streams before next batch
            torch.cuda.synchronize(self.device)

        # Step 4: Convert GPU accumulators to Python floats AFTER all batches
        results = {}
        for slot in active:
            a = accum[slot.name]
            t = a["total"]
            results[slot.name] = {
                "train_loss": a["loss"].item() / t,
                "train_concept_loss": a["concept_loss"].item() / t,
                "train_cls_loss": a["cls_loss"].item() / t,
                "train_accuracy": a["correct"].item() / t,
            }
        return results

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, Dict[str, Any]]:
        """
        Validate all active models in parallel (no_grad).

        Returns:
            Dict mapping model_name -> {val_loss, val_accuracy, val_macro_f1,
                                         val_per_class_f1}
        """
        active = self._active_slots()
        if not active:
            return {}

        for slot in active:
            slot.model.eval()

        # Per-slot collectors: GPU tensors for loss, GPU lists for preds
        collectors = {
            slot.name: {
                "loss": torch.zeros(1, device=self.device),
                "preds": [],
                "total": 0,
            }
            for slot in active
        }
        # Labels are shared across all models — collect once
        all_batch_labels: List[torch.Tensor] = []

        default_stream = torch.cuda.default_stream(self.device)

        for batch in val_loader:
            features = batch["features"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            bs = features.size(0)

            concept_gt = batch.get("concept_gt")
            if concept_gt is not None:
                concept_gt = concept_gt.to(self.device, non_blocking=True)
            has_ogle = batch.get("has_ogle_match")
            if has_ogle is not None:
                has_ogle = has_ogle.to(self.device, non_blocking=True)

            for slot in active:
                with torch.cuda.stream(slot.stream):
                    slot.stream.wait_stream(default_stream)

                    output = slot.model(features)

                    loss_kwargs: Dict[str, Any] = {
                        "model_output": output,
                        "targets": labels,
                    }
                    if concept_gt is not None:
                        loss_kwargs["concept_targets"] = concept_gt
                    if has_ogle is not None:
                        loss_kwargs["has_concept_gt"] = has_ogle

                    loss_dict = slot.loss_fn(**loss_kwargs)

                    # Accumulate on GPU — no .item()/.cpu() inside stream
                    collectors[slot.name]["loss"] += loss_dict["total_loss"].detach() * bs
                    collectors[slot.name]["preds"].append(
                        output["logits"].argmax(dim=1)  # stay on GPU
                    )
                    collectors[slot.name]["total"] += bs

            # Collect labels once per batch (on default stream, outside slot loop)
            all_batch_labels.append(labels)

            torch.cuda.synchronize(self.device)

        # Convert to CPU AFTER full sync
        labels_np = torch.cat(all_batch_labels).cpu().numpy()
        all_label_ids = list(range(N_CLASSES))

        results = {}
        for slot in active:
            c = collectors[slot.name]
            preds_np = torch.cat(c["preds"]).cpu().numpy()
            n = c["total"]

            accuracy = accuracy_score(labels_np, preds_np)
            macro_f1 = f1_score(
                labels_np, preds_np, labels=all_label_ids,
                average="macro", zero_division=0
            )
            per_class_f1 = f1_score(
                labels_np, preds_np, labels=all_label_ids,
                average=None, zero_division=0
            )

            results[slot.name] = {
                "val_loss": c["loss"].item() / n,
                "val_accuracy": float(accuracy),
                "val_macro_f1": float(macro_f1),
                "val_per_class_f1": per_class_f1.tolist(),
            }
        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_id: int = 0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Full parallel training loop with per-model early stopping and checkpointing.

        Training continues until all models have triggered early stopping or
        max_epochs is reached. Each model independently tracks its best metric
        and patience counter.

        Args:
            train_loader: Shared training DataLoader
            val_loader:   Shared validation DataLoader
            fold_id:      Current CV fold index (for checkpoint naming)

        Returns:
            Dict mapping model_name -> {best_epoch, best_metric, training_time,
                                         total_epochs, history}
        """
        n_models = len(self.slots)
        if n_models == 0:
            return {}

        active_names = [s.name for s in self.slots]
        total_params = sum(
            sum(p.numel() for p in s.model.parameters()) for s in self.slots
        )

        print(f"\n{'='*60}")
        print(f"Fold {fold_id} | Parallel training {n_models} models | "
              f"Total params: {total_params:,}")
        print(f"  Models: {', '.join(active_names)}")
        print(f"{'='*60}")

        start_time = time.time()
        final_epoch = 0

        for epoch in range(self.max_epochs):
            final_epoch = epoch

            if not self._active_slots():
                break

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Per-model: update scheduler, early stopping, checkpointing
            for slot in self.slots:
                if slot.done:
                    continue

                slot.scheduler.step()
                current_lr = slot.optimizer.param_groups[0]["lr"]

                tm = train_metrics.get(slot.name, {})
                vm = val_metrics.get(slot.name, {})

                # History record
                epoch_record = {
                    "epoch": epoch,
                    "lr": current_lr,
                    **{k: v for k, v in tm.items()},
                    **{k: v for k, v in vm.items() if k != "val_per_class_f1"},
                }
                slot.history.append(epoch_record)

                current_f1 = vm.get("val_macro_f1", 0.0)

                if current_f1 > slot.best_metric:
                    slot.best_metric = current_f1
                    slot.best_epoch = epoch
                    slot.patience_counter = 0

                    # Save checkpoint
                    ckpt_path = (
                        self.checkpoint_dir
                        / f"best_{slot.name}_fold{fold_id}.pt"
                    )
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": slot.model.state_dict(),
                        "optimizer_state_dict": slot.optimizer.state_dict(),
                        "best_metric": slot.best_metric,
                        "val_metrics": vm,
                    }, ckpt_path)
                else:
                    slot.patience_counter += 1

                if slot.patience_counter >= self.patience:
                    print(
                        f"  [{slot.name}] Early stop at epoch {epoch}. "
                        f"Best: epoch {slot.best_epoch}, "
                        f"F1={slot.best_metric:.4f}"
                    )
                    slot.done = True

            # Periodic logging
            if epoch % 10 == 0 or not self._active_slots():
                active_status = []
                for slot in self.slots:
                    vm = val_metrics.get(slot.name, {})
                    f1 = vm.get("val_macro_f1", 0.0)
                    tag = "done" if slot.done else f"F1={f1:.4f}"
                    active_status.append(f"{slot.name}:{tag}")
                print(f"  Epoch {epoch:3d} | {' | '.join(active_status)}")

        elapsed = time.time() - start_time
        print(f"\nParallel training time: {elapsed:.1f}s "
              f"(avg {elapsed/n_models:.1f}s per model)")

        # Load best checkpoints and save training logs
        results = {}
        for slot in self.slots:
            ckpt_path = (
                self.checkpoint_dir / f"best_{slot.name}_fold{fold_id}.pt"
            )
            if ckpt_path.exists():
                best_ckpt = torch.load(
                    ckpt_path, weights_only=False, map_location=self.device,
                )
                slot.model.load_state_dict(best_ckpt["model_state_dict"])

            # Save CSV log
            if slot.history:
                log_path = (
                    self.log_dir / f"training_log_{slot.name}_fold{fold_id}.csv"
                )
                with open(log_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=slot.history[0].keys()
                    )
                    writer.writeheader()
                    for record in slot.history:
                        row = {
                            k: (json.dumps(v) if isinstance(v, list) else v)
                            for k, v in record.items()
                        }
                        writer.writerow(row)

            results[slot.name] = {
                "best_epoch": slot.best_epoch,
                "best_metric": slot.best_metric,
                "training_time": elapsed,
                "total_epochs": final_epoch + 1,
                "history": slot.history,
            }

        return results
