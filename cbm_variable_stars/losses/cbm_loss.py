"""
CBM loss functions for variable star classification.

Provides joint training loss, calibration loss with partial concept GT,
sequential loss for Plan B, and independent training loss.

Loss summary:
    - CBMJointLoss:       L = alpha * L_concept + beta * L_cls
    - CBMCalibrationLoss: Plan B joint loss with has_concept_gt mask
    - CBMSequentialLoss:  Two-stage (stage1: concept only, stage2: cls only)
    - CBMIndependentLoss: Independent concept + cls loss with detach

Plan A:  alpha=0, beta=1  (no concept loss)
Plan B:  alpha=1, beta=1  (joint concept + cls loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from cbm_variable_stars.shared.constants import NUM_CLASSES


class CBMJointLoss(nn.Module):
    """
    CBM joint training loss function.

    L_total = alpha * L_concept + beta * L_classification

    Key design decisions:

    1. L_concept definition:
       - Plan A / Plan A-Linear: L_concept = 0 (alpha=0)
         Concept layer IS the input; no "prediction" error
       - Plan B (concept calibration): L_concept = (1/12) * SUM_i MSE(c_pred_i, c_gt_i)
         c_gt_i comes from OGLE high-precision extracted values

    2. L_classification:
       - CrossEntropyLoss + class weights + label_smoothing
       - Class weight formula: w_k = N_total / (K * N_k)

    3. alpha and beta:
       - Plan A: alpha=0, beta=1
       - Plan B recommended: alpha=1.0, beta=1.0 (starting values)

    Args:
        alpha:             Concept loss weight (default 0.0 for Plan A)
        beta:              Classification loss weight (default 1.0)
        class_weights:     Class weight tensor, shape (num_classes,)
        concept_loss_type: "mse" or "per_concept_mse"
        use_concept_loss:  Whether to compute concept loss (False for Plan A)
        label_smoothing:   Label smoothing coefficient (default 0.05)
    """

    def __init__(
        self,
        alpha: float = 0.0,
        beta: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        concept_loss_type: str = "mse",
        use_concept_loss: bool = False,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_concept_loss = use_concept_loss
        self.concept_loss_type = concept_loss_type

        # Store class_weights as a buffer so .to(device) moves them automatically
        if class_weights is not None:
            self.register_buffer("_class_weights", class_weights)
        else:
            self._class_weights = None
        self._label_smoothing = label_smoothing
        self.classification_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.concept_mse = nn.MSELoss(reduction="mean")

    def _ensure_device(self, device: torch.device) -> None:
        """Rebuild CrossEntropyLoss if weights are on wrong device."""
        w = self.classification_loss.weight
        if w is not None and w.device != device:
            self.classification_loss = nn.CrossEntropyLoss(
                weight=w.to(device),
                label_smoothing=self._label_smoothing,
            )

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute joint loss.

        Args:
            model_output:    Model forward() return dict
            targets:         Classification labels, shape (batch_size,), dtype=long
            concept_targets: Concept ground truth, shape (batch_size, 12)
                             Only required for Plan B
            **kwargs:        Additional keyword arguments (ignored, for signature compat)

        Returns:
            dict with keys:
                "total_loss":          Total loss (for backward())
                "concept_loss":        Concept loss (for logging)
                "classification_loss": Classification loss (for logging)
        """
        logits = model_output["logits"]
        self._ensure_device(logits.device)
        L_cls = self.classification_loss(logits, targets)

        L_concept = torch.tensor(0.0, device=logits.device)
        if self.use_concept_loss and concept_targets is not None:
            concepts_pred = model_output["concepts"]

            if self.concept_loss_type == "mse":
                L_concept = self.concept_mse(concepts_pred, concept_targets)
            elif self.concept_loss_type == "per_concept_mse":
                # Note: per_concept_mse is mathematically equivalent to standard MSE
                # with reduction='mean'. Retained for potential future per-concept weighting.
                per_concept = []
                for i in range(concepts_pred.size(1)):
                    mse_i = F.mse_loss(
                        concepts_pred[:, i], concept_targets[:, i]
                    )
                    per_concept.append(mse_i)
                L_concept = torch.stack(per_concept).mean()

        L_total = self.alpha * L_concept + self.beta * L_cls

        return {
            "total_loss": L_total,
            "concept_loss": L_concept.detach(),
            "classification_loss": L_cls.detach(),
        }


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    strategy: str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.

    Strategies:
        "inverse_freq":  w_k = N_total / (K * N_k)
        "sqrt_inverse":  w_k = sqrt(N_total / (K * N_k))
        "effective_num": Cui et al. (CVPR 2019) effective sample number method

    Args:
        labels:      All training labels (1-D tensor of class indices)
        num_classes: Number of classes
        strategy:    Weight computation strategy

    Returns:
        Class weight tensor of shape (num_classes,), mean-normalized to 1.
    """
    class_counts = torch.bincount(labels, minlength=num_classes).float()
    class_counts = class_counts.clamp(min=1)  # Guard against division by zero
    N_total = labels.size(0)

    if strategy == "inverse_freq":
        weights = N_total / (num_classes * class_counts)
    elif strategy == "sqrt_inverse":
        weights = torch.sqrt(N_total / (num_classes * class_counts))
    elif strategy == "effective_num":
        beta = 0.999
        effective = (1.0 - beta ** class_counts) / (1.0 - beta)
        weights = 1.0 / effective
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(
            f"Unknown strategy: '{strategy}'. "
            f"Choose from: 'inverse_freq', 'sqrt_inverse', 'effective_num'."
        )

    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


class CBMCalibrationLoss(CBMJointLoss):
    """
    Plan B specialized loss -- supports samples with partial concept ground truth.

    For samples with OGLE cross-match: L = alpha * L_concept + beta * L_cls
    For samples without cross-match:   L = beta * L_cls

    Ground truth sources:
        Strategy 1 (recommended): OGLE long-baseline high-precision extraction values
                                   (same-source cross-matching)
        Strategy 2:               High-confidence subset self-regression
        Strategy 3:               Input as GT (degenerates to autoencoder, ablation baseline)
    """

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        has_concept_gt: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute calibration loss with optional concept GT mask.

        Args:
            model_output:    Model forward() return dict
            targets:         Classification labels, shape (batch_size,)
            concept_targets: Concept ground truth, shape (batch_size, 12)
            has_concept_gt:  Boolean mask, shape (batch_size,).
                             True = this sample has OGLE ground truth.
            **kwargs:        Ignored additional keyword arguments.

        Returns:
            dict with keys:
                "total_loss", "concept_loss", "classification_loss"
        """
        logits = model_output["logits"]
        L_cls = self.classification_loss(logits, targets)

        L_concept = torch.tensor(0.0, device=logits.device)
        if concept_targets is not None and has_concept_gt is not None:
            mask = has_concept_gt.bool()
            if mask.any():
                pred_masked = model_output["concepts"][mask]
                gt_masked = concept_targets[mask]
                L_concept = F.mse_loss(pred_masked, gt_masked)

        L_total = self.alpha * L_concept + self.beta * L_cls

        return {
            "total_loss": L_total,
            "concept_loss": L_concept.detach(),
            "classification_loss": L_cls.detach(),
        }


class CBMSequentialLoss(nn.Module):
    """
    Sequential training loss -- two-stage training:

    Stage 1: Train concept calibration layer, freeze label predictor.
        L = MSE(c_pred, c_gt)

    Stage 2: Train label predictor, freeze concept calibration layer.
        L = CrossEntropy(logits, labels)

    [Fix M1] Only for Plan B (HardCBM_Calibrated). Plan A has no separate
    concept prediction layer and does not support sequential training.

    Inherits nn.Module so it can be used with Trainer.forward() interface.
    Use set_stage(1) or set_stage(2) to switch between training stages.

    Args:
        class_weights:   Class weight tensor (optional)
        label_smoothing: Label smoothing coefficient
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.concept_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.stage = 1  # Mutable state: 1 = concept stage, 2 = classification stage

    def set_stage(self, stage: int) -> None:
        """Switch between training stages (1 = concept, 2 = classification)."""
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.stage = stage

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the current stage.

        Stage 1: concept loss only (requires concept_targets).
        Stage 2: classification loss only.

        Args:
            model_output:    Model forward() return dict
            targets:         Classification labels, shape (batch_size,), dtype=long
            concept_targets: Concept ground truth, shape (batch_size, 12)
            **kwargs:        Additional keyword arguments (ignored, for signature compat)

        Returns:
            dict with keys: "total_loss", "concept_loss", "classification_loss"
        """
        if self.stage == 1:
            # Stage 1: concept loss only
            if concept_targets is None:
                zero = torch.tensor(0.0, device=targets.device)
                return {"total_loss": zero, "concept_loss": zero,
                        "classification_loss": torch.tensor(0.0, device=targets.device)}
            concepts = model_output["concepts"]
            L_concept = self.concept_loss_fn(concepts, concept_targets)
            return {
                "total_loss": L_concept,
                "concept_loss": L_concept.detach(),
                "classification_loss": torch.tensor(0.0, device=targets.device),
            }
        else:
            # Stage 2: classification loss only
            logits = model_output["logits"]
            L_cls = self.classification_loss_fn(logits, targets)
            return {
                "total_loss": L_cls,
                "concept_loss": torch.tensor(0.0, device=targets.device),
                "classification_loss": L_cls.detach(),
            }

    # --- Legacy compatibility methods (kept for direct stage-specific calls) ---

    def stage1_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        concept_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Stage 1: concept loss only (legacy interface)."""
        L = self.concept_loss_fn(
            model_output["concepts"], concept_targets
        )
        return {
            "total_loss": L,
            "concept_loss": L.detach(),
            "classification_loss": torch.tensor(0.0, device=L.device),
        }

    def stage2_loss(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Stage 2: classification loss only (legacy interface)."""
        L = self.classification_loss_fn(model_output["logits"], targets)
        return {
            "total_loss": L,
            "concept_loss": torch.tensor(0.0, device=L.device),
            "classification_loss": L.detach(),
        }


class CBMIndependentLoss(nn.Module):
    """
    Independent training loss -- concept calibration layer and label predictor
    trained completely independently.

    Training flow:
        1. Concept predictor: input raw features, target "clean" concepts
        2. Label predictor: input concept predictor output (detach!), target class labels

    Key: concepts.detach() cuts gradient propagation. In true independent mode,
    the model should be called with detached concepts for the label predictor.
    Here we approximate by summing both losses; the classification loss gradient
    only flows through the label predictor weights.

    [Fix M1] Only for Plan B (HardCBM_Calibrated).

    Inherits nn.Module so it can be used with Trainer.forward() interface.

    Args:
        class_weights:   Class weight tensor (optional)
        label_smoothing: Label smoothing coefficient
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.concept_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute independent training loss.

        Concept loss uses normal gradient flow. Classification loss is computed
        on logits (the label predictor should receive detached concepts for true
        independent training).

        Args:
            model_output:    Model forward() return dict
            targets:         Classification labels, shape (batch_size,), dtype=long
            concept_targets: Concept ground truth, shape (batch_size, 12)
            **kwargs:        Additional keyword arguments (ignored, for signature compat)

        Returns:
            dict with keys: "total_loss", "concept_loss", "classification_loss"
        """
        concepts = model_output["concepts"]

        # [C2 FIX] Use logits computed from detached concepts so that
        # classification loss gradients only flow through the label predictor,
        # not back through the concept calibrators.  This makes "independent"
        # training truly independent (concept predictor and label predictor
        # receive non-overlapping gradient signals).
        # Falls back to normal logits for models that don't provide the key.
        logits = model_output.get(
            "logits_detached_concepts", model_output["logits"]
        )

        # Concept loss (normal gradient through concept predictor)
        L_concept = torch.tensor(0.0, device=targets.device)
        if concept_targets is not None:
            L_concept = self.concept_loss_fn(concepts, concept_targets)

        # Classification loss (gradient only through label predictor)
        L_cls = self.classification_loss_fn(logits, targets)

        return {
            "total_loss": L_concept + L_cls,
            "concept_loss": L_concept.detach(),
            "classification_loss": L_cls.detach(),
        }

    # --- Legacy compatibility methods ---

    def concept_loss(
        self,
        concepts_pred: torch.Tensor,
        concept_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and target concepts (legacy interface)."""
        return self.concept_loss_fn(concepts_pred, concept_targets)

    def classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss on detached concept representations (legacy interface)."""
        return self.classification_loss_fn(logits, targets)
