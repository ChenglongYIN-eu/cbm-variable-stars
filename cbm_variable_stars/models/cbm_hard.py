"""
Hard Concept Bottleneck Model variants.

Three variants:
    - HardCBM:            Plan A  -- input-as-concepts, MLP label predictor [64,32]->6
    - HardCBM_Linear:     Plan A-Linear -- Linear(12,6) only, 78 params
    - HardCBM_Calibrated: Plan B  -- 12 independent calibrator heads + MLP predictor

Output dict keys (unified interface):
    All models return at least:
        "concepts":      concept bottleneck values, shape (batch, 12)
        "logits":        classification logits, shape (batch, 6)
        "probabilities": classification probabilities, shape (batch, 6)

    HardCBM_Calibrated additionally returns:
        "concepts_raw":  raw input before calibration, shape (batch, 12)
    This extra key is intentional: it allows downstream code to compare
    raw vs. calibrated concept values for calibration quality analysis.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES, NUM_CONCEPTS, NUM_CLASSES,
)
from cbm_variable_stars.models.concept_encoder import PhaseCurveEncoder


class HardCBM(nn.Module):
    """
    Hard Concept Bottleneck Model -- Plan A (input-as-concepts).

    Architecture:
        Input (12-dim standardized physical features)
        -> [directly used as bottleneck]
        -> Label predictor MLP [64, 32] -> 6-class output

    All classification information must pass through the 12-dim concept
    bottleneck, ensuring each prediction is fully traceable to a physical
    quantity.

    Assumption: Input data has already been StandardScaler-normalized by
    the data pipeline. This model does NOT apply any additional scaling.

    Args:
        num_concepts:    Number of concepts (default 12)
        num_classes:     Number of classes (default 6)
        hidden_dims:     Hidden dimensions for label predictor (default [64, 32])
        dropout_rate:    Dropout rate for label predictor (default 0.3)
        use_batch_norm:  Whether to use BatchNorm (default True)
    """

    def __init__(
        self,
        num_concepts: int = NUM_CONCEPTS,
        num_classes: int = NUM_CLASSES,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.concept_names = CONCEPT_NAMES

        # ===== Label Predictor =====
        # Architecture: 12 -> 64 -> 32 -> 6
        # Design rationale:
        #   - Two hidden layers provide sufficient nonlinear mapping
        #   - 64->32 decreasing structure progressively compresses representation
        #   - BatchNorm accelerates convergence and provides regularization
        #   - Dropout=0.3 prevents overfitting (~14K samples vs ~3K params)
        #   - No Softmax on output: CrossEntropyLoss includes LogSoftmax internally

        layers: List[nn.Module] = []
        in_dim = num_concepts  # 12

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim

        # Output layer: no activation (raw logits)
        layers.append(nn.Linear(in_dim, num_classes))

        self.label_predictor = nn.Sequential(*layers)

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: Kaiming for hidden layers (ReLU), Xavier for output layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_classes or m.out_features == 1:
                    # Output layer (no subsequent ReLU): Xavier initialization
                    nn.init.xavier_normal_(m.weight)
                else:
                    # Hidden layer (followed by ReLU): Kaiming initialization
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        concept_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input feature tensor, shape (batch_size, 12).
               StandardScaler-normalized physical concept features.
            concept_override: Optional concept intervention tensor,
               shape (batch_size, 12). When not None, used to replace
               bottleneck layer values (simulating astronomer correction).
               Supports partial intervention: NaN positions retain original values.

        Returns:
            dict with keys:
                "concepts":      concept bottleneck values, shape (batch_size, 12)
                "logits":        classification logits, shape (batch_size, 6)
                "probabilities": classification probabilities, shape (batch_size, 6)
        """
        # Bottleneck = input itself (core of Plan A)
        concepts = x  # (batch_size, 12)

        # Concept intervention: supports partial intervention (overrides only non-NaN positions)
        if concept_override is not None:
            mask = ~torch.isnan(concept_override)
            concepts = torch.where(mask, concept_override, concepts)

        # Label prediction
        logits = self.label_predictor(concepts)  # (batch_size, 6)
        probabilities = torch.softmax(logits, dim=-1)  # (batch_size, 6)

        return {
            "concepts": concepts,
            "logits": logits,
            "probabilities": probabilities,
        }

    def get_concept_importance(self) -> torch.Tensor:
        """
        Get per-concept importance via absolute values of the first-layer weights
        in the label predictor.

        Returns:
            Importance matrix of shape (hidden_dims[0], num_concepts)
        """
        first_layer = self.label_predictor[0]  # nn.Linear(12, 64)
        return first_layer.weight.data.abs()  # (64, 12)

    def intervene(
        self,
        x: torch.Tensor,
        concept_idx: int,
        new_value: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Intervene on a single concept (convenience interface).

        Args:
            x:            Raw input, shape (batch_size, 12)
            concept_idx:  Index of concept to intervene on (0-11)
            new_value:    New concept value (normalized, z-score space)

        Returns:
            Forward pass result after intervention.
        """
        override = torch.full_like(x, float("nan"))
        override[:, concept_idx] = new_value
        return self.forward(x, concept_override=override)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HardCBM_Linear(nn.Module):
    """
    Hard CBM Linear variant -- pure linear label predictor.

    Architecture:
        Input (12-dim standardized features)
        -> [directly used as bottleneck]
        -> Linear(12, 6) -> 6-class output

    Core value:
        - Weight matrix W (6x12) can be directly interpreted as
          "each concept's contribution to each class"
        - W[k, j] > 0 means increasing concept j favors class k
        - This is the maximally interpretable CBM variant
        - If accuracy loss < 2%, this is an important paper highlight

    Parameter count: 12*6 + 6 = 78 parameters

    Args:
        num_concepts: Number of concepts
        num_classes:  Number of classes
    """

    def __init__(
        self,
        num_concepts: int = NUM_CONCEPTS,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.concept_names = CONCEPT_NAMES

        # Pure linear label predictor: no hidden layers, no activation, no Dropout
        self.label_predictor = nn.Linear(num_concepts, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.label_predictor.weight)
        nn.init.zeros_(self.label_predictor.bias)

    def forward(
        self,
        x: torch.Tensor,
        concept_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified forward interface."""
        concepts = x

        if concept_override is not None:
            mask = ~torch.isnan(concept_override)
            concepts = torch.where(mask, concept_override, concepts)

        logits = self.label_predictor(concepts)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            "concepts": concepts,
            "logits": logits,
            "probabilities": probabilities,
        }

    def get_concept_class_weights(self) -> torch.Tensor:
        """
        Return concept-class weight matrix for interpretability analysis.

        Returns:
            Weight matrix of shape (num_classes, num_concepts) = (6, 12).
            W[k, j] represents the linear contribution of concept j to class k.
        """
        return self.label_predictor.weight.data  # (6, 12)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def intervene(
        self,
        x: torch.Tensor,
        concept_idx: int,
        new_value: float,
    ) -> Dict[str, torch.Tensor]:
        """Intervene on a single concept (convenience interface)."""
        override = torch.full_like(x, float("nan"))
        override[:, concept_idx] = new_value
        return self.forward(x, concept_override=override)


class HardCBM_Calibrated(nn.Module):
    """
    Hard CBM Plan B -- with concept calibration layer.

    Architecture:
        Input (12-dim noisy standardized features)
        -> 12 independent concept calibration heads
        -> Calibrated concepts (12-dim)
        -> Label predictor MLP
        -> 6-class output

    Concept calibration layer purpose: map noisy features extracted from
    sparse Gaia DR3 light curves to "clean" concept values. Ground truth
    comes from OGLE high-precision extraction or high-confidence subsets.

    Design details:
        - Each concept has an independent calibration head (not shared network),
          because different physical quantities have different noise
          characteristics (e.g., period estimation error patterns differ
          from amplitude estimation).
        - calibrator_input_mode controls each head's input:
          * "cross" (default): Linear(12, hidden) -- all 12 features as input,
            allowing cross-concept denoising but reducing concept independence.
          * "independent": Linear(1, hidden) -- only the corresponding single
            concept dimension as input, preserving strict concept independence.
        - [Fix M3] All concepts use unconstrained (Identity) output:
          Data is already StandardScaler-normalized to z-score space,
          no Sigmoid constraint needed. Forced Sigmoid limits calibration
          layer expressive capacity.

    Intervention experiment value:
        Plan B is the primary architecture for intervention experiments (Fix S5).
        Because of the calibration layer, c_pred != x (calibrated concept !=
        input feature), so intervention (replacing c_pred with c_true) produces
        meaningful performance differences.

    Parameter count (calibrator_input_mode="cross"): ~6,002 parameters
        - 12 calibration heads: 12 * (12*16 + 16 + 16*1 + 1) = 12 * 225 = 2,700
        - Label predictor: same as Plan A = 3,302
        - Total: 6,002

    Parameter count (calibrator_input_mode="independent"): ~3,654 parameters
        - 12 calibration heads: 12 * (1*16 + 16 + 16*1 + 1) = 12 * 49 = 588
        - Label predictor: same as Plan A = 3,302
        - Total: 3,890

    Output dict keys:
        "concepts_raw":  raw input before calibration, shape (batch, 12).
                         This extra key (not present in other models) allows
                         downstream code to compare raw vs. calibrated concept
                         values for calibration quality analysis.
        "concepts":      calibrated concept values, shape (batch, 12)
        "logits":        classification logits, shape (batch, 6)
        "probabilities": classification probabilities, shape (batch, 6)

    Args:
        num_concepts:          Number of concepts
        num_classes:           Number of classes
        calibrator_hidden:     Calibration head hidden dimension
        calibrator_input_mode: "cross" (each head receives all 12 dims) or
                               "independent" (each head receives only its own dim)
        predictor_hidden_dims: Label predictor hidden dimensions
        dropout_rate:          Dropout rate
    """

    def __init__(
        self,
        num_concepts: int = NUM_CONCEPTS,
        num_classes: int = NUM_CLASSES,
        calibrator_hidden: int = 16,
        calibrator_input_mode: str = "cross",
        predictor_hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if calibrator_input_mode not in ("cross", "independent"):
            raise ValueError(
                f"calibrator_input_mode must be 'cross' or 'independent', "
                f"got '{calibrator_input_mode}'"
            )

        if predictor_hidden_dims is None:
            predictor_hidden_dims = [64, 32]

        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.concept_names = CONCEPT_NAMES
        self.calibrator_input_mode = calibrator_input_mode

        # ===== Concept Calibration Layer: 12 independent calibration heads =====
        # calibrator_input_mode controls input dimensionality:
        #   "cross":       Linear(12, hidden) -- cross-concept denoising
        #   "independent": Linear(1, hidden)  -- strict concept independence
        calibrator_in_dim = num_concepts if calibrator_input_mode == "cross" else 1

        self.concept_calibrators = nn.ModuleList()
        for i in range(num_concepts):
            head = nn.Sequential(
                nn.Linear(calibrator_in_dim, calibrator_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(calibrator_hidden, 1),
                # [Fix M3] No activation function (Identity output)
                # Data is already StandardScaler-normalized, concept values
                # are in z-score space, no Sigmoid/tanh value range constraint needed
            )
            self.concept_calibrators.append(head)

        # ===== Label Predictor (same as Plan A) =====
        layers: List[nn.Module] = []
        in_dim = num_concepts
        for h_dim in predictor_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.label_predictor = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: Kaiming for hidden layers (ReLU), Xavier for output layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_classes or m.out_features == 1:
                    # Output layer (no subsequent ReLU): Xavier initialization
                    nn.init.xavier_normal_(m.weight)
                else:
                    # Hidden layer (followed by ReLU): Kaiming initialization
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        concept_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input feature tensor, shape (batch_size, 12).
            concept_override: Optional concept intervention tensor,
               shape (batch_size, 12). NaN positions retain calibrated values.

        Returns:
            dict with keys:
                "concepts_raw":  raw input (batch, 12) -- unique to Calibrated variant,
                                 allows comparing raw vs. calibrated concept values
                "concepts":      calibrated concepts (batch, 12)
                "logits":        (batch, 6)
                "probabilities": (batch, 6)
        """
        # Concept calibration: each calibration head processes independently
        calibrated = []
        if self.calibrator_input_mode == "cross":
            for i, head in enumerate(self.concept_calibrators):
                c_i = head(x).squeeze(-1)  # (batch_size,)
                calibrated.append(c_i)
        else:  # "independent"
            for i, head in enumerate(self.concept_calibrators):
                c_i = head(x[:, i:i + 1]).squeeze(-1)  # (batch_size,)
                calibrated.append(c_i)

        concepts = torch.stack(calibrated, dim=1)  # (batch_size, 12)

        # Concept intervention
        if concept_override is not None:
            mask = ~torch.isnan(concept_override)
            concepts = torch.where(mask, concept_override, concepts)

        # Label prediction
        logits = self.label_predictor(concepts)
        probabilities = torch.softmax(logits, dim=-1)

        # [C2 FIX] Also compute logits from detached concepts for independent
        # training mode.  This allows CBMIndependentLoss to use classification
        # gradients that only flow through the label predictor, not through the
        # concept calibrators, making independent training truly independent.
        logits_detached = self.label_predictor(concepts.detach())

        return {
            "concepts_raw": x,
            "concepts": concepts,
            "logits": logits,
            "logits_detached_concepts": logits_detached,
            "probabilities": probabilities,
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def intervene(
        self,
        x: torch.Tensor,
        concept_idx: int,
        new_value: float,
    ) -> Dict[str, torch.Tensor]:
        """Intervene on a single concept (convenience interface)."""
        override = torch.full_like(x, float("nan"))
        override[:, concept_idx] = new_value
        return self.forward(x, concept_override=override)


# ============================================================
# End-to-End Hard CBM (solves x=c degeneration)
# ============================================================


class EndToEndHardCBM(nn.Module):
    """
    End-to-end Hard CBM with learned concept encoder.

    Architecture:
        Phase-folded light curve (batch, 4, 100)
        -> PhaseCurveEncoder (1D-CNN) -> predicted concepts (batch, 12)
        -> Label predictor MLP [64, 32] -> 6-class output

    This solves the x=c degeneration problem: the concept encoder g is a
    non-trivial 1D-CNN that learns to extract physical concepts from raw
    photometric data, rather than using an identity mapping.

    Args:
        n_bins:           Number of phase bins in input (default 100).
        num_concepts:     Number of concepts (default 12).
        num_classes:      Number of classes (default 6).
        encoder_channels: CNN channel progression (default [32, 64, 128]).
        encoder_kernels:  CNN kernel sizes (default [7, 5, 3]).
        encoder_dropout:  Dropout in encoder (default 0.2).
        hidden_dims:      Label predictor hidden dims (default [64, 32]).
        predictor_dropout: Dropout in label predictor (default 0.3).
        use_batch_norm:   BatchNorm in label predictor (default True).
    """

    def __init__(
        self,
        n_bins: int = 100,
        num_concepts: int = NUM_CONCEPTS,
        num_classes: int = NUM_CLASSES,
        encoder_channels: Optional[List[int]] = None,
        encoder_kernels: Optional[List[int]] = None,
        encoder_dropout: float = 0.2,
        hidden_dims: Optional[List[int]] = None,
        predictor_dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.concept_names = CONCEPT_NAMES

        self.concept_encoder = PhaseCurveEncoder(
            n_bins=n_bins,
            num_concepts=num_concepts,
            n_input_channels=4,  # magnitude curve + period + color + mean_mag
            channels=encoder_channels,
            kernel_sizes=encoder_kernels,
            dropout_rate=encoder_dropout,
        )

        layers: List[nn.Module] = []
        in_dim = num_concepts
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=predictor_dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.label_predictor = nn.Sequential(*layers)

        self._init_predictor_weights()

    def _init_predictor_weights(self) -> None:
        for m in self.label_predictor.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_classes:
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_weights(self) -> None:
        """Re-initialize all weights (called by Trainer.reset)."""
        self.concept_encoder._init_weights()
        self._init_predictor_weights()

    def forward(
        self,
        x: torch.Tensor,
        concept_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Phase-folded light curve, shape (batch, n_input_channels, n_bins).
            concept_override: Optional concept intervention tensor,
               shape (batch, 12). NaN positions retain predicted values.
        """
        concepts = self.concept_encoder(x)  # (batch, 12)

        if concept_override is not None:
            mask = ~torch.isnan(concept_override)
            concepts = torch.where(mask, concept_override, concepts)

        logits = self.label_predictor(concepts)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            "concepts": concepts,
            "logits": logits,
            "probabilities": probabilities,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def intervene(
        self,
        x: torch.Tensor,
        concept_idx: int,
        new_value: float,
    ) -> Dict[str, torch.Tensor]:
        """Intervene on a single concept.

        [M8 FIX] Uses forward() with concept_override instead of computing
        concepts separately first, which previously caused a redundant
        forward pass through the concept encoder with potentially different
        results when dropout is active.
        """
        # Build an override tensor: NaN everywhere except the target concept
        # forward() will encode concepts once and apply the override
        override = torch.full((x.size(0), self.num_concepts), float("nan"),
                              device=x.device)
        override[:, concept_idx] = new_value
        return self.forward(x, concept_override=override)

    def get_concept_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw concept predictions without classification."""
        return self.concept_encoder(x)
