"""
Standard MLP baseline (no concept bottleneck).

Architecture: 12 -> 128 -> 64 -> 6, ~10K params.
Follows the same forward interface (returns dict) as CBM models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from cbm_variable_stars.shared.constants import NUM_CONCEPTS, NUM_CLASSES


class BaselineMLP(nn.Module):
    """
    Standard MLP baseline -- no concept bottleneck layer.

    Architecture: 12 -> 128 -> 64 -> 6

    Design rationale:
        - 128-dim hidden layer is much wider than CBM's 12-dim bottleneck,
          validating the accuracy cost of the information bottleneck
        - Uses the same training pipeline as CBM for fair comparison
        - ~10K total parameters, about 3x Hard CBM (3K)

    Note: BaselineMLP follows the unified forward interface (returns dict),
    but the "concepts" key returns the input itself (no bottleneck semantics).
    concept_override has no effect (no bottleneck layer to intervene on).

    Args:
        input_dim:    Input dimension (default 12)
        num_classes:  Number of classes (default 6)
        hidden_dims:  Hidden layer dimensions (default [128, 64])
        dropout_rate: Dropout rate (default 0.3)
    """

    def __init__(
        self,
        input_dim: int = NUM_CONCEPTS,
        num_classes: int = NUM_CLASSES,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.num_classes = num_classes
        self.network = nn.Sequential(*layers)
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
        Unified forward interface.

        concept_override has no effect on BaselineMLP (no bottleneck layer).

        Args:
            x: Input feature tensor, shape (batch_size, 12).
            concept_override: Ignored for BaselineMLP.

        Returns:
            dict with keys:
                "concepts":      input x (placeholder, no bottleneck semantics)
                "logits":        (batch_size, 6)
                "probabilities": (batch_size, 6)
        """
        logits = self.network(x)
        return {
            "concepts": x,  # MLP has no bottleneck; return input as placeholder
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
