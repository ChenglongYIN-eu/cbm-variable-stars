"""
Soft CBM variant (ablation study).

Each concept embedder sees only 1 dimension of input (Fix M2).
Bottleneck dim = 12 * embed_dim = 48.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES, NUM_CONCEPTS, NUM_CLASSES,
)


class SoftCBM(nn.Module):
    """
    Soft CBM variant -- concept layer as continuous embeddings rather than hard predictions.

    Key difference from Hard CBM:
        - Concept layer is not forced to 12 dims; each concept has a high-dim
          embedding (e.g., 4-dim), total bottleneck = 12 * embed_dim = 48
        - Interpretability weaker than Hard CBM (embedding dims don't directly
          correspond to physical quantities)
        - Theoretically better accuracy than Hard CBM (wider information bottleneck)
        - Used for ablation experiments: quantifying accuracy cost of Hard CBM

    [Fix M2] Unified definition: each concept encoder sees only its corresponding
    1-dim input, preserving concept independence. This aligns with the core CBM
    design philosophy: concepts should not share information during encoding,
    otherwise the interpretability of the concept layer is undermined.

    Architecture per concept embedder: 1 -> 8 -> 4 (each concept independent)

    Parameter count:
        - Concept embeddings: 12 * (1*8 + 8 + 8*4 + 4) = 12 * 52 = 624
        - Label predictor: Linear(48,64)=3,136 + BN(64)=128 + Linear(64,32)=2,080
                           + BN(32)=64 + Linear(32,6)=198 = 5,606
        - Total: 6,230 parameters

    Args:
        num_concepts:      Number of concepts
        concept_embed_dim: Embedding dimension per concept (default 4)
        num_classes:       Number of classes
        hidden_dims:       Label predictor hidden dimensions
        dropout_rate:      Dropout rate
    """

    def __init__(
        self,
        num_concepts: int = NUM_CONCEPTS,
        concept_embed_dim: int = 4,
        num_classes: int = NUM_CLASSES,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.concept_embed_dim = concept_embed_dim
        self.bottleneck_dim = num_concepts * concept_embed_dim  # 48
        self.concept_names = CONCEPT_NAMES

        # ===== Concept Embedding Layer =====
        # [Fix M2] Each concept encoder receives only its corresponding 1-dim input
        # Input: (batch, 12) -> split into 12 (batch, 1) -> embed each -> concatenate
        # Architecture: 1 -> 8 -> 4 (each concept independent)
        self.concept_embedders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, concept_embed_dim * 2),   # 1 -> 8
                nn.ReLU(inplace=True),
                nn.Linear(concept_embed_dim * 2, concept_embed_dim),  # 8 -> 4
            )
            for _ in range(num_concepts)
        ])

        # ===== Label Predictor =====
        layers: List[nn.Module] = []
        in_dim = self.bottleneck_dim  # 48
        for h_dim in hidden_dims:
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

        Intervention semantics (Fix #16):
            SoftCBM intervention operates at the **input concept level** (before
            embedding), NOT at the bottleneck embedding level. This means:
            - concept_override replaces raw concept values (z-score space) before
              they are passed through the per-concept embedders.
            - This is an input-level intervention: the embedders will re-encode
              the overridden values, so the effect propagates through the learned
              embedding transformation.
            - For bottleneck-level intervention (directly replacing embeddings),
              users would need to modify the bottleneck tensor manually.
            This design is intentional for API uniformity with HardCBM, but users
            should be aware that intervention effects differ from HardCBM where
            concepts ARE the bottleneck.

        Args:
            x: (batch, 12) standardized concept features.
            concept_override: (batch, 12) concept intervention values; NaN retains
                original value. Intervention is applied at the input level before
                embedding (see intervention semantics above).

        Returns:
            dict with keys:
                "concepts":      (batch, 12) concept values (after any override)
                "bottleneck":    (batch, 48) post-embedding bottleneck representation.
                                 This is the concatenation of all per-concept embeddings
                                 and serves as the full information passed to the label
                                 predictor. Useful for visualization, probing, or
                                 bottleneck-level analysis.
                "logits":        (batch, 6) raw classification logits
                "probabilities": (batch, 6) softmax classification probabilities
        """
        concepts = x

        if concept_override is not None:
            mask = ~torch.isnan(concept_override)
            concepts = torch.where(mask, concept_override, concepts)

        # [Fix M2] Each embedder receives only its corresponding 1-dim input
        embeddings = []
        for i, embedder in enumerate(self.concept_embedders):
            c_i = concepts[:, i:i + 1]  # (batch, 1) -- only the i-th concept
            e_i = embedder(c_i)          # (batch, embed_dim)
            embeddings.append(e_i)

        bottleneck = torch.cat(embeddings, dim=1)  # (batch, 48)

        # Label prediction
        logits = self.label_predictor(bottleneck)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            "concepts": concepts,
            "bottleneck": bottleneck,
            "logits": logits,
            "probabilities": probabilities,
        }

    def intervene(
        self,
        x: torch.Tensor,
        concept_idx: int,
        new_value: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Intervene on a single concept (convenience interface).

        Overrides the specified concept at the input level before embedding.
        See forward() docstring for intervention semantics.

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
