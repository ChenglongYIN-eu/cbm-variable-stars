"""
Concept Embedding Model (CEM) variant (ablation study).

Reference: Espinosa Zarlenga et al., NeurIPS 2022.

Each concept is represented by two embedding vectors:
    - c_pos: embedding for "concept value high"
    - c_neg: embedding for "concept value low"
    - concept activation = similarity with c_pos vs c_neg
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES, NUM_CONCEPTS, NUM_CLASSES,
)


class ConceptEmbeddingModel(nn.Module):
    """
    Concept Embedding Model (Espinosa Zarlenga et al., NeurIPS 2022).

    Core idea: each concept is represented by two embedding vectors:
        - c_pos: embedding for "concept value high"
        - c_neg: embedding for "concept value low"
        - Concept activation = similarity with c_pos vs c_neg

    Adaptation for this project:
        The original CEM is designed for binary concepts. Our concepts are
        continuous values. Adaptation: encode continuous concept values as
        soft labels, then use the CEM framework.

    Architecture:
        Input (12-dim) -> 12 per-concept encoders -> (batch, 12, embed_dim)
        -> Compute sim_pos and sim_neg per concept
        -> concept_probs = sigmoid(sim_pos - sim_neg)   (batch, 12)
        -> mixed = prob * c_pos + (1-prob) * c_neg      (batch, 12, embed_dim)
        -> bottleneck = flatten(mixed)                  (batch, 96)
        -> Label predictor MLP -> 6-class output

    Args:
        num_concepts:      Number of concepts
        concept_embed_dim: Concept embedding dimension (default 8)
        num_classes:       Number of classes
        hidden_dims:       Label predictor hidden dimensions
        dropout_rate:      Dropout rate
    """

    def __init__(
        self,
        num_concepts: int = NUM_CONCEPTS,
        concept_embed_dim: int = 8,
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
        self.concept_names = CONCEPT_NAMES

        # Per-concept encoders (Zarlenga et al. 2022)
        # [M1 NOTE] Each encoder receives the FULL input (all 12 concepts),
        # matching the original CEM paper design. This means information can
        # leak across concepts, so concept encoders are NOT independent in the
        # strict sense. SoftCBM uses Linear(1, ...) per concept for true
        # independence; CEM's design trades independence for richer embeddings.
        self.concept_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_concepts, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, concept_embed_dim),
            ) for _ in range(num_concepts)
        ])

        # Positive / negative embedding prototypes per concept
        self.concept_pos = nn.Parameter(
            torch.randn(num_concepts, concept_embed_dim) * 0.1
        )
        self.concept_neg = nn.Parameter(
            torch.randn(num_concepts, concept_embed_dim) * 0.1
        )

        # Label predictor
        bottleneck_dim = num_concepts * concept_embed_dim
        layers: List[nn.Module] = []
        in_dim = bottleneck_dim
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
        concept_prob_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Intervention semantics (Fix #16):
            CEM supports two levels of intervention:

            1. Input-level intervention (concept_override):
               Replaces raw input concept values BEFORE encoding. The overridden
               values are re-encoded by the per-concept encoders, so the effect
               propagates through the full CEM pipeline. This is analogous to
               "correcting an astronomer's raw measurement."

            2. Bottleneck-level intervention (concept_prob_override):
               Replaces concept activation probabilities AFTER encoding but
               BEFORE computing the mixed embedding. This directly controls the
               concept probability layer (the CEM bottleneck), which is the
               interpretable layer analogous to HardCBM's concept values.
               This is analogous to "directly correcting the model's concept
               belief" and is the more meaningful intervention for CEM.

            Both can be used simultaneously: concept_override is applied first
            (at input), then concept_prob_override (at bottleneck).

        Args:
            x: Input feature tensor, shape (batch_size, 12).
            concept_override: Optional concept intervention tensor,
               shape (batch_size, 12). Applied at the input level before
               encoding. NaN positions retain original values.
            concept_prob_override: Optional concept probability intervention
               tensor, shape (batch_size, 12). Applied at the concept_probs
               level after encoding. NaN positions retain computed values.
               Values should be in [0, 1] range (probabilities).

        Returns:
            dict with keys:
                "concepts":       input concept values (batch, 12), after any
                                  input-level override
                "concept_probs":  concept activation probabilities (batch, 12),
                                  after any prob-level override
                "bottleneck":     mixed embeddings flattened (batch, num_concepts * embed_dim).
                                  This is the full representation passed to the label
                                  predictor: prob * c_pos + (1-prob) * c_neg for each
                                  concept, concatenated. Useful for visualization,
                                  probing, or bottleneck-level analysis.
                "logits":         (batch, 6) raw classification logits
                "probabilities":  (batch, 6) softmax classification probabilities
        """
        # Apply concept override at input level if provided
        x_input = x
        if concept_override is not None:
            mask = ~torch.isnan(concept_override)
            x_input = torch.where(mask, concept_override, x)

        batch_size = x_input.size(0)

        # Per-concept encoding (each encoder independently maps input to concept embedding)
        encoded_list = [enc(x_input) for enc in self.concept_encoders]
        encoded = torch.stack(encoded_list, dim=1)  # (batch, num_concepts, embed_dim)

        # Compute similarity with positive / negative prototypes
        sim_pos = (encoded * self.concept_pos.unsqueeze(0)).sum(dim=-1)  # (batch, 12)
        sim_neg = (encoded * self.concept_neg.unsqueeze(0)).sum(dim=-1)  # (batch, 12)

        # Concept probabilities
        concept_probs = torch.sigmoid(sim_pos - sim_neg)  # (batch, 12)

        # Apply concept probability override at bottleneck level if provided
        if concept_prob_override is not None:
            mask = ~torch.isnan(concept_prob_override)
            concept_probs = torch.where(mask, concept_prob_override, concept_probs)

        # Mixed embedding: prob * c_pos + (1-prob) * c_neg
        prob_expanded = concept_probs.unsqueeze(-1)  # (batch, 12, 1)
        mixed = (
            prob_expanded * self.concept_pos.unsqueeze(0)
            + (1 - prob_expanded) * self.concept_neg.unsqueeze(0)
        )  # (batch, 12, 8)

        bottleneck = mixed.view(batch_size, -1)  # (batch, 96)

        # Label prediction
        logits = self.label_predictor(bottleneck)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            "concepts": x_input,
            "concept_probs": concept_probs,
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
        Intervene on a single concept at the input level (convenience interface).

        Overrides the specified concept value before encoding. For bottleneck-level
        intervention (concept_prob_override), use forward() directly.

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
