"""
1D-CNN concept encoder for end-to-end CBM.

Maps multi-channel phase-folded representations to concept predictions.

Input format (batch, n_channels, n_bins):
    Channel 0: Phase-folded magnitude curve (100 bins)
    Channel 1: log10(period) broadcast to 100 bins
    Channel 2: color_bp_rp broadcast to 100 bins
    Channel 3: mean_mag broadcast to 100 bins

This mirrors real astronomy: alongside the light curve, catalog-level
metadata (period, color, brightness) is always available.  The encoder
must learn to extract Fourier parameters, statistical moments, and
variability indices from the curve shape -- a non-trivial mapping.

Architecture:
    Conv1d(4, 32, k=7) -> BN -> ReLU
    Conv1d(32, 64, k=5) -> BN -> ReLU
    Conv1d(64, 128, k=3) -> BN -> ReLU
    AdaptiveAvgPool1d(1) -> Flatten
    Linear(128, num_concepts)
"""

import torch
import torch.nn as nn
from typing import Optional

from cbm_variable_stars.shared.constants import NUM_CONCEPTS


class PhaseCurveEncoder(nn.Module):
    """
    1D-CNN that encodes multi-channel phase-folded representations
    into concept predictions.

    Input:  (batch, n_input_channels, n_bins)
    Output: (batch, num_concepts)

    Args:
        n_bins:            Number of phase bins (default 100).
        num_concepts:      Number of output concepts (default 12).
        n_input_channels:  Number of input channels (default 4).
        channels:          Channel progression for conv layers (default [32, 64, 128]).
        kernel_sizes:      Kernel sizes for conv layers (default [7, 5, 3]).
        dropout_rate:      Dropout before final linear (default 0.2).
    """

    def __init__(
        self,
        n_bins: int = 100,
        num_concepts: int = NUM_CONCEPTS,
        n_input_channels: int = 4,
        channels: Optional[list] = None,
        kernel_sizes: Optional[list] = None,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        assert len(channels) == len(kernel_sizes)

        self.num_concepts = num_concepts

        # Build convolutional layers
        layers = []
        in_ch = n_input_channels
        for out_ch, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(channels[-1], num_concepts)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights: Kaiming for conv/hidden, Xavier for output."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Multi-channel input, shape (batch, n_input_channels, n_bins).

        Returns:
            Predicted concepts, shape (batch, num_concepts).
        """
        h = self.conv_layers(x)      # (batch, 128, n_bins)
        h = self.pool(h)             # (batch, 128, 1)
        h = h.squeeze(-1)            # (batch, 128)
        h = self.dropout(h)          # (batch, 128)
        concepts = self.fc(h)        # (batch, num_concepts)
        return concepts

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
