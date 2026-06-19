"""
Phase-folded light curve dataset for end-to-end CBM training.

Synthesizes multi-channel phase-folded light curves:
  Channel 0: Fourier-synthesized magnitude curve (100 bins)
  Channel 1: log10(period) broadcast to 100 bins
  Channel 2: color_bp_rp broadcast to 100 bins
  Channel 3: mean_mag broadcast to 100 bins

This multi-channel approach mirrors real astronomy: alongside the light curve,
catalog-level metadata (period, color, brightness) is always available.
The concept encoder must learn to extract Fourier parameters (R21, R31, phi21),
statistical moments (skewness, kurtosis), and variability indices from the
curve shape -- a genuinely non-trivial mapping.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12, LABEL_MAP


# Feature indices in CONCEPT_NAMES_12
IDX_PERIOD = 0
IDX_AMPLITUDE = 1
IDX_RISE_FRACTION = 2
IDX_R21 = 3
IDX_R31 = 4
IDX_PHI21 = 5
IDX_COLOR_BP_RP = 10
IDX_MEAN_MAG = 11

N_CHANNELS = 4  # magnitude curve + 3 catalog scalars


class LightCurveDataset(Dataset):
    """
    Variable star dataset with synthesized multi-channel phase-folded light curves.

    Returns (N_CHANNELS, N_BINS) tensors where:
      Channel 0: Phase-folded magnitude curve (Fourier synthesis + noise)
      Channel 1: log10(period) repeated across bins
      Channel 2: color_bp_rp repeated across bins
      Channel 3: mean_mag repeated across bins

    Args:
        raw_features:  Raw physical features, shape (N, 12), physical units.
        labels:        Integer-encoded labels, shape (N,).
        concept_gt:    Standardized concept ground truth, shape (N, 12).
        n_bins:        Number of phase bins (default 100).
        noise_level:   Noise as fraction of amplitude (default 0.03).
        augment:       Whether to add noise (True for train, False for eval).
    """

    def __init__(
        self,
        raw_features: np.ndarray,
        labels: np.ndarray,
        concept_gt: np.ndarray,
        n_bins: int = 100,
        noise_level: float = 0.03,
        augment: bool = True,
    ) -> None:
        assert raw_features.shape[0] == labels.shape[0] == concept_gt.shape[0]
        assert raw_features.shape[1] == 12
        assert concept_gt.shape[1] == 12

        self.raw_features = raw_features.astype(np.float32)
        self.concept_gt = torch.tensor(concept_gt, dtype=torch.float32)
        self.n_bins = n_bins
        self.noise_level = noise_level
        self.augment = augment

        # Phase grid (fixed)
        self.phi = np.linspace(0, 1, n_bins, endpoint=False).astype(np.float32)
        self.two_pi = 2.0 * np.pi

        # Label encoding
        if labels.dtype.kind in ("U", "S", "O"):
            self.labels = torch.tensor(
                [LABEL_MAP.get(str(l), -1) for l in labels], dtype=torch.long
            )
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def _synthesize_multichannel(self, raw: np.ndarray) -> np.ndarray:
        """
        Synthesize a multi-channel phase-folded representation.

        Channel 0: Fourier-synthesized magnitude curve
        Channel 1: log10(period) broadcast
        Channel 2: color_bp_rp broadcast
        Channel 3: mean_mag broadcast

        Args:
            raw: 12-dim raw feature vector.

        Returns:
            Multi-channel array of shape (N_CHANNELS, n_bins).
        """
        period = raw[IDX_PERIOD]
        amplitude = raw[IDX_AMPLITUDE]
        rise_fraction = raw[IDX_RISE_FRACTION]
        R21 = raw[IDX_R21]
        R31 = raw[IDX_R31]
        phi21 = raw[IDX_PHI21]
        color_bp_rp = raw[IDX_COLOR_BP_RP]
        mean_mag = raw[IDX_MEAN_MAG]

        phi = self.phi

        # --- Channel 0: Fourier-synthesized magnitude curve ---
        # [M2 FIX] Use cos() base function to match the Fourier decomposition
        # convention in fourier.py: m(t) = m0 + Σ A_k cos(2πk·φ + φ_k)
        # With φ_1 = 0 (arbitrary reference), φ_2 = φ_21, φ_3 = 0 (φ_31 unavailable)
        A1 = amplitude / 2.0
        A2 = R21 * A1
        A3 = R31 * A1

        curve = (
            mean_mag
            + A1 * np.cos(self.two_pi * phi)
            + A2 * np.cos(2 * self.two_pi * phi + phi21)
            + A3 * np.cos(3 * self.two_pi * phi)
        )

        # Asymmetry from rise_fraction
        if 0.1 < rise_fraction < 0.9:
            asym_strength = 0.15 * A1 * abs(rise_fraction - 0.5)
            sawtooth = np.where(
                phi < rise_fraction,
                phi / rise_fraction,
                1.0 - (phi - rise_fraction) / (1.0 - rise_fraction),
            )
            curve += asym_strength * (sawtooth - 0.5)

        # Noise augmentation
        if self.augment:
            noise_std = self.noise_level * max(amplitude, 0.01)
            curve = curve + np.random.normal(0, noise_std, self.n_bins).astype(
                np.float32
            )

        # --- Channel 1: log10(period) ---
        log_period = np.full(self.n_bins, np.log10(max(period, 1e-3)), dtype=np.float32)

        # --- Channel 2: color_bp_rp ---
        color_channel = np.full(self.n_bins, color_bp_rp, dtype=np.float32)

        # --- Channel 3: mean_mag ---
        mag_channel = np.full(self.n_bins, mean_mag, dtype=np.float32)

        return np.stack([curve, log_period, color_channel, mag_channel], axis=0)  # (4, n_bins)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw = self.raw_features[idx]
        multichannel = self._synthesize_multichannel(raw)  # (4, n_bins)

        return {
            "features": torch.tensor(multichannel, dtype=torch.float32),  # (4, n_bins)
            "concept_gt": self.concept_gt[idx],  # (12,)
            "label": self.labels[idx],
        }
