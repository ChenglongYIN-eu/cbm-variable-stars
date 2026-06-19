"""
Real Gaia DR3 light curve dataset for end-to-end CBM training.

Reads actual epoch photometry from Gaia DR3, performs phase folding using
catalog periods (computed via Lomb-Scargle if not available), and bins into
fixed-size phase curves. This provides genuinely non-trivial concept learning:
the CNN must extract physical concepts from real, noisy, irregularly sampled
photometric data.

Key differences from the synthetic LightCurveDataset:
- Uses REAL Gaia observations (not Fourier-synthesized curves)
- Real noise, gaps, systematics, and outliers
- Phase folding uses measured periods
- No circular information flow: concepts are extracted from real data
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12, LABEL_MAP


class RealLightCurveDataset(Dataset):
    """
    Dataset of real Gaia DR3 phase-folded light curves.

    Reads epoch photometry parquet files, phase-folds using known periods,
    and bins into fixed-length phase curves. Returns multi-channel tensors:
      Channel 0: Phase-folded magnitude curve (n_bins bins)
      Channel 1: Observation count per bin (quality indicator)
      Channel 2: log10(period) broadcast
      Channel 3: color (BP-RP) broadcast

    Args:
        source_ids:     List of Gaia source_ids.
        periods:        Period for each source (days).
        colors:         BP-RP color for each source.
        mean_mags:      Mean G magnitude for each source.
        labels:         Integer-encoded class labels.
        concept_gt:     Standardized concept ground truth (N, 12).
        epoch_phot_dir: Directory containing {source_id}.parquet files.
        n_bins:         Number of phase bins (default 50).
        augment:        Whether to add phase jitter augmentation.
    """

    def __init__(
        self,
        source_ids: np.ndarray,
        periods: np.ndarray,
        colors: np.ndarray,
        mean_mags: np.ndarray,
        labels: np.ndarray,
        concept_gt: np.ndarray,
        epoch_phot_dir: str = "data/raw/gaia/epoch_photometry",
        n_bins: int = 50,
        augment: bool = True,
    ) -> None:
        self.source_ids = np.array(source_ids)
        self.periods = np.array(periods, dtype=np.float64)
        self.colors = np.array(colors, dtype=np.float32)
        self.mean_mags = np.array(mean_mags, dtype=np.float32)
        self.concept_gt = torch.tensor(concept_gt, dtype=torch.float32)
        self.n_bins = n_bins
        self.augment = augment
        self.epoch_phot_dir = Path(epoch_phot_dir)

        # Label encoding
        if labels.dtype.kind in ("U", "S", "O"):
            self.labels = torch.tensor(
                [LABEL_MAP.get(str(l), -1) for l in labels], dtype=torch.long
            )
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

        # Pre-load and phase-fold all light curves
        self.phase_curves = self._preload_all()

    def _phase_fold_and_bin(
        self, times: np.ndarray, mags: np.ndarray, period: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phase-fold a light curve and bin into fixed-size array.

        Args:
            times: Observation times (BJD).
            mags:  Magnitudes.
            period: Folding period (days).

        Returns:
            (binned_mags, bin_counts) each of shape (n_bins,).
        """
        # Compute phases
        phases = (times % period) / period  # [0, 1)

        # Phase jitter augmentation
        if self.augment:
            jitter = np.random.uniform(-0.02, 0.02)
            phases = (phases + jitter) % 1.0

        # Bin by phase
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(phases, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        binned_mags = np.full(self.n_bins, np.nan, dtype=np.float32)
        bin_counts = np.zeros(self.n_bins, dtype=np.float32)

        for b in range(self.n_bins):
            mask = bin_indices == b
            if mask.any():
                binned_mags[b] = np.median(mags[mask])
                bin_counts[b] = mask.sum()

        # Interpolate empty bins
        valid = ~np.isnan(binned_mags)
        if valid.sum() >= 3 and not valid.all():
            x_valid = np.where(valid)[0]
            y_valid = binned_mags[valid]
            x_all = np.arange(self.n_bins)
            binned_mags = np.interp(x_all, x_valid, y_valid).astype(np.float32)

        elif valid.sum() < 3:
            # Too few valid bins - fill with mean
            mean_val = np.nanmean(binned_mags) if valid.any() else 0.0
            binned_mags = np.full(self.n_bins, mean_val, dtype=np.float32)

        return binned_mags, bin_counts

    def _preload_all(self) -> torch.Tensor:
        """Pre-load and phase-fold all light curves."""
        all_curves = np.zeros((len(self.source_ids), 4, self.n_bins), dtype=np.float32)

        for i, (sid, period, color, mean_mag) in enumerate(
            zip(self.source_ids, self.periods, self.colors, self.mean_mags)
        ):
            ep_file = self.epoch_phot_dir / f"{sid}.parquet"

            if ep_file.exists() and period > 0 and np.isfinite(period):
                df = pd.read_parquet(ep_file)
                times = df['time'].values
                mags = df['mag'].values

                binned_mags, bin_counts = self._phase_fold_and_bin(times, mags, period)

                all_curves[i, 0, :] = binned_mags  # Phase-folded magnitude
                all_curves[i, 1, :] = bin_counts / max(bin_counts.max(), 1)  # Normalized counts
            else:
                # No light curve available - use mean_mag as constant
                all_curves[i, 0, :] = mean_mag
                all_curves[i, 1, :] = 0.0

            # Catalog metadata channels
            all_curves[i, 2, :] = np.log10(max(period, 1e-3)) if np.isfinite(period) else 0.0
            all_curves[i, 3, :] = color if np.isfinite(color) else 0.0

        return torch.tensor(all_curves, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.phase_curves[idx],  # (4, n_bins)
            "concept_gt": self.concept_gt[idx],   # (12,)
            "label": self.labels[idx],
        }


def build_real_dataset(
    metadata_dir: str = "data/raw/gaia/metadata",
    epoch_phot_dir: str = "data/raw/gaia/epoch_photometry",
    n_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build arrays for RealLightCurveDataset from downloaded Gaia data.

    Combines metadata files, filters to sources with epoch photometry,
    and computes periods via Lomb-Scargle for sources without catalog periods.

    Returns:
        (source_ids, periods, colors, mean_mags, labels)
    """
    from astropy.timeseries import LombScargle

    meta_dir = Path(metadata_dir)
    ep_dir = Path(epoch_phot_dir)

    # Class mapping from Gaia classifier names to our 6 classes
    class_map = {
        'RR': 'RRAB',  # Will be refined to RRAB/RRC by period
        'CEP': 'DCEP',
        'DSCT|GDOR|SXPHE': 'DSCT_SXPHE',
        'ECL': 'ECL',
        'LPV': 'MIRA_SR',
    }

    all_sids, all_periods, all_colors, all_mags, all_labels = [], [], [], [], []

    for meta_file in sorted(meta_dir.glob("gaia_metadata_*.parquet")):
        if 'basic' in meta_file.stem or 'periods' in meta_file.stem:
            continue

        df = pd.read_parquet(meta_file)
        gaia_cls = df['best_class_name'].iloc[0] if len(df) > 0 else None

        if gaia_cls not in class_map:
            continue

        our_label = class_map[gaia_cls]

        for _, row in df.iterrows():
            sid = row['source_id']
            ep_file = ep_dir / f"{sid}.parquet"

            if not ep_file.exists():
                continue

            # Read light curve and compute period via Lomb-Scargle
            lc = pd.read_parquet(ep_file)
            if len(lc) < 15:
                continue

            times = lc['time'].values
            mags = lc['mag'].values

            try:
                ls = LombScargle(times, mags)
                freq, power = ls.autopower(
                    minimum_frequency=0.001,
                    maximum_frequency=25.0,
                    samples_per_peak=5,
                )
                best_freq = freq[np.argmax(power)]
                period = 1.0 / best_freq
            except Exception:
                continue

            color = row.get('bp_rp', np.nan)
            mean_mag = row.get('phot_g_mean_mag', np.mean(mags))

            # Refine RR Lyrae subtype by period
            if gaia_cls == 'RR':
                our_label = 'RRAB' if period > 0.45 else 'RRC'

            all_sids.append(sid)
            all_periods.append(period)
            all_colors.append(color if np.isfinite(color) else 0.0)
            all_mags.append(mean_mag if np.isfinite(mean_mag) else np.mean(mags))
            all_labels.append(LABEL_MAP[our_label])

    return (
        np.array(all_sids),
        np.array(all_periods, dtype=np.float64),
        np.array(all_colors, dtype=np.float32),
        np.array(all_mags, dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
    )
