"""
PyTorch Dataset and DataLoader factory for variable star classification.

[Fix S3] Data standardization is handled entirely by the data pipeline
(StandardScaler). This Dataset assumes input data is already standardized
and does NOT apply any additional scaling.

Data pipeline delivery format:
    features_gaia.parquet -> columns: source_id, label, period, ..., mean_mag
    Already StandardScaler-normalized; scaler.pkl saved for inverse transform.
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional

from cbm_variable_stars.shared.constants import LABEL_MAP

logger = logging.getLogger(__name__)


class VariableStarDataset(Dataset):
    """
    Variable star classification dataset.

    [Fix S3] Data standardization is handled entirely by the data pipeline.
    This Dataset assumes input features are already standardized and does
    NOT apply any internal Scaler.

    Args:
        features:        Standardized feature ndarray, shape (N, 12)
        labels:          Label ndarray, shape (N,); int-encoded or string
        concept_gt:      Optional concept ground truth for Plan B, shape (N, 12)
        has_ogle_match:  Optional OGLE cross-match indicator, shape (N,)
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        concept_gt: Optional[np.ndarray] = None,
        has_ogle_match: Optional[np.ndarray] = None,
    ) -> None:
        # Features (already standardized, convert directly to tensor)
        self.features = torch.tensor(features, dtype=torch.float32)

        # Label encoding
        if labels.dtype.kind in ("U", "S", "O"):  # string type
            mapped = []
            unknown = set()
            for l in labels:
                idx = LABEL_MAP.get(str(l))
                if idx is None:
                    unknown.add(str(l))
                    idx = -1
                mapped.append(idx)
            if unknown:
                logger.warning(f"Unknown labels mapped to -1: {unknown}")
            self.labels = torch.tensor(mapped, dtype=torch.long)
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

        # Concept ground truth (Plan B)
        if concept_gt is not None:
            if concept_gt.shape[0] != features.shape[0]:
                raise ValueError(f"concept_gt rows ({concept_gt.shape[0]}) != features rows ({features.shape[0]})")
            if concept_gt.shape[1] != features.shape[1]:
                logger.warning(f"concept_gt cols ({concept_gt.shape[1]}) != features cols ({features.shape[1]})")
            self.concept_gt = torch.tensor(concept_gt, dtype=torch.float32)
        else:
            self.concept_gt = None

        # OGLE cross-match indicator
        if has_ogle_match is not None:
            self.has_ogle_match = torch.tensor(has_ogle_match, dtype=torch.bool)
        else:
            self.has_ogle_match = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "features": self.features[idx],
            "label": self.labels[idx],
        }
        if self.concept_gt is not None:
            item["concept_gt"] = self.concept_gt[idx]
        if self.has_ogle_match is not None:
            item["has_ogle_match"] = self.has_ogle_match[idx]
        return item


def create_dataloader(
    dataset: VariableStarDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    device: str = "cpu",
) -> DataLoader:
    """
    Create a DataLoader for variable star dataset.

    Design choices:
        batch_size=256: Training set ~14K samples -> ~55 steps/epoch
        num_workers=0:  CPU training, data very small (~672KB), loading is not a bottleneck
        pin_memory:     Auto-set to True when device is 'cuda' (speeds up host→device transfer)

    Args:
        dataset:     VariableStarDataset instance
        batch_size:  Batch size (default 256)
        shuffle:     Whether to shuffle (True for training, False for eval)
        num_workers: Number of data loading workers (default 0)
        pin_memory:  Whether to use pinned memory (None = auto based on device)
        device:      Target device; when 'cuda', pin_memory defaults to True

    Returns:
        Configured DataLoader instance.
    """
    if pin_memory is None:
        pin_memory = device.startswith("cuda")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
