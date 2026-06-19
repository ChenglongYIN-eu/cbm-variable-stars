"""Random seed management for reproducibility."""

from __future__ import annotations
import random
import numpy as np

from cbm_variable_stars.shared.constants import RANDOM_SEED
from cbm_variable_stars.shared.logger import logger


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for all relevant libraries.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import os
        # [Min3 FIX] Required for full GPU determinism with CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    logger.info(f"Global random seed set to {seed}")
