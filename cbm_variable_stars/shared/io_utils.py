"""File I/O utilities for Parquet, JSON, and pickle."""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from cbm_variable_stars.shared.logger import logger


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to Parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.debug(f"Saved Parquet: {path} ({len(df)} rows)")


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load DataFrame from Parquet file."""
    path = Path(path)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.debug(f"Loaded Parquet: {path} ({len(df)} rows)")
    return df


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file with numpy serialization support."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder, ensure_ascii=False)
    logger.debug(f"Saved JSON: {path}")


def load_json(path: str | Path) -> Any:
    """Load data from JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(f"Loaded JSON: {path}")
    return data


def save_pickle(obj: Any, path: str | Path) -> None:
    """Save object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug(f"Saved pickle: {path}")


def load_pickle(path: str | Path) -> Any:
    """Load object from pickle file."""
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.debug(f"Loaded pickle: {path}")
    return obj
