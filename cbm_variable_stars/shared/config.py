"""Configuration loading and validation."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig

from cbm_variable_stars.shared.logger import logger


def load_config(
    config_path: str | Path = "configs/default.yaml",
    overrides: Optional[list[str]] = None,
    strict: bool = False,
) -> DictConfig:
    """
    Load YAML configuration with optional CLI overrides.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.
    overrides : list of str, optional
        CLI-style overrides, e.g. ["training.lr=0.01", "dataset.batch_size=128"]
    strict : bool, optional
        If True, raise FileNotFoundError when the config file does not exist.
        If False (default), fall back to an empty config with a warning.

    Returns
    -------
    DictConfig
        Merged configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        if strict:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.warning(f"Config file not found: {config_path}, using empty config")
        cfg = OmegaConf.create({})
    else:
        cfg = OmegaConf.load(config_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    logger.info(f"Config saved to {path}")
