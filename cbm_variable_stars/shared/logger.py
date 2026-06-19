"""Logging system using loguru."""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


# Remove default handler
logger.remove()

# Add stdout handler with nice formatting
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


def setup_logger(
    log_level: str = "INFO",
    log_dir: Optional[str | Path] = None,
    log_file: str = "cbm_variable_stars.log",
) -> None:
    """
    Configure the global logger.

    Parameters
    ----------
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR.
    log_dir : str or Path, optional
        Directory for log files. If None, only console logging.
    log_file : str
        Log file name.
    """
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            encoding="utf-8",
        )

    logger.info(f"Logger initialized (level={log_level})")
