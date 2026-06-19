#!/usr/bin/env python
"""
Script 01: Download Gaia DR3 variable star metadata and epoch photometry.

Downloads all 6 variable star types via Gaia TAP+ (ADQL) and DataLink.
Saves metadata parquet files and epoch photometry parquet files to
the configured data directory.

Usage
-----
    python scripts/01_download_gaia.py
    python scripts/01_download_gaia.py --config configs/default.yaml
    python scripts/01_download_gaia.py --output-dir /path/to/data/raw/gaia
    python scripts/01_download_gaia.py --var-type RRAB  # single type only
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add project root to path for development installs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.config import load_config
from cbm_variable_stars.shared.logger import setup_logger, logger
from cbm_variable_stars.shared.constants import CLASS_NAMES
from cbm_variable_stars.data.gaia_download import (
    download_all_gaia_data,
    query_gaia_metadata,
    download_epoch_photometry,
)
from cbm_variable_stars.data.validators import validate_gaia_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Gaia DR3 variable star data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: cfg.paths.raw_gaia)",
    )
    parser.add_argument(
        "--var-type",
        choices=CLASS_NAMES + ["all"],
        default="all",
        help="Variable type to download (default: all 6 types)",
    )
    parser.add_argument(
        "--skip-epoch-photometry",
        action="store_true",
        help="Skip epoch photometry download (metadata only)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation on downloaded data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    cfg = load_config(args.config)
    setup_logger(
        log_level=getattr(getattr(cfg, "project", {}), "log_level", "INFO"),
        log_dir=getattr(getattr(cfg, "project", {}), "log_dir", None),
        log_file="01_download_gaia.log",
    )

    logger.info("=" * 60)
    logger.info("Step 01: Download Gaia DR3 data")
    logger.info("=" * 60)
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Var type: {args.var_type}")

    output_dir = args.output_dir or getattr(
        getattr(cfg, "paths", {}), "raw_gaia", "data/raw/gaia"
    )
    output_dir = Path(output_dir)
    logger.info(f"  Output dir: {output_dir}")

    import pandas as pd

    if args.var_type == "all":
        # Download all types
        gaia_metadata = download_all_gaia_data(cfg, output_dir=output_dir)
    else:
        # Single type
        var_type = args.var_type
        meta_dir = output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        df = query_gaia_metadata(var_type=var_type, cfg=cfg)
        if len(df) > 0:
            meta_path = meta_dir / f"gaia_metadata_{var_type}.parquet"
            df.to_parquet(meta_path, index=False)
            logger.info(f"Saved metadata: {meta_path}")

            if not args.skip_epoch_photometry:
                ep_dir = output_dir / "epoch_photometry"
                ep_dir.mkdir(parents=True, exist_ok=True)
                downloaded = download_epoch_photometry(
                    source_ids=df["source_id"].tolist(),
                    output_dir=ep_dir,
                    cfg=cfg,
                )
                df["has_epoch_photometry"] = df["source_id"].isin(downloaded.keys())

            gaia_metadata = df
        else:
            logger.error(f"No data retrieved for {var_type}")
            sys.exit(1)

    if args.validate and len(gaia_metadata) > 0:
        logger.info("Running data validation...")
        report = validate_gaia_metadata(gaia_metadata)
        logger.info(f"Validation report:")
        logger.info(f"  Total:          {report['n_total']}")
        logger.info(f"  Valid:          {report['n_valid']}")
        logger.info(f"  Unknown labels: {report['unknown_labels']}")
        logger.info(f"  Passed:         {report['passed']}")
        if report["label_distribution"]:
            logger.info(f"  Label dist:     {report['label_distribution']}")

    logger.info("=" * 60)
    logger.info("Step 01 complete.")
    logger.info(f"  Downloaded: {len(gaia_metadata)} sources")
    if "has_epoch_photometry" in gaia_metadata.columns:
        n_ep = int(gaia_metadata["has_epoch_photometry"].sum())
        logger.info(f"  With epoch photometry: {n_ep}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
