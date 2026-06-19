#!/usr/bin/env python
"""
Script 02: Download OGLE-IV variable star catalog data.

Downloads params.dat files and individual light curve .dat files
for all 6 variable star types. Prefers HTTP over FTP (M4 correction).

Usage
-----
    python scripts/02_download_ogle.py
    python scripts/02_download_ogle.py --config configs/default.yaml
    python scripts/02_download_ogle.py --var-type RRAB --target-n 2000
    python scripts/02_download_ogle.py --no-http  # force FTP
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.config import load_config
from cbm_variable_stars.shared.logger import setup_logger, logger
from cbm_variable_stars.shared.constants import CLASS_NAMES
from cbm_variable_stars.data.ogle_download import download_ogle_catalog
from cbm_variable_stars.data.validators import validate_ogle_params

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OGLE-IV variable star data",
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
        help="Override output directory (default: cfg.paths.raw_ogle)",
    )
    parser.add_argument(
        "--var-type",
        choices=CLASS_NAMES + ["all"],
        default="all",
        help="Variable type to download (default: all)",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=None,
        help="Override target number of sources per type",
    )
    parser.add_argument(
        "--no-http",
        action="store_true",
        help="Disable HTTP download, force FTP",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=None,
        help="Minimum observations per light curve (default: cfg.ogle.min_observations)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    setup_logger(
        log_level=getattr(getattr(cfg, "project", {}), "log_level", "INFO"),
        log_dir=getattr(getattr(cfg, "project", {}), "log_dir", None),
        log_file="02_download_ogle.log",
    )

    logger.info("=" * 60)
    logger.info("Step 02: Download OGLE-IV data")
    logger.info("=" * 60)

    output_dir = args.output_dir or getattr(
        getattr(cfg, "paths", {}), "raw_ogle", "data/raw/ogle"
    )
    output_dir = Path(output_dir)
    logger.info(f"  Output dir: {output_dir}")

    # Download parameters
    ogle_cfg = getattr(cfg, "ogle", {})
    prefer_http = not args.no_http and ogle_cfg.get("prefer_http", True)
    min_observations = args.min_obs or ogle_cfg.get("min_observations", 50)
    max_retries = ogle_cfg.get("max_retries", 3)
    timeout = ogle_cfg.get("download_timeout", 300)

    logger.info(f"  HTTP preferred: {prefer_http}")
    logger.info(f"  Min observations: {min_observations}")

    # Determine which types to download
    if args.var_type == "all":
        var_types_to_process = CLASS_NAMES
    else:
        var_types_to_process = [args.var_type]

    all_params_list = []
    total_lc_paths = {}

    for var_type in var_types_to_process:
        # Get target N from config or override
        if args.target_n is not None:
            target_n = args.target_n
        else:
            var_types_cfg = getattr(cfg, "var_types", {})
            vt_cfg = var_types_cfg.get(var_type, {}) if hasattr(var_types_cfg, "get") else {}
            target_n = vt_cfg.get("target_n_ogle", 2000) if hasattr(vt_cfg, "get") else 2000

        logger.info(f"\n[{var_type}] Target: {target_n} sources")

        params_df, lc_paths = download_ogle_catalog(
            var_type=var_type,
            target_n=target_n,
            output_dir=output_dir,
            min_observations=min_observations,
            max_retries=max_retries,
            timeout=timeout,
            prefer_http=prefer_http,
        )

        if len(params_df) > 0:
            # Validate
            report = validate_ogle_params(params_df)
            logger.info(
                f"[{var_type}] Validation: {report['n_valid']}/{report['n_total']} valid"
            )
            all_params_list.append(params_df)
            total_lc_paths.update(lc_paths)

    # Combine and save all metadata
    if all_params_list:
        combined = pd.concat(all_params_list, ignore_index=True)
        meta_dir = output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        combined_path = meta_dir / "ogle_all_params.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info(f"\n  Combined metadata saved: {combined_path}")
    else:
        combined = pd.DataFrame()

    logger.info("=" * 60)
    logger.info("Step 02 complete.")
    logger.info(f"  Total sources: {len(combined)}")
    logger.info(f"  Total light curves: {len(total_lc_paths)}")
    if len(combined) > 0 and "label_name" in combined.columns:
        dist = combined["label_name"].value_counts().to_dict()
        logger.info(f"  Distribution: {dist}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
