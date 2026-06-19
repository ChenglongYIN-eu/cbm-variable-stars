#!/usr/bin/env python
"""
Script 04: Validate extracted features and generate quality report.

Loads the raw feature parquets from step 03, runs sigma-clipping
and physical-bounds validation, and generates a summary report.

Usage
-----
    python scripts/04_validate_features.py
    python scripts/04_validate_features.py --config configs/default.yaml
    python scripts/04_validate_features.py --input-dir data/interim
    python scripts/04_validate_features.py --sigma-clip 4.0
    python scripts/04_validate_features.py --strict  # strict physical bounds
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.config import load_config
from cbm_variable_stars.shared.logger import setup_logger, logger
from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12, CLASS_NAMES
from cbm_variable_stars.features.quality import validate_features

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate extracted features and generate quality report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input directory with raw feature parquets (default: cfg.paths.interim)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for validated features (default: same as input-dir)",
    )
    parser.add_argument(
        "--sigma-clip",
        type=float,
        default=3.0,
        help="Sigma clipping threshold for outlier detection",
    )
    parser.add_argument(
        "--max-nan-frac",
        type=float,
        default=0.3,
        help="Maximum NaN fraction per source to be considered valid",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict physical bounds (flag out-of-range as bad instead of acceptable)",
    )
    parser.add_argument(
        "--source",
        choices=["gaia", "ogle", "both"],
        default="both",
        help="Which feature files to validate",
    )
    return parser.parse_args()


def print_feature_stats(df: pd.DataFrame, source_name: str) -> None:
    """Print per-concept statistics for a feature DataFrame."""
    logger.info(f"\n{'='*50}")
    logger.info(f"{source_name} Feature Statistics:")
    logger.info(f"{'='*50}")
    logger.info(f"Total sources: {len(df)}")

    available = [c for c in CONCEPT_NAMES_12 if c in df.columns]
    logger.info(f"Concept columns: {len(available)}/{len(CONCEPT_NAMES_12)}")

    for col in available:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            logger.info(f"  {col:20s}: ALL NaN")
            continue
        nan_count = df[col].isna().sum()
        logger.info(
            f"  {col:20s}: "
            f"mean={col_data.mean():8.4f}, "
            f"std={col_data.std():7.4f}, "
            f"min={col_data.min():8.4f}, "
            f"max={col_data.max():8.4f}, "
            f"NaN={nan_count:5d} ({100.0*nan_count/len(df):.1f}%)"
        )

    if "quality_flag" in df.columns:
        flag_counts = df["quality_flag"].value_counts().sort_index()
        logger.info(f"\nQuality flags:")
        logger.info(f"  Good (0):       {flag_counts.get(0, 0)}")
        logger.info(f"  Acceptable (1): {flag_counts.get(1, 0)}")
        logger.info(f"  Bad (2):        {flag_counts.get(2, 0)}")

    if "label_name" in df.columns:
        dist = df["label_name"].value_counts().to_dict()
        logger.info(f"\nClass distribution: {dist}")

    if "alias_flag" in df.columns:
        n_alias = df["alias_flag"].sum()
        logger.info(f"\nAlias-flagged sources: {n_alias} ({100.0*n_alias/len(df):.1f}%)")


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    setup_logger(
        log_level=getattr(getattr(cfg, "project", {}), "log_level", "INFO"),
        log_dir=getattr(getattr(cfg, "project", {}), "log_dir", None),
        log_file="04_validate_features.log",
    )

    logger.info("=" * 60)
    logger.info("Step 04: Feature Validation")
    logger.info("=" * 60)

    paths_cfg = getattr(cfg, "paths", {})
    input_dir = Path(args.input_dir or paths_cfg.get("interim", "data/interim"))
    output_dir = Path(args.output_dir or input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Input dir:    {input_dir}")
    logger.info(f"  Sigma clip:   {args.sigma_clip}")
    logger.info(f"  Max NaN frac: {args.max_nan_frac}")
    logger.info(f"  Strict mode:  {args.strict}")

    sources_processed = []

    # ---- Gaia features ----
    if args.source in ("gaia", "both"):
        gaia_raw_path = input_dir / "gaia_features_raw.parquet"
        if gaia_raw_path.exists():
            gaia_df = pd.read_parquet(gaia_raw_path)
            logger.info(f"\nValidating Gaia features: {len(gaia_df)} sources")

            gaia_validated = validate_features(
                gaia_df,
                sigma_clip=args.sigma_clip,
                max_nan_fraction=args.max_nan_frac,
                strict_physical_bounds=args.strict,
            )
            print_feature_stats(gaia_validated, "Gaia")

            out_path = output_dir / "gaia_features_validated.parquet"
            gaia_validated.to_parquet(out_path, index=False)
            logger.info(f"\nGaia validated features saved: {out_path}")
            sources_processed.append("gaia")
        else:
            logger.warning(f"Gaia raw features not found: {gaia_raw_path}")
            if args.source == "gaia":
                sys.exit(1)

    # ---- OGLE features ----
    if args.source in ("ogle", "both"):
        ogle_raw_path = input_dir / "ogle_features_raw.parquet"
        if ogle_raw_path.exists():
            ogle_df = pd.read_parquet(ogle_raw_path)
            logger.info(f"\nValidating OGLE features: {len(ogle_df)} sources")

            ogle_validated = validate_features(
                ogle_df,
                sigma_clip=args.sigma_clip,
                max_nan_fraction=args.max_nan_frac,
                strict_physical_bounds=args.strict,
            )
            print_feature_stats(ogle_validated, "OGLE")

            out_path = output_dir / "ogle_features_validated.parquet"
            ogle_validated.to_parquet(out_path, index=False)
            logger.info(f"\nOGLE validated features saved: {out_path}")
            sources_processed.append("ogle")
        else:
            logger.warning(f"OGLE raw features not found: {ogle_raw_path}")

    if not sources_processed:
        logger.error("No feature files found to validate. Run script 03 first.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Step 04 complete.")
    logger.info(f"  Processed: {', '.join(sources_processed)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
