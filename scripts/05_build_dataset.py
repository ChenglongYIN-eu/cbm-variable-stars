#!/usr/bin/env python
"""
Script 05: Build train/test datasets with the S2 split scheme.

Applies the corrected data split:
    - 15% hold-out -> test_in_domain
    - 85% -> cv_pool (5-fold stratified CV)
    - OGLE -> test_cross_survey (optional)

Fits a StandardScaler on the cv_pool and saves all splits as parquet files.

Usage
-----
    python scripts/05_build_dataset.py
    python scripts/05_build_dataset.py --config configs/default.yaml
    python scripts/05_build_dataset.py --no-ogle  # Gaia-only
    python scripts/05_build_dataset.py --ogle-mode 12dim_with_match
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.config import load_config
from cbm_variable_stars.shared.logger import setup_logger, logger
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12, CLASS_NAMES
from cbm_variable_stars.dataset.builder import build_datasets, get_cv_fold
from cbm_variable_stars.data.ogle_crossband import prepare_ogle_cross_survey_dataset

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/test datasets with S2 split scheme",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--gaia-features",
        default=None,
        help=(
            "Path to validated Gaia features parquet "
            "(default: cfg.paths.interim/gaia_features_validated.parquet)"
        ),
    )
    parser.add_argument(
        "--ogle-features",
        default=None,
        help="Path to validated OGLE features parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for processed datasets (default: cfg.paths.processed)",
    )
    parser.add_argument(
        "--no-ogle",
        action="store_true",
        help="Exclude OGLE cross-survey test set",
    )
    parser.add_argument(
        "--ogle-mode",
        choices=["10dim", "12dim_with_match", "12dim_fill_median"],
        default="10dim",
        help="OGLE cross-survey dataset mode (default: 10dim)",
    )
    parser.add_argument(
        "--verify-folds",
        action="store_true",
        help="Verify CV fold integrity after building",
    )
    return parser.parse_args()


def verify_datasets(datasets: dict, output_dir: Path) -> None:
    """Verify built datasets for integrity."""
    logger.info("\nDataset verification:")

    for name, df in datasets.items():
        available_concepts = [c for c in CONCEPT_NAMES_12 if c in df.columns]
        nan_rate = float(df[available_concepts].isna().mean().mean()) if available_concepts else float("nan")

        logger.info(f"  {name}:")
        logger.info(f"    Rows:          {len(df)}")
        logger.info(f"    Concept cols:  {len(available_concepts)}/{len(CONCEPT_NAMES_12)}")
        logger.info(f"    NaN rate:      {nan_rate:.1%}")

        if "label_name" in df.columns:
            dist = df["label_name"].value_counts().to_dict()
            logger.info(f"    Class dist:    {dist}")

        if "quality_flag" in df.columns:
            bad = int((df["quality_flag"] == 2).sum())
            logger.info(f"    Bad sources:   {bad}")

    # Verify CV folds
    cv_folds_path = output_dir / "cv_folds.pkl"
    if cv_folds_path.exists() and "cv_pool" in datasets:
        logger.info("  CV fold verification:")
        for k in range(5):
            try:
                fold_train, fold_val = get_cv_fold(
                    datasets["cv_pool"], k, cv_folds_path=cv_folds_path
                )
                logger.info(
                    f"    Fold {k}: train={len(fold_train)}, val={len(fold_val)}, "
                    f"total={len(fold_train)+len(fold_val)}"
                )
            except Exception as e:
                logger.error(f"    Fold {k} verification failed: {e}")


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    setup_logger(
        log_level=getattr(getattr(cfg, "project", {}), "log_level", "INFO"),
        log_dir=getattr(getattr(cfg, "project", {}), "log_dir", None),
        log_file="05_build_dataset.log",
    )
    if hasattr(cfg, "project") and hasattr(cfg.project, "random_seed"):
        set_global_seed(cfg.project.random_seed)

    logger.info("=" * 60)
    logger.info("Step 05: Build Datasets (S2 split scheme)")
    logger.info("=" * 60)

    paths_cfg = getattr(cfg, "paths", {})
    interim_dir = Path(paths_cfg.get("interim", "data/interim"))
    output_dir = Path(args.output_dir or paths_cfg.get("processed", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Interim dir: {interim_dir}")
    logger.info(f"  Output dir:  {output_dir}")
    logger.info(f"  OGLE mode:   {args.ogle_mode}")

    # ---- Load Gaia features ----
    gaia_path = args.gaia_features or interim_dir / "gaia_features_validated.parquet"
    gaia_path = Path(gaia_path)

    # Fallback to raw if validated not available
    if not gaia_path.exists():
        gaia_raw_path = interim_dir / "gaia_features_raw.parquet"
        if gaia_raw_path.exists():
            logger.warning(
                f"Validated Gaia features not found at {gaia_path}. "
                f"Using raw features (run script 04 for quality validation)."
            )
            gaia_path = gaia_raw_path
        else:
            logger.error(
                f"Gaia features not found: {gaia_path}. "
                f"Run scripts 01-03 first."
            )
            sys.exit(1)

    gaia_features = pd.read_parquet(gaia_path)
    logger.info(f"Loaded Gaia features: {len(gaia_features)} sources from {gaia_path}")

    # ---- Load OGLE features (optional) ----
    ogle_features = None
    if not args.no_ogle:
        ogle_path = args.ogle_features or interim_dir / "ogle_features_validated.parquet"
        ogle_path = Path(ogle_path)

        if not ogle_path.exists():
            ogle_raw_path = interim_dir / "ogle_features_raw.parquet"
            if ogle_raw_path.exists():
                logger.warning(
                    f"Validated OGLE features not found, using raw: {ogle_raw_path}"
                )
                ogle_path = ogle_raw_path

        if ogle_path.exists():
            ogle_features_raw = pd.read_parquet(ogle_path)
            logger.info(f"Loaded OGLE features: {len(ogle_features_raw)} sources")

            # Apply cross-survey mode preparation
            ogle_features = prepare_ogle_cross_survey_dataset(
                ogle_features_raw,
                mode=args.ogle_mode,
            )
            logger.info(
                f"OGLE {args.ogle_mode} mode: {len(ogle_features)} sources"
            )
        else:
            logger.info("No OGLE features found, building Gaia-only dataset")

    # ---- Build datasets ----
    logger.info("\nBuilding datasets...")
    datasets = build_datasets(
        gaia_features=gaia_features,
        ogle_features=ogle_features,
        cfg=cfg,
        output_dir=output_dir,
    )

    # ---- Verification ----
    if args.verify_folds:
        verify_datasets(datasets, output_dir)
    else:
        # Basic summary
        for name, df in datasets.items():
            logger.info(f"  {name}: {len(df)} sources")

    # ---- Final report ----
    logger.info("=" * 60)
    logger.info("Step 05 complete.")
    logger.info(f"  CV pool:          {len(datasets['cv_pool'])} sources")
    logger.info(f"  Hold-out test:    {len(datasets['test_in_domain'])} sources")
    if "test_cross_survey" in datasets:
        ogle_test = datasets["test_cross_survey"]
        n_matched = 0
        if "has_gaia_match" in ogle_test.columns:
            n_matched = int(ogle_test["has_gaia_match"].sum())
        logger.info(
            f"  Cross-survey test: {len(ogle_test)} sources "
            f"({n_matched} with Gaia match)"
        )
    logger.info(f"  Scaler saved:     {output_dir / 'scaler.pkl'}")
    logger.info(f"  CV folds saved:   {output_dir / 'cv_folds.pkl'}")
    logger.info(f"  Output dir:       {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
