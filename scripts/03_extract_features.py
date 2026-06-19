#!/usr/bin/env python
"""
Script 03: Extract features from Gaia and OGLE light curves.

Runs the full 12-concept feature extraction pipeline for both
Gaia (G-band) and OGLE (I-band) data. Optionally performs
Gaia-OGLE cross-matching to enrich OGLE features with Gaia
photometry (C11/C12).

Usage
-----
    python scripts/03_extract_features.py
    python scripts/03_extract_features.py --config configs/default.yaml
    python scripts/03_extract_features.py --source gaia  # Gaia only
    python scripts/03_extract_features.py --source ogle  # OGLE only
    python scripts/03_extract_features.py --skip-crossmatch
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
from cbm_variable_stars.features.extractor import extract_features_batch
from cbm_variable_stars.features.quality import validate_features
from cbm_variable_stars.data.crossmatch import (
    query_gavras_crossmatch,
    coordinate_crossmatch,
)
from cbm_variable_stars.data.ogle_crossband import enrich_ogle_with_gaia_photometry

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 12-concept features from Gaia and OGLE light curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--source",
        choices=["gaia", "ogle", "both"],
        default="both",
        help="Which data source to process",
    )
    parser.add_argument(
        "--gaia-metadata",
        default=None,
        help="Path to Gaia metadata parquet (default: cfg.paths.raw_gaia/metadata/all_metadata.parquet)",
    )
    parser.add_argument(
        "--gaia-lc-dir",
        default=None,
        help="Path to Gaia epoch photometry directory",
    )
    parser.add_argument(
        "--ogle-metadata",
        default=None,
        help="Path to OGLE params parquet (default: cfg.paths.raw_ogle/metadata/ogle_all_params.parquet)",
    )
    parser.add_argument(
        "--ogle-lc-dir",
        default=None,
        help="Path to OGLE light curves directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for feature parquets (default: cfg.paths.interim)",
    )
    parser.add_argument(
        "--skip-crossmatch",
        action="store_true",
        help="Skip Gaia-OGLE cross-match enrichment (OGLE C11/C12 will be NaN)",
    )
    parser.add_argument(
        "--crossmatch-radius",
        type=float,
        default=1.0,
        help="Cross-match radius in arcseconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    setup_logger(
        log_level=getattr(getattr(cfg, "project", {}), "log_level", "INFO"),
        log_dir=getattr(getattr(cfg, "project", {}), "log_dir", None),
        log_file="03_extract_features.log",
    )
    if hasattr(cfg, "project") and hasattr(cfg.project, "random_seed"):
        set_global_seed(cfg.project.random_seed)

    logger.info("=" * 60)
    logger.info("Step 03: Feature Extraction")
    logger.info("=" * 60)

    # Resolve paths
    paths_cfg = getattr(cfg, "paths", {})
    raw_gaia = Path(args.gaia_lc_dir or paths_cfg.get("raw_gaia", "data/raw/gaia"))
    raw_ogle = Path(args.ogle_lc_dir or paths_cfg.get("raw_ogle", "data/raw/ogle"))
    output_dir = Path(args.output_dir or paths_cfg.get("interim", "data/interim"))
    output_dir.mkdir(parents=True, exist_ok=True)

    gaia_features = None
    ogle_features = None
    gaia_metadata = None

    # ---- Gaia feature extraction ----
    if args.source in ("gaia", "both"):
        meta_path = args.gaia_metadata or raw_gaia / "metadata" / "all_metadata.parquet"
        meta_path = Path(meta_path)

        if not meta_path.exists():
            logger.error(f"Gaia metadata not found: {meta_path}. Run script 01 first.")
            if args.source == "gaia":
                sys.exit(1)
        else:
            gaia_metadata = pd.read_parquet(meta_path)
            logger.info(f"Loaded Gaia metadata: {len(gaia_metadata)} sources")

            # Filter to sources with epoch photometry
            if "has_epoch_photometry" in gaia_metadata.columns:
                gaia_with_ep = gaia_metadata[gaia_metadata["has_epoch_photometry"]]
                logger.info(
                    f"Sources with epoch photometry: {len(gaia_with_ep)}/{len(gaia_metadata)}"
                )
            else:
                gaia_with_ep = gaia_metadata

            ep_dir = raw_gaia / "epoch_photometry"
            logger.info(f"Extracting Gaia features from {ep_dir}...")

            gaia_features = extract_features_batch(
                metadata_df=gaia_with_ep,
                lc_dir=ep_dir,
                cfg=cfg,
                source="gaia",
            )

            if len(gaia_features) > 0:
                # Quality validation
                gaia_features = validate_features(gaia_features)

                # Save
                gaia_out = output_dir / "gaia_features_raw.parquet"
                gaia_features.to_parquet(gaia_out, index=False)
                logger.info(f"Gaia features saved: {gaia_out} ({len(gaia_features)} rows)")

    # ---- OGLE feature extraction ----
    if args.source in ("ogle", "both"):
        ogle_meta_path = args.ogle_metadata or raw_ogle / "metadata" / "ogle_all_params.parquet"
        ogle_meta_path = Path(ogle_meta_path)

        if not ogle_meta_path.exists():
            logger.error(f"OGLE metadata not found: {ogle_meta_path}. Run script 02 first.")
            if args.source == "ogle":
                sys.exit(1)
        else:
            ogle_metadata = pd.read_parquet(ogle_meta_path)
            logger.info(f"Loaded OGLE metadata: {len(ogle_metadata)} sources")

            ogle_lc_dir = raw_ogle / "light_curves"
            logger.info(f"Extracting OGLE features from {ogle_lc_dir}...")

            ogle_features = extract_features_batch(
                metadata_df=ogle_metadata,
                lc_dir=ogle_lc_dir,
                cfg=cfg,
                source="ogle",
            )

            if len(ogle_features) > 0:
                # Quality validation
                ogle_features = validate_features(ogle_features)

                # Cross-match enrichment for C11/C12
                if not args.skip_crossmatch and gaia_metadata is not None:
                    logger.info("Running Gaia-OGLE cross-match for C11/C12 enrichment...")

                    xm_cache = output_dir / "gaia_crossmatch.parquet"
                    crossmatch_df = query_gavras_crossmatch(output_path=xm_cache)

                    if len(crossmatch_df) == 0:
                        logger.info(
                            "Gavras cross-match empty, trying coordinate matching..."
                        )
                        crossmatch_df = coordinate_crossmatch(
                            gaia_df=gaia_metadata,
                            ogle_df=ogle_metadata,
                            radius_arcsec=args.crossmatch_radius,
                        )
                        if len(crossmatch_df) > 0:
                            crossmatch_df.to_parquet(xm_cache, index=False)

                    if len(crossmatch_df) > 0:
                        ogle_features = enrich_ogle_with_gaia_photometry(
                            ogle_features=ogle_features,
                            crossmatch_df=crossmatch_df,
                            gaia_metadata=gaia_metadata,
                        )
                        logger.info(
                            f"OGLE enriched: "
                            f"{ogle_features.get('has_gaia_match', pd.Series([False])).sum()} "
                            f"sources with Gaia BP-RP"
                        )
                    else:
                        logger.warning(
                            "No cross-match found. OGLE C11 will be NaN. "
                            "Use 10-dim mode for cross-survey experiments."
                        )

                # Save
                ogle_out = output_dir / "ogle_features_raw.parquet"
                ogle_features.to_parquet(ogle_out, index=False)
                logger.info(f"OGLE features saved: {ogle_out} ({len(ogle_features)} rows)")

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Step 03 complete.")
    if gaia_features is not None:
        logger.info(f"  Gaia features: {len(gaia_features)} sources")
    if ogle_features is not None:
        logger.info(f"  OGLE features: {len(ogle_features)} sources")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
