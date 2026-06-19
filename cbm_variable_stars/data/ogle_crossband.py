# cbm_variable_stars/data/ogle_crossband.py
"""
OGLE cross-band feature processing module.

Addresses audit issues:
  M6: Missing OGLE-to-Gaia feature unification handling
  S6: OGLE C11 (color index) and C12 (mean magnitude) not directly available

This module provides:
  1. Cross-band feature compatibility analysis (which concepts are
     directly comparable, approximately comparable, or not comparable)
  2. Enrich OGLE features with Gaia photometry via cross-match (scheme a)
  3. Missing value handling strategy when cross-match is unavailable (scheme b)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES_12,
    CONCEPTS_CROSS_SURVEY_10,
)

# ============================================================
# Cross-band feature compatibility analysis (M6)
# ============================================================
#
# | Concept         | Gaia G-band  | OGLE I-band  | Directly comparable? | Strategy         |
# |-----------------|-------------|-------------|---------------------|-----------------|
# | C1 period       | LS(G)       | LS(I)       | Yes                 | Period is band-independent |
# | C2 amplitude    | G-band amp  | I-band amp  | Approximate         | I amp systematically lower; CBM can learn offset |
# | C3 rise_frac    | G-band      | I-band      | Yes                 | Phase info, very weak band dependence |
# | C4 R21          | G-band      | I-band      | Yes                 | Amplitude ratio, weak band dependence (<5%) |
# | C5 R31          | G-band      | I-band      | Yes                 | Same as R21 |
# | C6 phi21        | G-band      | I-band      | Yes                 | Phase difference, theoretically band-independent |
# | C7 skewness     | G-band      | I-band      | Approximate         | Statistical, mild band dependence |
# | C8 kurtosis     | G-band      | I-band      | Approximate         | Same as skewness |
# | C9 stetson_K    | G-band      | I-band      | Approximate         | Weak band dependence |
# | C10 period_snr  | LS(G)       | LS(I)       | Approximate         | OGLE data denser, SNR systematically higher |
# | C11 color_bp_rp | Gaia bp_rp  | Not available | No               | Requires cross-match or mark missing |
# | C12 mean_mag    | G-band      | I-band      | No                  | Different bands, not comparable |
#
# Conclusion:
#   - C1-C9 + C10: Can be directly compared across surveys
#     (C2 amplitude and C10 SNR have systematic offsets but CBM can adapt)
#   - C11: Must be obtained via cross-match from Gaia, or marked as missing
#   - C12: Different bands, not comparable; mark as different band if used

DIRECTLY_COMPARABLE_CONCEPTS = [
    "period",        # C1: band-independent
    "rise_fraction", # C3: phase information
    "R21",           # C4: amplitude ratio
    "R31",           # C5: amplitude ratio
    "phi21",         # C6: phase difference
]

APPROXIMATELY_COMPARABLE_CONCEPTS = [
    "amplitude",     # C2: systematic offset (I < G)
    "skewness",      # C7: mild dependence
    "kurtosis",      # C8: mild dependence
    "stetson_K",     # C9: mild dependence
    "period_snr",    # C10: OGLE systematically higher
]

NOT_COMPARABLE_CONCEPTS = [
    "color_bp_rp",   # C11: OGLE has no GBP/GRP
    "mean_mag",      # C12: different bands
]


def enrich_ogle_with_gaia_photometry(
    ogle_features: pd.DataFrame,
    crossmatch_df: pd.DataFrame,
    gaia_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scheme a: Enrich OGLE features with Gaia BP-RP (C11) and mean G mag (C12)
    via Gaia-OGLE cross-match.

    Parameters
    ----------
    ogle_features : pd.DataFrame
        OGLE feature table, must contain "ogle_id" or "source_id" column
    crossmatch_df : pd.DataFrame
        Cross-match results with columns "ogle_id" and "source_id" (Gaia)
        from crossmatch.py output
    gaia_metadata : pd.DataFrame
        Gaia metadata table with columns "source_id", "bp_rp", "phot_g_mean_mag"

    Returns
    -------
    pd.DataFrame
        Enriched OGLE feature table with new/updated columns:
        - color_bp_rp: Gaia BP-RP color for matched sources (float, NaN for unmatched)
        - mean_mag_gaia_g: Gaia G-band mean magnitude for matched sources
        - has_gaia_match: bool, whether a Gaia cross-match was found

    Example
    -------
    >>> crossmatch = pd.read_parquet("data/interim/gaia_crossmatch.parquet")
    >>> gaia_meta = pd.read_parquet("data/raw/gaia/metadata/all_metadata.parquet")
    >>> ogle_enriched = enrich_ogle_with_gaia_photometry(
    ...     ogle_features, crossmatch, gaia_meta
    ... )
    >>> print(f"Match rate: {ogle_enriched['has_gaia_match'].mean():.1%}")
    """
    df = ogle_features.copy()

    # Determine OGLE ID column
    id_col = "ogle_id" if "ogle_id" in df.columns else "source_id"

    # Validate crossmatch and gaia_metadata have required columns
    required_xm = {"ogle_id", "source_id"}
    if not required_xm.issubset(crossmatch_df.columns):
        logger.error(
            f"[crossband] crossmatch_df missing columns. "
            f"Required: {required_xm}, got: {set(crossmatch_df.columns)}"
        )
        df["has_gaia_match"] = False
        return df

    required_gaia = {"source_id", "bp_rp", "phot_g_mean_mag"}
    if not required_gaia.issubset(gaia_metadata.columns):
        available = set(gaia_metadata.columns)
        missing = required_gaia - available
        logger.warning(
            f"[crossband] gaia_metadata missing columns: {missing}. "
            f"Available: {available}"
        )

    # Build gaia photometry lookup
    gaia_phot_cols = ["source_id"]
    if "bp_rp" in gaia_metadata.columns:
        gaia_phot_cols.append("bp_rp")
    if "phot_g_mean_mag" in gaia_metadata.columns:
        gaia_phot_cols.append("phot_g_mean_mag")

    gaia_phot = gaia_metadata[gaia_phot_cols].copy()
    gaia_phot["source_id"] = gaia_phot["source_id"].astype(str)

    # Join: crossmatch -> gaia photometry
    match_info = crossmatch_df[["ogle_id", "source_id"]].copy()
    match_info["source_id"] = match_info["source_id"].astype(str)
    match_info["ogle_id"] = match_info["ogle_id"].astype(str)

    match_info = match_info.merge(gaia_phot, on="source_id", how="left")

    # Rename columns for merge into OGLE features
    rename_cols = {"ogle_id": id_col}
    if "bp_rp" in match_info.columns:
        rename_cols["bp_rp"] = "_gaia_bp_rp"
    if "phot_g_mean_mag" in match_info.columns:
        rename_cols["phot_g_mean_mag"] = "mean_mag_gaia_g"

    match_info = match_info.rename(columns=rename_cols)
    # Drop duplicate source_id to avoid ambiguity; keep first match per ogle_id
    merge_cols = [id_col]
    if "_gaia_bp_rp" in match_info.columns:
        merge_cols.append("_gaia_bp_rp")
    if "mean_mag_gaia_g" in match_info.columns:
        merge_cols.append("mean_mag_gaia_g")

    match_info_dedup = match_info[merge_cols].drop_duplicates(subset=[id_col])

    # Merge into OGLE features
    df[id_col] = df[id_col].astype(str)
    df = df.merge(match_info_dedup, on=id_col, how="left")

    # Mark matched sources
    df["has_gaia_match"] = df["_gaia_bp_rp"].notna() if "_gaia_bp_rp" in df.columns else False
    n_matched = int(df["has_gaia_match"].sum())

    # Fill C11 for matched sources
    if "_gaia_bp_rp" in df.columns:
        df.loc[df["has_gaia_match"], "color_bp_rp"] = df.loc[
            df["has_gaia_match"], "_gaia_bp_rp"
        ]
        df.drop(columns=["_gaia_bp_rp"], inplace=True)
    elif "color_bp_rp" not in df.columns:
        df["color_bp_rp"] = np.nan

    # For unmatched sources, C11 remains NaN (filled with training median later)
    if "mean_mag_gaia_g" not in df.columns:
        df["mean_mag_gaia_g"] = np.nan

    logger.info(
        f"[crossband] OGLE-Gaia enrichment: "
        f"{n_matched}/{len(df)} ({100.0 * n_matched / max(len(df), 1):.1f}%) "
        f"sources have Gaia BP-RP"
    )

    return df


def prepare_ogle_cross_survey_dataset(
    ogle_features: pd.DataFrame,
    mode: str = "10dim",
) -> pd.DataFrame:
    """
    Scheme b: Prepare OGLE cross-survey test dataset with C11/C12 handling.

    Parameters
    ----------
    ogle_features : pd.DataFrame
        OGLE feature table (may already be enriched by enrich_ogle_with_gaia_photometry)
    mode : str
        Processing mode:
        - "10dim": Drop C11 and C12, use only 10 band-independent concepts.
          Best for fair cross-survey comparison experiments.
        - "12dim_with_match": Use 12 concepts, C11/C12 from Gaia cross-match.
          Include only sources with a Gaia match.
        - "12dim_fill_median": Use 12 concepts, fill missing C11/C12 with
          training set medians. Includes all sources but C11/C12 is
          uninformative for unmatched sources.

    Returns
    -------
    pd.DataFrame
        Processed OGLE feature table depending on mode:
        - "10dim": Contains only CONCEPTS_CROSS_SURVEY_10 concept columns
          plus metadata columns
        - "12dim_*": Contains full CONCEPT_NAMES_12 concept columns

    Example
    -------
    >>> ogle_10d = prepare_ogle_cross_survey_dataset(ogle_features, mode="10dim")
    >>> ogle_12d = prepare_ogle_cross_survey_dataset(
    ...     ogle_enriched, mode="12dim_with_match"
    ... )
    """
    df = ogle_features.copy()

    meta_cols = [
        "source_id", "ogle_id", "label", "label_name", "source",
        "n_obs", "quality_flag", "alias_flag", "has_gaia_match",
        "mean_mag_gaia_g",
    ]

    if mode == "10dim":
        # Keep only 10 band-independent concepts + metadata
        concept_cols = CONCEPTS_CROSS_SURVEY_10
        keep_meta = [c for c in meta_cols if c in df.columns]
        keep_cols = keep_meta + [c for c in concept_cols if c in df.columns]
        df = df[[c for c in keep_cols if c in df.columns]]

        n_concepts_available = sum(1 for c in concept_cols if c in df.columns)
        logger.info(
            f"[crossband] 10-dim mode: {n_concepts_available}/{len(concept_cols)} "
            f"concepts available, {len(df)} sources"
        )

    elif mode == "12dim_with_match":
        # Only include sources with Gaia cross-match
        if "has_gaia_match" in df.columns:
            n_before = len(df)
            df = df[df["has_gaia_match"]].copy()
            logger.info(
                f"[crossband] 12-dim matched mode: "
                f"{len(df)}/{n_before} ({100.0 * len(df) / max(n_before, 1):.1f}%) "
                f"sources have Gaia match"
            )
        else:
            logger.warning(
                "[crossband] No cross-match info available. "
                "Returning all sources (C11/C12 may be NaN)"
            )

    elif mode == "12dim_fill_median":
        # C11/C12 missing values will be filled during standardization
        n_c11_nan = (
            df["color_bp_rp"].isna().sum()
            if "color_bp_rp" in df.columns
            else len(df)
        )
        n_c12_nan = (
            df["mean_mag"].isna().sum()
            if "mean_mag" in df.columns
            else 0
        )
        logger.info(
            f"[crossband] 12-dim fill-median mode: "
            f"C11 missing {n_c11_nan}/{len(df)}, "
            f"C12 missing {n_c12_nan}/{len(df)}"
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}. "
            f"Valid options: '10dim', '12dim_with_match', '12dim_fill_median'"
        )

    return df
