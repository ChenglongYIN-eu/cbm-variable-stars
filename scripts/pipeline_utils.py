"""
Shared utility functions for pipeline scripts.

Extracted from run_real_data_pipeline.py and run_expanded_pipeline.py
to eliminate code duplication (Fix M1).

Provides:
  - QUERIES: ADQL query templates for each variable star class
  - query_gaia_class: query Gaia DR3 for a single variable star class
  - compute_derived_features: derive concept features from Gaia catalog columns
  - build_feature_matrix: extract 12-concept feature matrix
  - compute_imputation_stats: fit imputation on training data only
  - apply_imputation: transform features using pre-computed stats
"""

from __future__ import annotations
import time

import numpy as np
import pandas as pd

from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12, LABEL_TO_IDX
from cbm_variable_stars.shared.logger import logger


# ======================================================================
# ADQL Queries — use specific Gaia DR3 variability tables
# ======================================================================

QUERIES = {
    "RRAB": """
        SELECT r.source_id, r.pf AS period, r.peak_to_peak_g AS amplitude,
               r.r21_g, r.r31_g, r.phi21_g, r.phi31_g,
               r.pf_error, r.num_clean_epochs_g,
               s.skewness_mag_g_fov AS skewness,
               s.kurtosis_mag_g_fov AS kurtosis,
               s.stetson_mag_g_fov AS stetson_K,
               s.iqr_mag_g_fov AS iqr,
               s.std_dev_mag_g_fov AS mag_std,
               s.abbe_mag_g_fov AS abbe_value,
               g.bp_rp, g.phot_g_mean_mag AS mean_mag
        FROM gaiadr3.vari_rrlyrae AS r
        JOIN gaiadr3.vari_summary AS s ON r.source_id = s.source_id
        JOIN gaiadr3.gaia_source AS g ON r.source_id = g.source_id
        WHERE r.best_classification = 'RRab'
          AND r.pf > 0 AND r.peak_to_peak_g > 0
          AND s.num_selected_g_fov >= 20
    """,
    "RRC": """
        SELECT r.source_id, r.p1_o AS period, r.peak_to_peak_g AS amplitude,
               r.r21_g, r.r31_g, r.phi21_g, r.phi31_g,
               r.p1_o_error AS pf_error, r.num_clean_epochs_g,
               s.skewness_mag_g_fov AS skewness,
               s.kurtosis_mag_g_fov AS kurtosis,
               s.stetson_mag_g_fov AS stetson_K,
               s.iqr_mag_g_fov AS iqr,
               s.std_dev_mag_g_fov AS mag_std,
               s.abbe_mag_g_fov AS abbe_value,
               g.bp_rp, g.phot_g_mean_mag AS mean_mag
        FROM gaiadr3.vari_rrlyrae AS r
        JOIN gaiadr3.vari_summary AS s ON r.source_id = s.source_id
        JOIN gaiadr3.gaia_source AS g ON r.source_id = g.source_id
        WHERE r.best_classification = 'RRc'
          AND r.p1_o > 0 AND r.peak_to_peak_g > 0
          AND s.num_selected_g_fov >= 20
    """,
    "DCEP": """
        SELECT c.source_id, c.pf AS period, c.peak_to_peak_g AS amplitude,
               c.r21_g, c.r31_g, c.phi21_g, c.phi31_g,
               c.pf_error, c.num_clean_epochs_g,
               s.skewness_mag_g_fov AS skewness,
               s.kurtosis_mag_g_fov AS kurtosis,
               s.stetson_mag_g_fov AS stetson_K,
               s.iqr_mag_g_fov AS iqr,
               s.std_dev_mag_g_fov AS mag_std,
               s.abbe_mag_g_fov AS abbe_value,
               g.bp_rp, g.phot_g_mean_mag AS mean_mag
        FROM gaiadr3.vari_cepheid AS c
        JOIN gaiadr3.vari_summary AS s ON c.source_id = s.source_id
        JOIN gaiadr3.gaia_source AS g ON c.source_id = g.source_id
        WHERE c.type_best_classification = 'DCEP'
          AND c.pf > 0 AND c.peak_to_peak_g > 0
          AND s.num_selected_g_fov >= 20
    """,
    "DSCT_SXPHE": """
        SELECT d.source_id, 1.0/d.frequency AS period,
               d.amplitude_estimate AS amplitude,
               s.skewness_mag_g_fov AS skewness,
               s.kurtosis_mag_g_fov AS kurtosis,
               s.stetson_mag_g_fov AS stetson_K,
               s.iqr_mag_g_fov AS iqr,
               s.std_dev_mag_g_fov AS mag_std,
               s.abbe_mag_g_fov AS abbe_value,
               g.bp_rp, g.phot_g_mean_mag AS mean_mag
        FROM gaiadr3.vari_short_timescale AS d
        JOIN gaiadr3.vari_summary AS s ON d.source_id = s.source_id
        JOIN gaiadr3.gaia_source AS g ON d.source_id = g.source_id
        JOIN gaiadr3.vari_classifier_result AS v ON d.source_id = v.source_id
        WHERE v.best_class_name = 'DSCT|GDOR|SXPHE'
          AND d.frequency > 0
          AND s.num_selected_g_fov >= 20
    """,
    "ECL": """
        SELECT e.source_id, 1.0/e.frequency AS period,
               e.geom_model_gaussian1_depth AS amplitude,
               s.skewness_mag_g_fov AS skewness,
               s.kurtosis_mag_g_fov AS kurtosis,
               s.stetson_mag_g_fov AS stetson_K,
               s.iqr_mag_g_fov AS iqr,
               s.std_dev_mag_g_fov AS mag_std,
               s.abbe_mag_g_fov AS abbe_value,
               g.bp_rp, g.phot_g_mean_mag AS mean_mag
        FROM gaiadr3.vari_eclipsing_binary AS e
        JOIN gaiadr3.vari_summary AS s ON e.source_id = s.source_id
        JOIN gaiadr3.gaia_source AS g ON e.source_id = g.source_id
        WHERE e.frequency > 0
          AND s.num_selected_g_fov >= 20
    """,
    "MIRA_SR": """
        SELECT l.source_id, 1.0/l.frequency AS period,
               l.amplitude AS amplitude,
               s.skewness_mag_g_fov AS skewness,
               s.kurtosis_mag_g_fov AS kurtosis,
               s.stetson_mag_g_fov AS stetson_K,
               s.iqr_mag_g_fov AS iqr,
               s.std_dev_mag_g_fov AS mag_std,
               s.abbe_mag_g_fov AS abbe_value,
               g.bp_rp, g.phot_g_mean_mag AS mean_mag
        FROM gaiadr3.vari_long_period_variable AS l
        JOIN gaiadr3.vari_summary AS s ON l.source_id = s.source_id
        JOIN gaiadr3.gaia_source AS g ON l.source_id = g.source_id
        WHERE l.frequency > 0
          AND s.num_selected_g_fov >= 20
    """,
}

# Classes that have catalog Fourier parameters
HAS_FOURIER = {"RRAB", "RRC", "DCEP"}


def query_gaia_class(var_type: str, max_rows: int = 5000) -> pd.DataFrame:
    """Query Gaia DR3 for a single variable star class."""
    from astroquery.gaia import Gaia

    template = QUERIES[var_type]
    # Insert TOP N after first SELECT
    query = template.replace("SELECT ", f"SELECT TOP {max_rows} ", 1)

    logger.info(f"[{var_type}] Querying Gaia DR3 (TOP {max_rows})...")

    for attempt in range(3):
        try:
            job = Gaia.launch_job(query, verbose=False)
            result = job.get_results()
            df = result.to_pandas()

            df["label_name"] = var_type
            df["label"] = LABEL_TO_IDX[var_type]
            df["source_id"] = df["source_id"].astype(str)

            logger.info(f"[{var_type}] Retrieved {len(df)} sources")
            return df

        except Exception as e:
            logger.warning(f"[{var_type}] Attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(30)

    logger.error(f"[{var_type}] All attempts failed")
    return pd.DataFrame()


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived concept features from catalog columns."""
    df = df.copy()

    # C3: rise_fraction requires phase-folded light curve analysis.
    # When only catalog data is available, set to NaN for downstream imputation.
    # The previous tanh(skewness) approximation (range [0.4, 0.6]) had no physical
    # basis. NaN is more honest: it flows through compute_imputation_stats (per-class
    # median will be NaN if all-NaN), then apply_imputation keeps NaN, and finally
    # transforms.py nan_to_num replaces with 0.0.
    if "rise_fraction" not in df.columns:
        df["rise_fraction"] = np.nan
        logger.info("rise_fraction set to NaN (requires light curve analysis)")

    # C4-C6: Fourier params — use catalog values where available, else NaN
    for col in ["r21_g", "r31_g", "phi21_g"]:
        if col not in df.columns:
            df[col] = np.nan

    # Rename Gaia columns to our concept names
    rename_map = {"r21_g": "R21", "r31_g": "R31", "phi21_g": "phi21"}
    for gaia_col, concept_col in rename_map.items():
        if gaia_col in df.columns and concept_col not in df.columns:
            df[concept_col] = df[gaia_col]

    # C6: phi21 normalization to [0, 2*pi)
    if "phi21" in df.columns:
        df["phi21"] = df["phi21"] % (2 * np.pi)

    # C10: period_snr proxy: log10(period/pf_error)*2
    # NOTE: This is an approximation used when FAP is unavailable from catalog data.
    # It differs from the extractor.py definition (-log10(FAP)) which requires
    # light curve analysis. Within this pipeline, period_snr is computed consistently
    # using this proxy formula. See constants.py C10 for the canonical definition.
    if "pf_error" in df.columns and "period" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = df["period"] / df["pf_error"].replace(0, np.nan)
            df["period_snr"] = np.log10(ratio.clip(1, None)) * 2
    else:
        df["period_snr"] = 5.0

    # C11: color_bp_rp
    if "color_bp_rp" not in df.columns and "bp_rp" in df.columns:
        df["color_bp_rp"] = df["bp_rp"]

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build the 12-concept feature matrix from the combined DataFrame.

    Returns (features, labels, metadata_df)
    """
    for col in CONCEPT_NAMES_12:
        if col not in df.columns:
            logger.warning(f"  Missing concept column: {col}, filling with NaN")
            df[col] = np.nan

    features = df[CONCEPT_NAMES_12].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)

    n_total = len(df)
    for i, col in enumerate(CONCEPT_NAMES_12):
        n_valid = np.sum(~np.isnan(features[:, i]))
        pct = 100 * n_valid / n_total
        if pct < 100:
            logger.info(f"  {col}: {n_valid}/{n_total} valid ({pct:.1f}%)")

    return features, labels, df


def compute_imputation_stats(features: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute per-class median imputation statistics on training data only.

    MUST be called ONLY on training/CV data to prevent data leakage (Fix C1).

    Returns:
        dict with 'per_class_medians' and 'global_medians' per feature column.
    """
    assert features.ndim == 2
    stats = {"per_class_medians": {}, "global_medians": {}}
    for j in range(features.shape[1]):
        col = features[:, j]
        class_meds = {}
        for cls in np.unique(labels):
            vals = col[(labels == cls) & ~np.isnan(col)]
            if len(vals) > 0:
                class_meds[int(cls)] = float(np.median(vals))
        stats["per_class_medians"][j] = class_meds
        valid = col[~np.isnan(col)]
        stats["global_medians"][j] = float(np.median(valid)) if len(valid) > 0 else 0.0
    return stats


def apply_imputation(
    features: np.ndarray,
    labels: np.ndarray,
    stats: dict,
    use_class_labels: bool = True,
) -> np.ndarray:
    """
    Apply pre-computed imputation statistics to fill NaN values.

    Args:
        features:         Feature matrix with potential NaN values.
        labels:           Label array (used for per-class routing if use_class_labels=True).
        stats:            Imputation statistics from compute_imputation_stats().
        use_class_labels: If True, use per-class medians (appropriate for training data
                          where labels are known). If False, use only global medians
                          (appropriate for test data to prevent label leakage).
    """
    features = features.copy()
    for j in range(features.shape[1]):
        if np.any(np.isnan(features[:, j])):
            if use_class_labels:
                for cls in np.unique(labels):
                    mask = (labels == cls) & np.isnan(features[:, j])
                    if np.any(mask):
                        med = stats["per_class_medians"].get(j, {}).get(
                            int(cls), stats["global_medians"].get(j, 0.0)
                        )
                        features[mask, j] = med
            remaining = np.isnan(features[:, j])
            if np.any(remaining):
                features[remaining, j] = stats["global_medians"].get(j, 0.0)
    return features
