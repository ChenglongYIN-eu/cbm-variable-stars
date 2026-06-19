# cbm_variable_stars/data/validators.py
"""
Data validation module for Gaia and OGLE metadata tables.

Checks required columns, value ranges, and data integrity.
Returns validation reports to inform downstream processing.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    LABEL_TO_IDX,
    PHYSICAL_PRIOR_RANGES,
)


# Required columns for Gaia metadata
GAIA_REQUIRED_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "label_name",
    "label",
]

# Required columns for OGLE params
OGLE_REQUIRED_COLUMNS = [
    "ogle_id",
    "field",
    "var_class",
    "label_name",
    "label",
]

# Value range checks for Gaia metadata
GAIA_VALUE_RANGES = {
    "phot_g_mean_mag": (3.0, 22.0),
    "ra": (0.0, 360.0),
    "dec": (-90.0, 90.0),
}


def validate_gaia_metadata(
    df: pd.DataFrame,
    strict: bool = False,
) -> Dict[str, object]:
    """
    Validate Gaia DR3 metadata table.

    Parameters
    ----------
    df : pd.DataFrame
        Gaia metadata table to validate
    strict : bool
        If True, raise ValueError on critical failures.
        If False (default), log warnings and return validation report.

    Returns
    -------
    dict
        Validation report with keys:
        - "n_total": total number of rows
        - "n_valid": rows passing all checks
        - "missing_columns": list of missing required columns
        - "out_of_range": dict {col: n_violations}
        - "unknown_labels": number of rows with unrecognized label_name
        - "nan_fractions": dict {col: nan_fraction}
        - "label_distribution": dict {label_name: count}
        - "passed": bool, True if no critical errors

    Example
    -------
    >>> report = validate_gaia_metadata(gaia_df)
    >>> if not report["passed"]:
    ...     logger.error(f"Validation failed: {report}")
    """
    report: Dict[str, object] = {
        "n_total": len(df),
        "n_valid": 0,
        "missing_columns": [],
        "out_of_range": {},
        "unknown_labels": 0,
        "nan_fractions": {},
        "label_distribution": {},
        "passed": True,
    }

    if len(df) == 0:
        logger.warning("[validate_gaia] Empty DataFrame")
        report["passed"] = False
        return report

    # Check required columns
    missing = [c for c in GAIA_REQUIRED_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing
    if missing:
        logger.warning(f"[validate_gaia] Missing required columns: {missing}")
        if strict:
            raise ValueError(f"Gaia metadata missing columns: {missing}")
        report["passed"] = False

    # NaN fractions for key columns
    for col in GAIA_REQUIRED_COLUMNS + ["bp_rp", "period"]:
        if col in df.columns:
            nan_frac = float(df[col].isna().mean())
            report["nan_fractions"][col] = nan_frac
            if nan_frac > 0.5:
                logger.warning(
                    f"[validate_gaia] Column '{col}' has {nan_frac:.1%} NaN values"
                )

    # Value range checks
    for col, (vmin, vmax) in GAIA_VALUE_RANGES.items():
        if col in df.columns:
            valid_mask = df[col].notna()
            n_oob = int(((df.loc[valid_mask, col] < vmin) | (df.loc[valid_mask, col] > vmax)).sum())
            report["out_of_range"][col] = n_oob
            if n_oob > 0:
                logger.debug(
                    f"[validate_gaia] Column '{col}': {n_oob} values outside "
                    f"[{vmin}, {vmax}]"
                )

    # Check label validity
    if "label_name" in df.columns:
        label_dist = df["label_name"].value_counts().to_dict()
        report["label_distribution"] = label_dist

        unknown = df[~df["label_name"].isin(CLASS_NAMES)]
        report["unknown_labels"] = len(unknown)
        if len(unknown) > 0:
            logger.warning(
                f"[validate_gaia] {len(unknown)} rows with unknown label_name: "
                f"{unknown['label_name'].unique()[:5]}"
            )

        logger.info(f"[validate_gaia] Label distribution: {label_dist}")

    # Check source_id uniqueness
    if "source_id" in df.columns:
        n_dup = df["source_id"].duplicated().sum()
        if n_dup > 0:
            logger.warning(f"[validate_gaia] {n_dup} duplicate source_ids")

    # Count valid rows (no NaN in critical columns)
    critical_cols = [c for c in ["source_id", "ra", "dec", "label_name"] if c in df.columns]
    if critical_cols:
        valid_mask = df[critical_cols].notna().all(axis=1)
        report["n_valid"] = int(valid_mask.sum())
    else:
        report["n_valid"] = len(df)

    logger.info(
        f"[validate_gaia] Validation complete: "
        f"{report['n_valid']}/{report['n_total']} valid rows, "
        f"passed={report['passed']}"
    )

    return report


def validate_ogle_params(
    df: pd.DataFrame,
    strict: bool = False,
) -> Dict[str, object]:
    """
    Validate OGLE stellar parameters table.

    Parameters
    ----------
    df : pd.DataFrame
        OGLE params table to validate
    strict : bool
        If True, raise ValueError on critical failures.

    Returns
    -------
    dict
        Validation report with keys:
        - "n_total": total number of rows
        - "n_valid": rows passing all checks
        - "missing_columns": list of missing required columns
        - "out_of_range": dict {col: n_violations}
        - "unknown_labels": number of rows with unrecognized label_name
        - "nan_fractions": dict {col: nan_fraction}
        - "label_distribution": dict {label_name: count}
        - "passed": bool

    Example
    -------
    >>> report = validate_ogle_params(ogle_df)
    >>> print(f"OGLE valid: {report['n_valid']}/{report['n_total']}")
    """
    report: Dict[str, object] = {
        "n_total": len(df),
        "n_valid": 0,
        "missing_columns": [],
        "out_of_range": {},
        "unknown_labels": 0,
        "nan_fractions": {},
        "label_distribution": {},
        "passed": True,
    }

    if len(df) == 0:
        logger.warning("[validate_ogle] Empty DataFrame")
        report["passed"] = False
        return report

    # Check required columns
    missing = [c for c in OGLE_REQUIRED_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing
    if missing:
        logger.warning(f"[validate_ogle] Missing required columns: {missing}")
        if strict:
            raise ValueError(f"OGLE params missing columns: {missing}")
        report["passed"] = False

    # NaN fractions
    for col in OGLE_REQUIRED_COLUMNS + ["period", "amplitude_i", "mean_mag_i"]:
        if col in df.columns:
            nan_frac = float(df[col].isna().mean())
            report["nan_fractions"][col] = nan_frac
            if nan_frac > 0.5:
                logger.warning(
                    f"[validate_ogle] Column '{col}' has {nan_frac:.1%} NaN values"
                )

    # Period range check
    if "period" in df.columns:
        p_min, p_max = PHYSICAL_PRIOR_RANGES["period"]
        valid_p = df["period"].notna()
        n_oob = int(
            ((df.loc[valid_p, "period"] < p_min) | (df.loc[valid_p, "period"] > p_max)).sum()
        )
        report["out_of_range"]["period"] = n_oob
        if n_oob > 0:
            logger.debug(
                f"[validate_ogle] period: {n_oob} values outside [{p_min}, {p_max}]"
            )

    # Check label validity
    if "label_name" in df.columns:
        label_dist = df["label_name"].value_counts().to_dict()
        report["label_distribution"] = label_dist

        unknown = df[~df["label_name"].isin(CLASS_NAMES)]
        report["unknown_labels"] = len(unknown)
        if len(unknown) > 0:
            logger.warning(
                f"[validate_ogle] {len(unknown)} rows with unknown label_name: "
                f"{unknown['label_name'].unique()[:5]}"
            )

        logger.info(f"[validate_ogle] Label distribution: {label_dist}")

    # Check ogle_id uniqueness
    if "ogle_id" in df.columns:
        n_dup = df["ogle_id"].duplicated().sum()
        if n_dup > 0:
            logger.warning(f"[validate_ogle] {n_dup} duplicate ogle_ids")

    # Count valid rows
    critical_cols = [c for c in ["ogle_id", "label_name"] if c in df.columns]
    if critical_cols:
        valid_mask = df[critical_cols].notna().all(axis=1)
        report["n_valid"] = int(valid_mask.sum())
    else:
        report["n_valid"] = len(df)

    logger.info(
        f"[validate_ogle] Validation complete: "
        f"{report['n_valid']}/{report['n_total']} valid rows, "
        f"passed={report['passed']}"
    )

    return report
