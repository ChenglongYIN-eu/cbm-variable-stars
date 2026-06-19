# cbm_variable_stars/features/photometric.py
"""
Photometric features extracted from Gaia metadata and light curves.

C11: color_bp_rp - Gaia BP-RP color index (mag)
C12: mean_mag    - Mean G-band magnitude (mag)
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from cbm_variable_stars.features.extractor import LightCurve

from cbm_variable_stars.shared.logger import logger


def extract_photometric_features(
    metadata: dict,
    lc: "LightCurve",
    mean_mag_method: str = "weighted_mean",
) -> tuple[float, float]:
    """
    Extract color index (C11) and mean magnitude (C12) from metadata and light curve.

    Parameters
    ----------
    metadata : dict
        Source metadata dictionary. Expected keys:
        - "bp_rp": Gaia BP-RP color (preferred source for C11)
        - "phot_g_mean_mag": Gaia G-band mean magnitude (alternative for C12)
        - "phot_bp_mean_mag", "phot_rp_mean_mag": BP and RP mean mags
          (used if bp_rp not directly available)
    lc : LightCurve
        Light curve data (used as fallback for mean_mag if metadata missing)
    mean_mag_method : str
        Method to compute mean magnitude from the light curve:
        - "weighted_mean": Inverse-variance weighted mean (default)
        - "median": Robust median
        - "simple_mean": Unweighted mean

    Returns
    -------
    tuple[float, float]
        - color_bp_rp: BP-RP color index in magnitudes.
          NaN if not available in metadata and cannot be computed.
        - mean_mag: Mean magnitude. Prefers metadata value, falls back
          to light curve computation if metadata value is missing.

    Notes
    -----
    For OGLE sources:
        - color_bp_rp is NaN unless cross-matched with Gaia (see ogle_crossband.py)
        - mean_mag is computed from the OGLE I-band light curve

    For Gaia sources:
        - color_bp_rp comes from gaia_source.bp_rp
        - mean_mag comes from gaia_source.phot_g_mean_mag or light curve

    Example
    -------
    >>> meta = {"bp_rp": 1.2, "phot_g_mean_mag": 15.3}
    >>> color, mean_mag = extract_photometric_features(meta, lc)
    >>> print(f"BP-RP={color:.3f}, G={mean_mag:.3f}")
    """
    # ---- C11: color_bp_rp ----
    color_bp_rp = float("nan")

    # Try direct bp_rp first
    if "bp_rp" in metadata and metadata["bp_rp"] is not None:
        val = metadata["bp_rp"]
        if not _is_nan_like(val):
            color_bp_rp = float(val)

    # Compute from individual band mags if bp_rp not available
    if np.isnan(color_bp_rp):
        bp = metadata.get("phot_bp_mean_mag")
        rp = metadata.get("phot_rp_mean_mag")
        if bp is not None and rp is not None:
            if not _is_nan_like(bp) and not _is_nan_like(rp):
                color_bp_rp = float(bp) - float(rp)

    # ---- C12: mean_mag ----
    mean_mag = float("nan")

    # Try metadata first
    for meta_key in ["phot_g_mean_mag", "mean_mag", "i_mean", "mean_mag_i"]:
        if meta_key in metadata and metadata[meta_key] is not None:
            val = metadata[meta_key]
            if not _is_nan_like(val):
                mean_mag = float(val)
                break

    # Fall back to computing from light curve
    if np.isnan(mean_mag) and lc is not None and len(lc.mag) > 0:
        try:
            valid = ~np.isnan(lc.mag)
            if valid.sum() > 0:
                mag_valid = lc.mag[valid]
                err_valid = lc.mag_err[valid] if lc.mag_err is not None else None

                if mean_mag_method == "weighted_mean" and err_valid is not None:
                    err_clean = err_valid[~np.isnan(err_valid)]
                    mag_clean = mag_valid[~np.isnan(err_valid)]
                    if len(mag_clean) > 0 and np.all(err_clean > 0):
                        weights = 1.0 / (err_clean ** 2)
                        mean_mag = float(np.sum(weights * mag_clean) / np.sum(weights))
                    else:
                        mean_mag = float(np.median(mag_valid))
                elif mean_mag_method == "median":
                    mean_mag = float(np.median(mag_valid))
                else:
                    mean_mag = float(np.mean(mag_valid))
        except Exception as e:
            logger.debug(f"[photometric] mean_mag computation failed: {e}")

    return color_bp_rp, mean_mag


def _is_nan_like(val: object) -> bool:
    """Check if a value is NaN, None, or otherwise invalid."""
    if val is None:
        return True
    try:
        return np.isnan(float(val))
    except (TypeError, ValueError):
        return True
