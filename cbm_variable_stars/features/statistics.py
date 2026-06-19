# cbm_variable_stars/features/statistics.py
"""
Statistical features extracted from magnitude distributions.

C7: skewness  - asymmetry of magnitude distribution
C8: kurtosis  - tail heaviness (Fisher excess kurtosis)
C9: stetson_K - Stetson K variability index (robust scatter measure)
"""

from __future__ import annotations
import numpy as np
from scipy import stats as scipy_stats


def extract_statistical_features(
    mag: np.ndarray,
    mag_err: np.ndarray,
    robust: bool = False,
    use_errors: bool = True,
) -> tuple[float, float, float]:
    """
    Extract skewness, kurtosis, and Stetson K from a magnitude array.

    Parameters
    ----------
    mag : np.ndarray, shape (N,)
        Magnitude values
    mag_err : np.ndarray, shape (N,)
        Magnitude errors (used for weighted statistics and Stetson K)
    robust : bool
        If True, use robust statistics (median-based) instead of
        moment-based statistics for skewness and kurtosis.
        Default False (use standard moment estimators).
    use_errors : bool
        If True, weight statistics by 1/mag_err^2 and use errors
        in Stetson K computation. Default True.

    Returns
    -------
    tuple[float, float, float]
        - skewness: Third standardized moment (Fisher definition).
          Positive = right tail (more faint-state points).
          Negative = left tail (more bright-state points).
        - kurtosis: Fourth standardized moment minus 3 (excess kurtosis,
          Fisher definition). 0 for normal distribution.
          Positive = heavier tails than normal.
        - stetson_K: Stetson K index, measuring variability scatter.
          Defined as: (1/sqrt(N)) * sum(|delta_i|) / sqrt(sum(delta_i^2))
          where delta_i = sqrt(N/(N-1)) * (mag_i - mean_mag) / mag_err_i
          Range approximately [0, 1] for normal-like distributions.
          Stetson (1996) PASP 108, 851.

    Notes
    -----
    For Stetson K, when use_errors=False, we use uniform weights.

    If fewer than 3 data points, returns NaN for all.

    Example
    -------
    >>> skew, kurt, stet_k = extract_statistical_features(mag, mag_err)
    >>> print(f"Skew={skew:.3f}, Kurt={kurt:.3f}, K={stet_k:.3f}")
    """
    n = len(mag)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    # Remove NaN values
    valid = ~(np.isnan(mag) | np.isnan(mag_err))
    mag = mag[valid]
    mag_err = mag_err[valid]
    n = len(mag)

    if n < 3:
        return float("nan"), float("nan"), float("nan")

    # ---- Skewness (C7) ----
    try:
        if robust:
            # Robust skewness: Bowley skewness using quartiles
            q1, q2, q3 = np.percentile(mag, [25, 50, 75])
            iqr = q3 - q1
            if iqr > 0:
                skewness = float((q1 + q3 - 2 * q2) / iqr)
            else:
                skewness = 0.0
        else:
            if use_errors and np.all(mag_err > 0):
                # Weighted skewness
                weights = 1.0 / (mag_err ** 2)
                w_sum = weights.sum()
                mean_w = float(np.sum(weights * mag) / w_sum)
                deviations = mag - mean_w
                var_w = float(np.sum(weights * deviations ** 2) / w_sum)
                std_w = np.sqrt(var_w)
                if std_w > 0:
                    skewness = float(
                        np.sum(weights * deviations ** 3) / (w_sum * std_w ** 3)
                    )
                else:
                    skewness = 0.0
            else:
                skewness = float(scipy_stats.skew(mag, bias=False))
    except Exception:
        skewness = float("nan")

    # ---- Kurtosis (C8) ----
    try:
        if robust:
            # Robust kurtosis: ratio of octile width to quartile width
            o1, o7 = np.percentile(mag, [12.5, 87.5])
            q1, q3 = np.percentile(mag, [25, 75])
            iqr = q3 - q1
            octile_range = o7 - o1
            if iqr > 0:
                kurtosis = float(octile_range / iqr - 1.23)  # normalized
            else:
                kurtosis = 0.0
        else:
            if use_errors and np.all(mag_err > 0):
                # Weighted excess kurtosis
                weights = 1.0 / (mag_err ** 2)
                w_sum = weights.sum()
                mean_w = float(np.sum(weights * mag) / w_sum)
                deviations = mag - mean_w
                var_w = float(np.sum(weights * deviations ** 2) / w_sum)
                std_w = np.sqrt(var_w)
                if std_w > 0:
                    kurtosis = float(
                        np.sum(weights * deviations ** 4) / (w_sum * std_w ** 4) - 3.0
                    )
                else:
                    kurtosis = 0.0
            else:
                # Fisher excess kurtosis (scipy default)
                kurtosis = float(scipy_stats.kurtosis(mag, fisher=True, bias=False))
    except Exception:
        kurtosis = float("nan")

    # ---- Stetson K (C9) ----
    try:
        if n < 2:
            stetson_k = float("nan")
        else:
            mean_mag = float(np.mean(mag))

            if use_errors and np.all(mag_err > 0):
                # Standard Stetson K with error weighting
                # delta_i = sqrt(N/(N-1)) * (mag_i - mean_mag) / mag_err_i
                delta = np.sqrt(float(n) / (n - 1.0)) * (mag - mean_mag) / mag_err
            else:
                # Unweighted version: use sample std as denominator
                std_mag = float(np.std(mag, ddof=1))
                if std_mag > 0:
                    delta = np.sqrt(float(n) / (n - 1.0)) * (mag - mean_mag) / std_mag
                else:
                    stetson_k = 0.0
                    return skewness, kurtosis, stetson_k

            sum_abs_delta = float(np.sum(np.abs(delta)))
            sum_sq_delta = float(np.sum(delta ** 2))

            if sum_sq_delta > 0:
                stetson_k = float(
                    (1.0 / np.sqrt(float(n))) * sum_abs_delta / np.sqrt(sum_sq_delta)
                )
            else:
                stetson_k = 0.0

    except Exception:
        stetson_k = float("nan")

    return skewness, kurtosis, stetson_k
