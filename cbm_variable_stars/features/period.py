# cbm_variable_stars/features/period.py
"""
Period extraction using generalized Lomb-Scargle periodogram.

Returns the best period, false alarm probability, and the full power spectrum
and frequency grid (for alias detection in alias_detection.py).

Key: C10 = period_snr = -log10(FAP) transformation is applied in extractor.py,
     NOT here. This function returns the raw FAP value.
"""

from __future__ import annotations
import numpy as np
from astropy.timeseries import LombScargle


def extract_period_and_fap(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: np.ndarray,
    min_frequency: float = 0.01,
    max_frequency: float = 25.0,
    samples_per_peak: int = 10,
    nyquist_factor: int = 5,
    fit_mean: bool = True,
    fap_method: str = "baluev",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Extract the best period and false alarm probability using Lomb-Scargle.

    Parameters
    ----------
    time : np.ndarray, shape (N,)
        Observation times in days
    mag : np.ndarray, shape (N,)
        Magnitudes
    mag_err : np.ndarray, shape (N,)
        Magnitude errors
    min_frequency : float
        Minimum search frequency in day^-1 (default: 0.01 = 100-day max period)
    max_frequency : float
        Maximum search frequency in day^-1 (default: 25.0 = 0.04-day min period)
    samples_per_peak : int
        Frequency grid oversampling factor, recommended 10
    nyquist_factor : int
        Factor beyond pseudo-Nyquist frequency
    fit_mean : bool
        Fit mean offset (generalized LS), recommended True
    fap_method : str
        "baluev" (analytic, fast) or "bootstrap" (Monte Carlo, slow)

    Returns
    -------
    tuple[float, float, np.ndarray, np.ndarray]
        - period: Best period in days
        - fap: False alarm probability (raw value, 0-1)
        - power: Full power spectrum, shape (M,)
        - frequency: Frequency grid, shape (M,) in day^-1
          [New] Also returned for use by alias_detection module

    Raises
    ------
    ValueError
        If fewer than 10 data points or empty frequency grid

    Notes
    -----
    Uses astropy.timeseries.LombScargle with standard normalization.
    The frequency grid is constructed from baseline-based df = 1/(spp*T).

    C10 transformation:
        period_snr = -log10(FAP)  [applied in extractor.py, NOT here]
        FAP=0 => period_snr = 300.0 (cap)

    Example
    -------
    >>> period, fap, power, freq = extract_period_and_fap(time, mag, mag_err)
    >>> period_snr = -np.log10(fap) if fap > 0 else 300.0
    """
    if len(time) < 10:
        raise ValueError(f"Insufficient data points: {len(time)} < 10")

    # Build frequency grid
    T_baseline = float(time.max() - time.min())
    if T_baseline <= 0:
        raise ValueError(f"Zero or negative time baseline: {T_baseline}")

    df = 1.0 / (samples_per_peak * T_baseline)
    frequency = np.arange(min_frequency, max_frequency, df)

    if len(frequency) == 0:
        raise ValueError(
            f"Empty frequency grid: min={min_frequency}, "
            f"max={max_frequency}, df={df}"
        )

    # Generalized Lomb-Scargle
    ls = LombScargle(
        time,
        mag,
        mag_err,
        fit_mean=fit_mean,
        center_data=True,
        normalization="standard",
    )

    power = ls.power(frequency)

    # Best period
    best_idx = int(np.argmax(power))
    best_frequency = float(frequency[best_idx])
    best_period = 1.0 / best_frequency

    # False alarm probability
    if fap_method in ("baluev", "bootstrap"):
        try:
            fap = float(
                ls.false_alarm_probability(
                    power[best_idx],
                    method=fap_method,
                    maximum_frequency=max_frequency,
                    minimum_frequency=min_frequency,
                )
            )
            # Clamp to valid range
            fap = max(0.0, min(1.0, fap))
        except Exception:
            fap = float("nan")
    else:
        fap = float("nan")

    return best_period, fap, power, frequency
