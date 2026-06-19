# cbm_variable_stars/features/amplitude.py
"""
Amplitude and rise time fraction extraction from phase-folded light curves.

C2: amplitude = percentile_high - percentile_low of the magnitude distribution
C3: rise_fraction = fraction of the phase cycle spent rising (brightening)
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cbm_variable_stars.features.extractor import LightCurve


def extract_amplitude_and_rise_time(
    lc: "LightCurve",
    period: float,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    n_phase_bins: int = 20,
    smoothing_window: int = 3,
) -> tuple[float, float]:
    """
    Extract amplitude (C2) and rise time fraction (C3) from a light curve.

    Parameters
    ----------
    lc : LightCurve
        Light curve data container (time, mag, mag_err, band, source_id)
    period : float
        Best period in days (from C1 extraction)
    percentile_low : float
        Lower percentile for amplitude calculation (default: 5.0)
        Magnitude at this percentile corresponds to bright end
    percentile_high : float
        Upper percentile for amplitude calculation (default: 95.0)
        Magnitude at this percentile corresponds to faint end
    n_phase_bins : int
        Number of phase bins for binned light curve (default: 20)
    smoothing_window : int
        Window size for moving average smoothing of binned curve (default: 3)
        Must be odd; if even, incremented by 1.

    Returns
    -------
    tuple[float, float]
        - amplitude: mag(percentile_high) - mag(percentile_low) in magnitudes.
          Higher = larger variability. NaN if extraction fails.
        - rise_fraction: fraction of phase cycle spent brightening (0 to 1).
          Defined as the phase interval where the smoothed magnitude is
          decreasing (smaller mag = brighter). NaN if extraction fails.

    Notes
    -----
    Amplitude definition:
        amplitude = np.percentile(mag, percentile_high) - np.percentile(mag, percentile_low)
        This is robust to outliers compared to max-min.

    Rise fraction definition:
        Phase-fold the light curve. Bin into n_phase_bins.
        Smooth the binned curve. Count bins where the magnitude is
        decreasing (star is brightening). rise_fraction = n_rising / n_bins.

        Convention: larger magnitude = fainter star
        "Rising" = magnitude decreasing = star getting brighter

    Example
    -------
    >>> from cbm_variable_stars.features.extractor import LightCurve
    >>> lc = LightCurve(time=t, mag=m, mag_err=e, band="G")
    >>> amp, rise_frac = extract_amplitude_and_rise_time(lc, period=0.5)
    >>> print(f"Amplitude: {amp:.3f} mag, Rise fraction: {rise_frac:.3f}")
    """
    if len(lc.mag) < 5:
        return float("nan"), float("nan")

    # ---- C2: Amplitude ----
    try:
        mag_lo = float(np.nanpercentile(lc.mag, percentile_low))
        mag_hi = float(np.nanpercentile(lc.mag, percentile_high))
        amplitude = mag_hi - mag_lo
        if amplitude < 0:
            amplitude = float("nan")
    except Exception:
        amplitude = float("nan")

    # ---- C3: Rise fraction ----
    try:
        if period <= 0:
            return amplitude, float("nan")

        # Phase fold
        phase, mag_sorted, _ = lc.phase_fold(period, epoch=0.0)

        if len(phase) < n_phase_bins:
            # Not enough points for binning
            return amplitude, float("nan")

        # Bin the phase-folded light curve
        bin_edges = np.linspace(0.0, 1.0, n_phase_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_mags = np.full(n_phase_bins, float("nan"))

        for i in range(n_phase_bins):
            mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_mags[i] = float(np.nanmedian(mag_sorted[mask]))

        # Interpolate NaN bins (linear)
        valid_bins = ~np.isnan(bin_mags)
        if valid_bins.sum() < 3:
            return amplitude, float("nan")

        bin_idx = np.arange(n_phase_bins)
        bin_mags_interp = np.interp(
            bin_idx,
            bin_idx[valid_bins],
            bin_mags[valid_bins],
        )

        # Smooth with moving average
        if smoothing_window % 2 == 0:
            smoothing_window += 1

        if smoothing_window >= n_phase_bins:
            smoothing_window = max(3, n_phase_bins // 3)
            if smoothing_window % 2 == 0:
                smoothing_window += 1

        half_w = smoothing_window // 2
        smoothed = np.convolve(
            np.concatenate([
                bin_mags_interp[-half_w:],
                bin_mags_interp,
                bin_mags_interp[:half_w],
            ]),
            np.ones(smoothing_window) / smoothing_window,
            mode="valid",
        )[:n_phase_bins]

        # Rise = magnitude decreasing (star brightening)
        # diff[i] = smoothed[i+1] - smoothed[i]
        diffs = np.diff(smoothed)
        n_rising = int((diffs < 0).sum())
        rise_fraction = float(n_rising) / float(len(diffs))

    except Exception:
        rise_fraction = float("nan")

    return amplitude, rise_fraction
