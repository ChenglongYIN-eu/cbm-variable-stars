# cbm_variable_stars/data/alias_detection.py
"""
Gaia satellite scan period alias detection module.

The Gaia satellite has a precession period of approximately 63 days,
which creates alias peaks in Lomb-Scargle periodograms. This module
automatically flags periods close to 1/63 day (or its harmonics).

Reference: Holl et al. (2023) Gaia DR3 Variability Processing.
"""

from __future__ import annotations
import numpy as np
from cbm_variable_stars.shared.constants import (
    GAIA_ALIAS_FREQUENCIES,
    ALIAS_FREQUENCY_TOLERANCE,
    GAIA_PRECESSION_PERIOD_DAYS,
)
from cbm_variable_stars.shared.logger import logger


def check_period_alias(
    period: float,
    ls_power: np.ndarray,
    frequency_grid: np.ndarray,
    alias_frequencies: list[float] | None = None,
    frequency_tolerance: float = ALIAS_FREQUENCY_TOLERANCE,
    secondary_peak_threshold: float = 0.8,
) -> bool:
    """
    Detect if the extracted period may be a Gaia scan alias.

    Parameters
    ----------
    period : float
        Extracted best period in days
    ls_power : np.ndarray, shape (M,)
        Lomb-Scargle power spectrum
    frequency_grid : np.ndarray, shape (M,)
        Corresponding frequency grid in day^-1
    alias_frequencies : list[float] or None
        Known alias frequencies in day^-1.
        Defaults to GAIA_ALIAS_FREQUENCIES (63-day precession and harmonics)
    frequency_tolerance : float
        Frequency matching tolerance in day^-1, default 0.002
    secondary_peak_threshold : float
        Secondary-to-primary peak power ratio threshold.
        If the primary peak is at an alias frequency and the secondary
        peak power > threshold * primary peak power, the period is
        flagged as a potential alias.

    Returns
    -------
    bool
        True = likely alias period, recommend manual inspection
        False = likely genuine period

    Detection logic
    ---------------
    1. Convert period to frequency: f_best = 1/period
    2. Check if f_best is within tolerance of any known alias frequency
    3. If yes, check for significant secondary peaks:
       - Mask the primary peak region (+/- 5*df)
       - Find the secondary peak
       - If secondary / primary > threshold, flag as alias
    4. Also check 2*f_best (half-period alias, common for eclipsing binaries)

    Conservative strategy: if the period is near an alias frequency,
    flag it even if secondary peak is not significant.

    Example
    -------
    >>> is_alias = check_period_alias(
    ...     period=63.1,
    ...     ls_power=power_spectrum,
    ...     frequency_grid=freq_grid,
    ... )
    >>> if is_alias:
    ...     logger.warning("Possible 63-day precession alias!")
    """
    if alias_frequencies is None:
        alias_frequencies = GAIA_ALIAS_FREQUENCIES

    if period <= 0:
        return False

    f_best = 1.0 / period

    # Step 1: Check proximity to known alias frequencies
    is_near_alias = False
    matched_alias_freq = None

    for f_alias in alias_frequencies:
        if abs(f_best - f_alias) < frequency_tolerance:
            is_near_alias = True
            matched_alias_freq = f_alias
            break
        # Also check half-period alias (eclipsing binary half-period confusion)
        if abs(2 * f_best - f_alias) < frequency_tolerance:
            is_near_alias = True
            matched_alias_freq = f_alias
            break

    if not is_near_alias:
        return False

    # Step 2: Check for significant secondary peaks
    best_idx = int(np.argmax(ls_power))
    best_power = float(ls_power[best_idx])

    if best_power <= 0:
        return True  # Conservative: near alias freq

    # Estimate frequency grid spacing
    if len(frequency_grid) > 10:
        df_grid = float(np.median(np.diff(frequency_grid)))
        # Mask width: 5 frequency grid steps around the peak
        mask_width = max(2, int(5.0 * frequency_tolerance / df_grid))
    else:
        mask_width = 2

    masked_power = ls_power.copy().astype(float)
    low = max(0, best_idx - mask_width)
    high = min(len(masked_power), best_idx + mask_width + 1)
    masked_power[low:high] = 0.0

    max_secondary = float(np.max(masked_power))

    if max_secondary > 0:
        ratio = max_secondary / best_power
        if ratio > secondary_peak_threshold:
            logger.debug(
                f"[alias_detection] period={period:.4f}d, f={f_best:.5f}, "
                f"matched alias f={matched_alias_freq:.5f}, "
                f"secondary/primary={ratio:.3f} > {secondary_peak_threshold} "
                f"-> ALIAS FLAGGED"
            )
            return True

    # Near alias frequency but weak secondary peak -> flag conservatively
    logger.debug(
        f"[alias_detection] period={period:.4f}d near alias f={matched_alias_freq:.5f}, "
        f"conservative flag (no strong secondary)"
    )
    return True


def suggest_dealiased_period(
    period: float,
    ls_power: np.ndarray,
    frequency_grid: np.ndarray,
    alias_frequencies: list[float] | None = None,
    frequency_tolerance: float = ALIAS_FREQUENCY_TOLERANCE,
) -> float:
    """
    Suggest a de-aliased period for a flagged alias period.

    Parameters
    ----------
    period : float
        Originally extracted period in days (may be alias)
    ls_power : np.ndarray
        Lomb-Scargle power spectrum
    frequency_grid : np.ndarray
        Frequency grid in day^-1
    alias_frequencies : list[float] or None
        Known alias frequencies in day^-1
    frequency_tolerance : float
        Frequency masking half-width in day^-1

    Returns
    -------
    float
        Suggested de-aliased period in days.
        Returns the original period if de-aliasing is not possible
        (e.g., all power is concentrated at alias frequencies).

    Algorithm
    ---------
    Mask all alias frequency regions (+/- tolerance) in the power spectrum.
    Also mask half-alias frequencies. Find the highest peak in the
    remaining (unmasked) power spectrum, and return its period.

    Example
    -------
    >>> new_period = suggest_dealiased_period(63.1, power, freq)
    >>> print(f"De-aliased: {63.1:.2f}d -> {new_period:.4f}d")
    """
    if alias_frequencies is None:
        alias_frequencies = GAIA_ALIAS_FREQUENCIES

    masked_power = ls_power.copy().astype(float)

    for f_alias in alias_frequencies:
        # Mask alias frequency region
        mask = np.abs(frequency_grid - f_alias) < frequency_tolerance
        masked_power[mask] = 0.0

        # Also mask half-frequency (eclipsing binary half-period)
        if f_alias > 0:
            mask_half = np.abs(frequency_grid - f_alias / 2.0) < frequency_tolerance
            masked_power[mask_half] = 0.0

    max_remaining = float(np.max(masked_power))
    if max_remaining > 0:
        new_idx = int(np.argmax(masked_power))
        new_freq = float(frequency_grid[new_idx])
        if new_freq > 0:
            new_period = 1.0 / new_freq
            logger.debug(
                f"[alias_detection] De-aliased: {period:.4f}d -> {new_period:.4f}d "
                f"(f: {1.0/period:.5f} -> {new_freq:.5f})"
            )
            return new_period

    # Cannot de-alias
    logger.debug(
        f"[alias_detection] Cannot de-alias period={period:.4f}d, "
        f"all power at alias frequencies"
    )
    return period
