# cbm_variable_stars/features/fourier.py
"""
Fourier series decomposition for light curve feature extraction.

Extracts amplitude ratios (R21, R31) and phase differences (phi21).

Key correction (M10): phi21 is normalized to [0, 2*pi) range.
This matches the Gaia DR3 SOS pipeline convention (Clementini et al. 2023).

Import note: LightCurve is imported from extractor to avoid circular imports.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

# Avoid circular imports: import LightCurve from extractor
from cbm_variable_stars.features.extractor import LightCurve


def extract_fourier_coefficients(
    lc: LightCurve,
    period: float,
    n_harmonics: int = 7,
    fit_method: str = "linear_least_squares",
) -> tuple[float, float, float]:
    """
    Fourier series decomposition extracting R21, R31, phi21.

    Parameters
    ----------
    lc : LightCurve
        Light curve data
    period : float
        Period in days (from C1 extraction)
    n_harmonics : int
        Number of Fourier harmonics to fit. Recommended 7 (consistent with
        Gaia DR3 SOS pipeline). Auto-reduced if n_harmonics > (N_data-1)//2.
    fit_method : str
        "linear_least_squares" (recommended, fast) or
        "scipy_optimize" (more robust for noisy data)

    Returns
    -------
    tuple[float, float, float]
        - R21: A2/A1 (Fourier amplitude ratio, dimensionless, >= 0)
        - R31: A3/A1 (Fourier amplitude ratio, dimensionless, >= 0)
        - phi21: phi2 - 2*phi1, normalized to [0, 2*pi) radians [M10]

    Fourier model
    -------------
    m(t) = m0 + sum_{k=1}^{N} [a_k * cos(2*pi*k*t/P) + b_k * sin(2*pi*k*t/P)]

    where:
        A_k = sqrt(a_k^2 + b_k^2)
        phi_k = arctan2(b_k, a_k)        range (-pi, pi]
        R21 = A2 / A1
        phi21 = (phi2 - 2*phi1) % (2*pi)  range [0, 2*pi)  [M10]

    M10 correction notes
    --------------------
    phi_k = arctan2(b_k, a_k) has range (-pi, pi]
    phi21_raw = phi2 - 2*phi1 has range (-3*pi, 3*pi)
    Normalized: phi21 = phi21_raw % (2*pi) -> range [0, 2*pi)
    This is consistent with Gaia DR3 SOS and the extended concepts module.

    Example
    -------
    >>> R21, R31, phi21 = extract_fourier_coefficients(lc, period=0.5)
    >>> print(f"R21={R21:.3f}, R31={R31:.3f}, phi21={phi21:.3f} rad")
    """
    if period <= 0 or lc.n_obs < 5:
        return float("nan"), float("nan"), float("nan")

    phase = (lc.time / period) % 1.0

    # Auto-adjust harmonics to not exceed data constraints
    max_allowed = max(3, (lc.n_obs - 1) // 2)
    n_harm = min(n_harmonics, max_allowed)

    if n_harm < 2:
        return float("nan"), float("nan"), float("nan")

    if fit_method == "linear_least_squares":
        n = lc.n_obs
        # Design matrix: [1, cos(2pi*phase), sin(2pi*phase), cos(4pi*phase), ...]
        A_matrix = np.ones((n, 1 + 2 * n_harm))
        for k in range(1, n_harm + 1):
            A_matrix[:, 2 * k - 1] = np.cos(2 * np.pi * k * phase)
            A_matrix[:, 2 * k] = np.sin(2 * np.pi * k * phase)

        # Weighted least squares if errors are valid
        if np.all(lc.mag_err > 0):
            # [Min5 FIX] Use broadcasting instead of O(N²) diagonal matrix:
            # W_diag = 1/σ² → scale rows of A and mag by 1/σ
            w = 1.0 / lc.mag_err  # (N,)
            Aw = A_matrix * w[:, np.newaxis]  # row-scaled design matrix
            mag_w = lc.mag * w
            try:
                coeffs, _, _, _ = np.linalg.lstsq(Aw, mag_w, rcond=None)
            except np.linalg.LinAlgError:
                return float("nan"), float("nan"), float("nan")
        else:
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A_matrix, lc.mag, rcond=None)
            except np.linalg.LinAlgError:
                return float("nan"), float("nan"), float("nan")

    elif fit_method == "scipy_optimize":
        def fourier_model(params: np.ndarray, phase_data: np.ndarray, n_h: int) -> np.ndarray:
            result = np.full_like(phase_data, params[0])
            for k in range(1, n_h + 1):
                result += params[2 * k - 1] * np.cos(2 * np.pi * k * phase_data)
                result += params[2 * k] * np.sin(2 * np.pi * k * phase_data)
            return result

        def residual_func(params: np.ndarray) -> np.ndarray:
            model = fourier_model(params, phase, n_harm)
            err = lc.mag_err if np.all(lc.mag_err > 0) else np.ones_like(lc.mag)
            return (lc.mag - model) / err

        x0 = np.zeros(1 + 2 * n_harm)
        x0[0] = float(np.mean(lc.mag))
        try:
            result = least_squares(residual_func, x0, method="lm")
            coeffs = result.x
        except Exception:
            return float("nan"), float("nan"), float("nan")

    else:
        raise ValueError(
            f"Unknown fit_method: '{fit_method}'. "
            f"Valid: 'linear_least_squares', 'scipy_optimize'"
        )

    # Extract amplitudes and phases from coefficients
    # coeffs[0] = m0
    # coeffs[2k-1] = a_k (cosine), coeffs[2k] = b_k (sine)
    amplitudes = np.zeros(n_harm)
    phases_fourier = np.zeros(n_harm)
    for k in range(n_harm):
        a_k = float(coeffs[2 * k + 1])
        b_k = float(coeffs[2 * k + 2])
        amplitudes[k] = np.sqrt(a_k ** 2 + b_k ** 2)
        phases_fourier[k] = np.arctan2(b_k, a_k)

    A1 = amplitudes[0]
    if A1 < 1e-10:
        return float("nan"), float("nan"), float("nan")

    R21 = float(amplitudes[1] / A1) if len(amplitudes) > 1 else float("nan")
    R31 = float(amplitudes[2] / A1) if len(amplitudes) > 2 else float("nan")

    # [M10] phi21 normalized to [0, 2*pi)
    if len(phases_fourier) > 1:
        phi21_raw = phases_fourier[1] - 2.0 * phases_fourier[0]
        phi21 = float(phi21_raw % (2.0 * np.pi))
    else:
        phi21 = float("nan")

    return R21, R31, phi21


def extract_fourier_coefficients_extended(
    lc: LightCurve,
    period: float,
    n_harmonics: int = 7,
    fit_method: str = "linear_least_squares",
) -> dict[str, float]:
    """
    Extended Fourier decomposition returning additional coefficients for
    the 20-concept extended feature set.

    Parameters
    ----------
    lc : LightCurve
        Light curve data
    period : float
        Period in days
    n_harmonics : int
        Number of Fourier harmonics
    fit_method : str
        Fitting method

    Returns
    -------
    dict[str, float]
        {
            "R21": A2/A1,
            "R31": A3/A1,
            "phi21": (phi2 - 2*phi1) % (2*pi),
            "R41": A4/A1,
            "R51": A5/A1,
            "phi31": (phi3 - 3*phi1) % (2*pi),
            "phi41": (phi4 - 4*phi1) % (2*pi),
        }
        All phi values normalized to [0, 2*pi).
        NaN for harmonics not available (n_harm too small).

    Example
    -------
    >>> result = extract_fourier_coefficients_extended(lc, period=0.5)
    >>> print(result["R41"], result["phi31"])
    """
    nan_result = {
        k: float("nan")
        for k in ["R21", "R31", "phi21", "R41", "R51", "phi31", "phi41"]
    }

    if period <= 0 or lc.n_obs < 5:
        return nan_result

    phase = (lc.time / period) % 1.0

    max_allowed = max(3, (lc.n_obs - 1) // 2)
    n_harm = min(n_harmonics, max_allowed)

    if n_harm < 2:
        return nan_result

    # Build design matrix
    n = lc.n_obs
    A_matrix = np.ones((n, 1 + 2 * n_harm))
    for k in range(1, n_harm + 1):
        A_matrix[:, 2 * k - 1] = np.cos(2 * np.pi * k * phase)
        A_matrix[:, 2 * k] = np.sin(2 * np.pi * k * phase)

    # Least squares fit
    if np.all(lc.mag_err > 0):
        # [Min5 FIX] Use broadcasting instead of O(N²) diagonal matrix
        w = 1.0 / lc.mag_err  # (N,)
        Aw = A_matrix * w[:, np.newaxis]
        mag_w = lc.mag * w
        try:
            coeffs, _, _, _ = np.linalg.lstsq(Aw, mag_w, rcond=None)
        except np.linalg.LinAlgError:
            return nan_result
    else:
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_matrix, lc.mag, rcond=None)
        except np.linalg.LinAlgError:
            return nan_result

    # Extract amplitudes and phases
    amplitudes = np.zeros(n_harm)
    phases_fourier = np.zeros(n_harm)
    for k in range(n_harm):
        a_k = float(coeffs[2 * k + 1])
        b_k = float(coeffs[2 * k + 2])
        amplitudes[k] = np.sqrt(a_k ** 2 + b_k ** 2)
        phases_fourier[k] = np.arctan2(b_k, a_k)

    A1 = amplitudes[0]
    if A1 < 1e-10:
        return nan_result

    result = {}
    result["R21"] = float(amplitudes[1] / A1) if n_harm > 1 else float("nan")
    result["R31"] = float(amplitudes[2] / A1) if n_harm > 2 else float("nan")
    result["R41"] = float(amplitudes[3] / A1) if n_harm > 3 else float("nan")
    result["R51"] = float(amplitudes[4] / A1) if n_harm > 4 else float("nan")

    # All phi values normalized to [0, 2*pi)
    result["phi21"] = (
        float((phases_fourier[1] - 2.0 * phases_fourier[0]) % (2.0 * np.pi))
        if n_harm > 1
        else float("nan")
    )
    result["phi31"] = (
        float((phases_fourier[2] - 3.0 * phases_fourier[0]) % (2.0 * np.pi))
        if n_harm > 2
        else float("nan")
    )
    result["phi41"] = (
        float((phases_fourier[3] - 4.0 * phases_fourier[0]) % (2.0 * np.pi))
        if n_harm > 3
        else float("nan")
    )

    return result
