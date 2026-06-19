# cbm_variable_stars/features/extractor.py
"""
Main feature extraction module for CBM Variable Star Classification.

Contains the LightCurve dataclass and batch extraction pipeline.

Key corrections:
    [S1]   Concept names from CONCEPT_NAMES_12 constants (snake_case, unified)
    [C10]  period_snr = -log10(FAP), positive value, higher = more significant
    [M5]   Gaia 'time' column required; no fallback to transit_id
    [M10]  phi21 normalized to [0, 2*pi) in fourier.py
    [I6]   Alias detection flag added
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig

from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12


@dataclass
class LightCurve:
    """
    Light curve data container.

    Attributes
    ----------
    time : np.ndarray
        Observation times (BJD or HJD), shape (N,)
    mag : np.ndarray
        Magnitudes, shape (N,)
    mag_err : np.ndarray
        Magnitude errors, shape (N,)
    band : str
        Band identifier: "G" (Gaia), "I" (OGLE I), "V" (OGLE V)
    source_id : str
        Unique source identifier (Gaia source_id or OGLE ID)
    """
    time: np.ndarray
    mag: np.ndarray
    mag_err: np.ndarray
    band: str = "G"
    source_id: str = ""

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return len(self.time)

    def phase_fold(
        self,
        period: float,
        epoch: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Phase-fold the light curve.

        Parameters
        ----------
        period : float
            Period in days
        epoch : float
            Reference epoch in days (default: 0.0)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (phase, mag, mag_err), phase in [0, 1), sorted by ascending phase
        """
        phase = ((self.time - epoch) / period) % 1.0
        sort_idx = np.argsort(phase)
        return phase[sort_idx], self.mag[sort_idx], self.mag_err[sort_idx]


def extract_all_features(
    lc: LightCurve,
    metadata: dict,
    cfg: DictConfig,
    var_type: str = "",
) -> dict[str, float]:
    """
    Extract all 12 concept features from a single light curve.

    Parameters
    ----------
    lc : LightCurve
        Light curve data
    metadata : dict
        Source metadata (contains bp_rp, phot_g_mean_mag, etc.)
    cfg : DictConfig
        Feature extraction configuration (cfg.features section)
    var_type : str
        Variable star type (used to adjust frequency range for MIRA_SR)

    Returns
    -------
    dict[str, float]
        {concept_name: value}, concept names strictly from CONCEPT_NAMES_12.
        NaN indicates extraction failure.
        Extra key: "alias_flag" (bool) marks suspected alias periods.

    Pipeline
    --------
    1. Period extraction (C1 period, C10 period_snr)
    2. Alias detection (alias_flag)
    3. Phase-fold -> amplitude (C2) + rise_fraction (C3)
    4. Fourier decomposition -> R21 (C4) + R31 (C5) + phi21 (C6)
    5. Statistical features -> skewness (C7) + kurtosis (C8) + stetson_K (C9)
    6. Photometric features -> color_bp_rp (C11) + mean_mag (C12)

    Key corrections
    ---------------
    [S1]   Concept names from CONCEPT_NAMES_12
    [C10]  period_snr = -log10(FAP), positive, higher = more significant
    [M10]  phi21 normalized to [0, 2*pi) in fourier.py
    [I6]   alias_flag added
    """
    # Import here to avoid circular imports at module load time
    from cbm_variable_stars.features.period import extract_period_and_fap
    from cbm_variable_stars.features.amplitude import extract_amplitude_and_rise_time
    from cbm_variable_stars.features.fourier import extract_fourier_coefficients
    from cbm_variable_stars.features.statistics import extract_statistical_features
    from cbm_variable_stars.features.photometric import extract_photometric_features
    from cbm_variable_stars.data.alias_detection import check_period_alias

    features: dict[str, object] = {name: np.nan for name in CONCEPT_NAMES_12}
    features["alias_flag"] = False

    if lc.n_obs < 10:
        logger.debug(f"[{lc.source_id}] Insufficient data: {lc.n_obs} < 10")
        return features

    # ---- Step 1: Period extraction ----
    try:
        min_freq = cfg.period.min_frequency
        max_freq = cfg.period.max_frequency

        # MIRA_SR has much longer periods -> lower frequency range
        if var_type == "MIRA_SR":
            min_freq = 0.002   # up to 500-day periods
            max_freq = 5.0

        period, fap, ls_power, frequency_grid = extract_period_and_fap(
            time=lc.time,
            mag=lc.mag,
            mag_err=lc.mag_err,
            min_frequency=min_freq,
            max_frequency=max_freq,
            samples_per_peak=cfg.period.samples_per_peak,
            nyquist_factor=cfg.period.nyquist_factor,
            fit_mean=cfg.period.fit_mean,
            fap_method=cfg.fap.method,
        )

        features["period"] = period

        # [S1/C10] period_snr = -log10(FAP), positive, higher = more significant
        if fap > 0:
            features["period_snr"] = float(-np.log10(fap))
        else:
            features["period_snr"] = 300.0  # FAP=0 -> extremely significant

        # [I6] Alias detection
        alias_enabled = False
        if hasattr(cfg, "alias_detection"):
            alias_enabled = cfg.alias_detection.get("enabled", False)
        if alias_enabled:
            features["alias_flag"] = check_period_alias(
                period=period,
                ls_power=ls_power,
                frequency_grid=frequency_grid,
            )

    except Exception as e:
        logger.debug(f"[{lc.source_id}] Period extraction failed: {e}")

    # ---- Step 2: Amplitude and rise time ----
    if not np.isnan(features["period"]):
        try:
            amplitude, rise_frac = extract_amplitude_and_rise_time(
                lc=lc,
                period=float(features["period"]),
                percentile_low=cfg.amplitude.percentile_low,
                percentile_high=cfg.amplitude.percentile_high,
                n_phase_bins=cfg.rise_time.n_phase_bins,
                smoothing_window=cfg.rise_time.smoothing_window,
            )
            features["amplitude"] = amplitude
            features["rise_fraction"] = rise_frac
        except Exception as e:
            logger.debug(f"[{lc.source_id}] Amplitude/rise time failed: {e}")

    # ---- Step 3: Fourier decomposition ----
    if not np.isnan(features["period"]):
        try:
            R21, R31, phi21 = extract_fourier_coefficients(
                lc=lc,
                period=float(features["period"]),
                n_harmonics=cfg.fourier.n_harmonics,
                fit_method=cfg.fourier.fit_method,
            )
            features["R21"] = R21
            features["R31"] = R31
            # [M10] phi21 already normalized to [0, 2*pi) inside fourier.py
            features["phi21"] = phi21
        except Exception as e:
            logger.debug(f"[{lc.source_id}] Fourier decomposition failed: {e}")

    # ---- Step 4: Statistical features ----
    try:
        skewness, kurtosis, stetson_k = extract_statistical_features(
            mag=lc.mag,
            mag_err=lc.mag_err,
            robust=cfg.moments.robust,
            use_errors=cfg.stetson.use_errors,
        )
        features["skewness"] = skewness
        features["kurtosis"] = kurtosis
        features["stetson_K"] = stetson_k
    except Exception as e:
        logger.debug(f"[{lc.source_id}] Statistical features failed: {e}")

    # ---- Step 5: Photometric features ----
    try:
        color, mean_mag = extract_photometric_features(
            metadata=metadata,
            lc=lc,
            mean_mag_method=cfg.mean_mag.method,
        )
        features["color_bp_rp"] = color
        features["mean_mag"] = mean_mag
    except Exception as e:
        logger.debug(f"[{lc.source_id}] Photometric features failed: {e}")

    return features


def extract_features_batch(
    metadata_df: pd.DataFrame,
    lc_dir: str | Path,
    cfg: DictConfig,
    source: str = "gaia",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Batch extraction of features for all sources.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Metadata table
    lc_dir : str or Path
        Directory containing light curve parquet files
    cfg : DictConfig
        Configuration
    source : str
        "gaia" or "ogle"
    n_jobs : int
        Parallel processes (currently sequential; future expansion)

    Returns
    -------
    pd.DataFrame
        Feature table with columns:
        [source_id, label, label_name, source, n_obs, alias_flag] + CONCEPT_NAMES_12

    Key corrections
    ---------------
    [M5] Gaia time column: if 'time' column is absent, log error and skip.
         No fallback to transit_id.
    """
    lc_dir = Path(lc_dir)
    results = []

    for idx, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc=f"Feature extraction ({source})",
    ):
        if source == "gaia":
            sid = str(row["source_id"])
            lc_path = lc_dir / f"{sid}.parquet"
        else:
            sid = str(row.get("ogle_id", row.get("source_id", "")))
            # OGLE light curves stored by type subdirectory
            label_name = row.get("label_name", "")
            lc_path = lc_dir / label_name / f"{sid}.parquet"
            if not lc_path.exists():
                lc_path = lc_dir / f"{sid}.parquet"

        if not lc_path.exists():
            continue

        try:
            lc_df = pd.read_parquet(lc_path)
        except Exception as e:
            logger.debug(f"[{sid}] Cannot read {lc_path}: {e}")
            continue

        if source == "gaia":
            # Gaia epoch photometry: filter to G band
            if "band" in lc_df.columns:
                g_data = lc_df[lc_df["band"] == "G"].copy()
            else:
                g_data = lc_df.copy()

            if len(g_data) < 10:
                continue

            # [M5] Time column required; no fallback to transit_id
            if "time" not in g_data.columns:
                logger.error(
                    f"[{sid}] Gaia epoch photometry missing 'time' column. "
                    f"Available columns: {list(g_data.columns)}. "
                    f"Check DataLink download. Skipping source."
                )
                continue

            # Compute magnitude errors from flux
            if "flux_error" in g_data.columns and "flux" in g_data.columns:
                flux = g_data["flux"].values.astype(float)
                flux_err = g_data["flux_error"].values.astype(float)
                valid_flux = (flux > 0) & (flux_err > 0)
                mag_err = np.where(
                    valid_flux,
                    flux_err / np.where(valid_flux, flux, 1.0) * 1.0857,
                    0.05,
                )
            elif "mag_error" in g_data.columns:
                mag_err = g_data["mag_error"].values.astype(float)
            else:
                mag_err = np.full(len(g_data), 0.05)
                logger.debug(f"[{sid}] No error column; using mag_err=0.05")

            lc = LightCurve(
                time=g_data["time"].values.astype(float),
                mag=g_data["mag"].values.astype(float),
                mag_err=mag_err,
                band="G",
                source_id=sid,
            )

        else:
            # OGLE: expects (hjd, mag, mag_err) format
            if "hjd" not in lc_df.columns:
                logger.debug(f"[{sid}] OGLE file missing 'hjd' column")
                continue

            lc = LightCurve(
                time=lc_df["hjd"].values.astype(float),
                mag=lc_df["mag"].values.astype(float),
                mag_err=lc_df["mag_err"].values.astype(float),
                band="I",
                source_id=sid,
            )

        # Extract features
        features = extract_all_features(
            lc=lc,
            metadata=row.to_dict(),
            cfg=cfg.features,
            var_type=str(row.get("label_name", "")),
        )

        # Add metadata columns
        features["source_id"] = sid
        features["label"] = int(row.get("label", -1))
        features["label_name"] = str(row.get("label_name", ""))
        features["n_obs"] = lc.n_obs
        features["source"] = source

        results.append(features)

    if not results:
        logger.warning(f"[{source}] No features extracted (empty results)")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Log NaN statistics
    concept_cols = [c for c in CONCEPT_NAMES_12 if c in df.columns]
    if concept_cols:
        nan_rate = float(df[concept_cols].isna().mean().mean())
        logger.info(
            f"[{source}] Feature extraction complete: {len(df)} sources, "
            f"NaN rate: {nan_rate:.1%}"
        )

    return df
