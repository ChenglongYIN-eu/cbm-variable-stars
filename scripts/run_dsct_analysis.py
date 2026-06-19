#!/usr/bin/env python3
"""
Analyze the internal heterogeneity of the DSCT_SXPHE merged class.

Gaia DR3 classifies DSCT, GDOR, and SXPhe together under the label
'DSCT|GDOR|SXPHE'.  This script quantifies the resulting class
heterogeneity — most notably the expected bimodality in the period
distribution (GDOR: 0.3–3 d vs DSCT: 0.02–0.25 d) — and compares
concept distributions of DSCT_SXPHE with the other five classes.

Output:  results/supplementary/dsct_analysis.json

Usage:
    python scripts/run_dsct_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde, skew, kurtosis as sp_kurtosis

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES_12,
    CLASS_NAMES,
    NUM_CLASSES,
)

DATA_PATH = PROJECT_ROOT / "data" / "real" / "gaia_all_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "supplementary"
OUTPUT_PATH = RESULTS_DIR / "dsct_analysis.json"

# Physical reference ranges for sub-populations within DSCT_SXPHE
DSCT_PERIOD_RANGE = (0.02, 0.25)    # Delta Scuti (days)
GDOR_PERIOD_RANGE = (0.3, 3.0)      # Gamma Doradus (days)
SXPHE_PERIOD_RANGE = (0.02, 0.10)   # SX Phoenicis (days, subset of DSCT range)

DSCT_LABEL_IDX = CLASS_NAMES.index("DSCT_SXPHE")  # == 3


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def safe_float(x: Any) -> Any:
    """Convert numpy scalars to native Python floats for JSON serialization."""
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def compute_concept_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute descriptive statistics for a single concept array."""
    valid = values[~np.isnan(values)]
    n_total = len(values)
    n_valid = len(valid)
    if n_valid == 0:
        return {
            "n_total": n_total,
            "n_valid": 0,
            "nan_fraction": 1.0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
            "q25": None,
            "q75": None,
        }
    return {
        "n_total": n_total,
        "n_valid": int(n_valid),
        "nan_fraction": safe_float(1.0 - n_valid / n_total),
        "mean": safe_float(np.mean(valid)),
        "std": safe_float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0,
        "median": safe_float(np.median(valid)),
        "min": safe_float(np.min(valid)),
        "max": safe_float(np.max(valid)),
        "skewness": safe_float(skew(valid, nan_policy="omit")) if n_valid > 2 else None,
        "kurtosis": safe_float(sp_kurtosis(valid, fisher=True, nan_policy="omit")) if n_valid > 3 else None,
        "q25": safe_float(np.percentile(valid, 25)),
        "q75": safe_float(np.percentile(valid, 75)),
    }


def kde_peak_analysis(
    values: np.ndarray,
    n_grid: int = 1000,
    bw_method: str | float | None = None,
    log_transform: bool = False,
) -> Dict[str, Any]:
    """
    Fit a Gaussian KDE and find local maxima (modes) in the density.

    Parameters
    ----------
    values : array of valid (non-NaN) values
    n_grid : number of evaluation points
    bw_method : bandwidth selector passed to gaussian_kde
    log_transform : if True, fit in log10-space (useful for period)

    Returns
    -------
    dict with peak info, number of modes, and Ashman D for the two
    highest modes (if ≥2 peaks).
    """
    if len(values) < 10:
        return {"n_peaks": 0, "peaks": [], "note": "too few values"}

    x_raw = np.log10(values) if log_transform else values

    # Skip if zero variance (e.g., constant imputed values)
    if np.std(x_raw) < 1e-10:
        return {"n_peaks": 1, "peaks": [{"location": float(np.mean(x_raw)), "density": 1.0}],
                "note": "constant values (zero variance)"}

    try:
        kde = gaussian_kde(x_raw, bw_method=bw_method)
    except np.linalg.LinAlgError:
        return {"n_peaks": 0, "peaks": [], "note": "KDE failed (singular covariance)"}
    x_grid = np.linspace(x_raw.min(), x_raw.max(), n_grid)
    density = kde(x_grid)

    maxima_idx = argrelextrema(density, np.greater, order=5)[0]
    minima_idx = argrelextrema(density, np.less, order=5)[0]

    peaks = []
    for idx in maxima_idx:
        loc = x_grid[idx]
        peak_val = float(density[idx])
        if log_transform:
            peaks.append({
                "log10_value": safe_float(loc),
                "value_days": safe_float(10 ** loc),
                "density": peak_val,
            })
        else:
            peaks.append({
                "value": safe_float(loc),
                "density": peak_val,
            })

    # Sort peaks by density descending
    peaks.sort(key=lambda p: p["density"], reverse=True)

    result: Dict[str, Any] = {
        "n_peaks": len(peaks),
        "peaks": peaks,
        "n_minima": len(minima_idx),
    }

    # Ashman D statistic for bimodality: D = |μ1−μ2| * sqrt(2) / sqrt(σ1²+σ2²)
    # D > 2 suggests clean bimodal separation
    if len(peaks) >= 2:
        if log_transform:
            p1, p2 = peaks[0]["log10_value"], peaks[1]["log10_value"]
        else:
            p1, p2 = peaks[0]["value"], peaks[1]["value"]
        # Split data at the deepest minimum between the two highest peaks
        split_point = None
        if len(minima_idx) > 0:
            left, right = sorted([p1, p2])
            between = [i for i in minima_idx if left < x_grid[i] < right]
            if between:
                split_point = x_grid[min(between, key=lambda i: density[i])]
        if split_point is not None:
            g1 = x_raw[x_raw <= split_point]
            g2 = x_raw[x_raw > split_point]
            if len(g1) > 1 and len(g2) > 1:
                mu1, mu2 = np.mean(g1), np.mean(g2)
                s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
                ashman_d = abs(mu1 - mu2) * np.sqrt(2) / np.sqrt(s1 ** 2 + s2 ** 2)
                result["ashman_D"] = safe_float(ashman_d)
                result["split_point"] = safe_float(split_point)
                result["group1_n"] = int(len(g1))
                result["group2_n"] = int(len(g2))
                if log_transform:
                    result["split_point_days"] = safe_float(10 ** split_point)
                    result["group1_mean_days"] = safe_float(10 ** mu1)
                    result["group2_mean_days"] = safe_float(10 ** mu2)

    return result


def diptest_pvalue(values: np.ndarray) -> Dict[str, Any]:
    """
    Hartigan's dip test for unimodality.

    Uses the ``diptest`` package if available; otherwise falls back to a
    note explaining the package is missing.
    """
    try:
        import diptest
        dip_stat, p_value = diptest.diptest(values)
        return {
            "dip_statistic": safe_float(dip_stat),
            "p_value": safe_float(p_value),
            "reject_unimodality_at_005": bool(p_value < 0.05),
        }
    except ImportError:
        return {
            "note": "diptest package not installed; skipping Hartigan dip test. "
                    "Install with: pip install diptest"
        }


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("DSCT_SXPHE Internal Heterogeneity Analysis")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Total sources: {len(df)}")
    for idx, name in enumerate(CLASS_NAMES):
        n = (df["label"] == idx).sum()
        print(f"  {name}: {n}")

    # ------------------------------------------------------------------
    # 2. Filter to DSCT_SXPHE
    # ------------------------------------------------------------------
    dsct_mask = df["label"] == DSCT_LABEL_IDX
    dsct_df = df[dsct_mask].copy()
    n_dsct = len(dsct_df)
    print(f"\nDSCT_SXPHE sources: {n_dsct}")

    results: Dict[str, Any] = {
        "class": "DSCT_SXPHE",
        "label_index": DSCT_LABEL_IDX,
        "gaia_label": "DSCT|GDOR|SXPHE",
        "n_sources": n_dsct,
        "description": (
            "This class merges Delta Scuti (DSCT), Gamma Doradus (GDOR), "
            "and SX Phoenicis (SXPhe) stars as classified by Gaia DR3 "
            "(vari_classifier_result.best_class_name = 'DSCT|GDOR|SXPHE'). "
            "The three sub-types occupy distinct period ranges — "
            "DSCT: 0.02–0.25 d, GDOR: 0.3–3 d, SXPhe: 0.02–0.10 d — "
            "leading to expected bimodality in the period distribution."
        ),
    }

    # ------------------------------------------------------------------
    # 3. Per-concept descriptive statistics for DSCT_SXPHE
    # ------------------------------------------------------------------
    print("\n--- Per-Concept Statistics (DSCT_SXPHE) ---")
    concept_stats: Dict[str, Any] = {}
    for concept in CONCEPT_NAMES_12:
        if concept in dsct_df.columns:
            vals = dsct_df[concept].values.astype(np.float64)
        else:
            vals = np.full(n_dsct, np.nan)
        st = compute_concept_stats(vals)
        concept_stats[concept] = st
        if st["mean"] is not None:
            print(f"  {concept:20s}  mean={st['mean']:.4f}  std={st['std']:.4f}  "
                  f"median={st['median']:.4f}  skew={st['skewness']:.3f}  "
                  f"NaN={st['nan_fraction']*100:.0f}%")
        else:
            print(f"  {concept:20s}  ALL NaN")
    results["concept_statistics"] = concept_stats

    # ------------------------------------------------------------------
    # 4. Multimodality analysis for each concept
    # ------------------------------------------------------------------
    print("\n--- Multimodality Analysis ---")
    multimodality: Dict[str, Any] = {}
    for concept in CONCEPT_NAMES_12:
        if concept in dsct_df.columns:
            valid = dsct_df[concept].dropna().values.astype(np.float64)
        else:
            valid = np.array([])
        if len(valid) < 20:
            multimodality[concept] = {"note": "insufficient valid values"}
            continue

        use_log = concept == "period"
        kde_result = kde_peak_analysis(valid, log_transform=use_log)
        dip_result = diptest_pvalue(valid)
        multimodality[concept] = {
            "kde_analysis": kde_result,
            "hartigan_dip_test": dip_result,
        }
        n_peaks = kde_result["n_peaks"]
        extra = ""
        if "ashman_D" in kde_result:
            extra = f"  Ashman D={kde_result['ashman_D']:.2f}"
        print(f"  {concept:20s}  KDE peaks: {n_peaks}{extra}")
        if "dip_statistic" in dip_result:
            print(f"  {'':20s}  Dip stat={dip_result['dip_statistic']:.4f}, "
                  f"p={dip_result['p_value']:.4e}")
    results["multimodality"] = multimodality

    # ------------------------------------------------------------------
    # 5. Period distribution deep-dive
    # ------------------------------------------------------------------
    print("\n--- Period Distribution Deep-Dive ---")
    periods = dsct_df["period"].dropna().values.astype(np.float64)
    periods = periods[periods > 0]
    print(f"  Valid periods: {len(periods)}")

    period_detail: Dict[str, Any] = {
        "n_valid": int(len(periods)),
    }

    # Sub-population counts based on physical period ranges
    n_dsct_range = int(np.sum((periods >= DSCT_PERIOD_RANGE[0]) & (periods <= DSCT_PERIOD_RANGE[1])))
    n_gdor_range = int(np.sum((periods >= GDOR_PERIOD_RANGE[0]) & (periods <= GDOR_PERIOD_RANGE[1])))
    n_sxphe_range = int(np.sum((periods >= SXPHE_PERIOD_RANGE[0]) & (periods <= SXPHE_PERIOD_RANGE[1])))
    n_gap = int(np.sum((periods > DSCT_PERIOD_RANGE[1]) & (periods < GDOR_PERIOD_RANGE[0])))
    n_outside = int(len(periods) - n_dsct_range - n_gap - n_gdor_range
                     - np.sum(periods > GDOR_PERIOD_RANGE[1]))
    period_detail["physical_ranges"] = {
        "DSCT_range_0.02_0.25d": n_dsct_range,
        "gap_0.25_0.3d": n_gap,
        "GDOR_range_0.3_3d": n_gdor_range,
        "SXPhe_within_DSCT_range_0.02_0.10d": n_sxphe_range,
        "fraction_DSCT_like": safe_float(n_dsct_range / len(periods)) if len(periods) > 0 else 0,
        "fraction_GDOR_like": safe_float(n_gdor_range / len(periods)) if len(periods) > 0 else 0,
    }
    print(f"  DSCT-like (0.02–0.25 d): {n_dsct_range} ({100*n_dsct_range/len(periods):.1f}%)")
    print(f"  Gap (0.25–0.30 d):       {n_gap}")
    print(f"  GDOR-like (0.3–3.0 d):   {n_gdor_range} ({100*n_gdor_range/len(periods):.1f}%)")

    # KDE on log10(period) — this is the key bimodality check
    if len(periods) >= 20:
        log_periods = np.log10(periods)
        kde = gaussian_kde(log_periods)
        x = np.linspace(log_periods.min(), log_periods.max(), 1000)
        density = kde(x)
        maxima = argrelextrema(density, np.greater, order=5)[0]
        print(f"\n  KDE on log10(period): {len(maxima)} peak(s)")
        for m in maxima:
            print(f"    Peak at log10(P)={x[m]:.3f}  →  P={10**x[m]:.4f} days  "
                  f"(density={density[m]:.3f})")

        # Detailed KDE result (reuse helper for consistency)
        period_detail["kde_log_period"] = kde_peak_analysis(
            periods, log_transform=True
        )

        # Also try with tighter bandwidth to resolve close peaks
        period_detail["kde_log_period_narrow_bw"] = kde_peak_analysis(
            periods, log_transform=True, bw_method=0.05
        )

    # Dip test on log10(period)
    if len(periods) >= 20:
        period_detail["dip_test_log_period"] = diptest_pvalue(np.log10(periods))

    # Period percentiles
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    period_detail["percentiles"] = {
        f"p{p}": safe_float(np.percentile(periods, p)) for p in pcts
    }

    results["period_analysis"] = period_detail

    # ------------------------------------------------------------------
    # 6. Compare DSCT_SXPHE concept distributions with other classes
    # ------------------------------------------------------------------
    print("\n--- Cross-Class Comparison ---")
    cross_class: Dict[str, Any] = {}
    for concept in CONCEPT_NAMES_12:
        if concept not in df.columns:
            continue
        concept_comp: Dict[str, Any] = {}
        dsct_vals = dsct_df[concept].dropna().values.astype(np.float64)
        if len(dsct_vals) < 10:
            cross_class[concept] = {"note": "insufficient DSCT_SXPHE data"}
            continue

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            if cls_idx == DSCT_LABEL_IDX:
                continue
            other_vals = df.loc[df["label"] == cls_idx, concept].dropna().values.astype(np.float64)
            if len(other_vals) < 10:
                concept_comp[cls_name] = {"note": "insufficient data"}
                continue
            # Mann-Whitney U test (non-parametric)
            try:
                u_stat, u_pval = sp_stats.mannwhitneyu(
                    dsct_vals, other_vals, alternative="two-sided"
                )
            except ValueError:
                u_stat, u_pval = np.nan, np.nan
            # Effect size: rank-biserial correlation = 1 - 2U/(n1*n2)
            n1, n2 = len(dsct_vals), len(other_vals)
            rank_biserial = 1.0 - 2.0 * u_stat / (n1 * n2) if n1 * n2 > 0 else np.nan
            # Cohen's d
            pooled_std = np.sqrt(
                ((n1 - 1) * np.var(dsct_vals, ddof=1) + (n2 - 1) * np.var(other_vals, ddof=1))
                / (n1 + n2 - 2)
            )
            cohens_d = (np.mean(dsct_vals) - np.mean(other_vals)) / pooled_std if pooled_std > 0 else np.nan
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = sp_stats.ks_2samp(dsct_vals, other_vals)

            concept_comp[cls_name] = {
                "dsct_mean": safe_float(np.mean(dsct_vals)),
                "other_mean": safe_float(np.mean(other_vals)),
                "cohens_d": safe_float(cohens_d),
                "mann_whitney_U": safe_float(u_stat),
                "mann_whitney_p": safe_float(u_pval),
                "rank_biserial_r": safe_float(rank_biserial),
                "ks_statistic": safe_float(ks_stat),
                "ks_p_value": safe_float(ks_pval),
            }
        cross_class[concept] = concept_comp
    results["cross_class_comparison"] = cross_class

    # Print summary of most distinctive concepts
    print("\n  Concepts where DSCT_SXPHE differs most from other classes (max |Cohen's d|):")
    for concept in CONCEPT_NAMES_12:
        if concept not in cross_class or "note" in cross_class[concept]:
            continue
        max_d = 0.0
        max_cls = ""
        for cls_name, vals in cross_class[concept].items():
            if isinstance(vals, dict) and "cohens_d" in vals and vals["cohens_d"] is not None:
                if abs(vals["cohens_d"]) > abs(max_d):
                    max_d = vals["cohens_d"]
                    max_cls = cls_name
        if max_cls:
            print(f"    {concept:20s}  max |d| = {abs(max_d):.2f} vs {max_cls}")

    # ------------------------------------------------------------------
    # 7. Within-class spread comparison (coefficient of variation)
    # ------------------------------------------------------------------
    print("\n--- Within-Class Spread (Coefficient of Variation) ---")
    cv_comparison: Dict[str, Dict[str, float]] = {}
    for concept in CONCEPT_NAMES_12:
        if concept not in df.columns:
            continue
        cv_row: Dict[str, float] = {}
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            vals = df.loc[df["label"] == cls_idx, concept].dropna().values.astype(np.float64)
            if len(vals) > 1 and np.mean(vals) != 0:
                cv = float(np.std(vals, ddof=1) / abs(np.mean(vals)))
            else:
                cv = float("nan")
            cv_row[cls_name] = safe_float(cv)
        cv_comparison[concept] = cv_row

    # Highlight concepts where DSCT_SXPHE has unusually high CV
    print(f"  {'Concept':20s}  {'DSCT_SXPHE CV':>15s}  {'Mean other CV':>15s}  {'Ratio':>8s}")
    for concept in CONCEPT_NAMES_12:
        if concept not in cv_comparison:
            continue
        dsct_cv = cv_comparison[concept].get("DSCT_SXPHE", float("nan"))
        other_cvs = [
            v for k, v in cv_comparison[concept].items()
            if k != "DSCT_SXPHE" and not np.isnan(v)
        ]
        if other_cvs and not np.isnan(dsct_cv):
            mean_other = np.mean(other_cvs)
            ratio = dsct_cv / mean_other if mean_other > 0 else float("nan")
            print(f"  {concept:20s}  {dsct_cv:15.3f}  {mean_other:15.3f}  {ratio:8.2f}")
    results["coefficient_of_variation"] = cv_comparison

    # ------------------------------------------------------------------
    # 8. Paper-ready summary
    # ------------------------------------------------------------------
    period_kde = results.get("period_analysis", {}).get("kde_log_period", {})
    n_peaks = period_kde.get("n_peaks", 0)
    peaks_str = ""
    if "peaks" in period_kde:
        for p in period_kde["peaks"]:
            if "value_days" in p:
                peaks_str += f"P={p['value_days']:.3f}d, "
    ashman = period_kde.get("ashman_D", None)

    summary_lines = [
        f"The DSCT_SXPHE class contains {n_dsct} sources from Gaia DR3 "
        f"(best_class_name = 'DSCT|GDOR|SXPHE').",
        f"Period distribution analysis reveals {n_peaks} KDE peak(s) in log10(period): {peaks_str.rstrip(', ')}.",
    ]
    if ashman is not None:
        summary_lines.append(
            f"Ashman D = {ashman:.2f} ({'bimodal separation confirmed (D>2)' if ashman > 2 else 'marginal separation (D<2)'})."
        )
    summary_lines.append(
        f"By physical period range: {n_dsct_range} DSCT-like (0.02–0.25 d, "
        f"{100*n_dsct_range/len(periods):.1f}%), "
        f"{n_gdor_range} GDOR-like (0.3–3 d, "
        f"{100*n_gdor_range/len(periods):.1f}%)."
    )
    summary_lines.append(
        "This heterogeneity is an intrinsic limitation of the Gaia DR3 classification scheme "
        "and should be acknowledged when interpreting per-class CBM concept predictions."
    )
    results["paper_summary"] = " ".join(summary_lines)
    print("\n--- Paper Summary ---")
    print(results["paper_summary"])

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=safe_float)
    print(f"\nResults saved to {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
