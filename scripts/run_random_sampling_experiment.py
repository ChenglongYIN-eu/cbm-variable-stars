#!/usr/bin/env python
"""
Brightness Bias Sensitivity Experiment for CBM Variable Star Classification.

Quantifies how the brightness-biased selection (brightest 3000/class) affects
HardCBM and Random Forest performance by training on magnitude-stratified
subsets of the existing data.

Since re-downloading random samples from the full Gaia catalog is not feasible,
we use the existing 18,000 sources (3000 x 6 classes, already the brightest per
class) and partition them by apparent magnitude to probe performance gradients.

Experiments
-----------
S1: Magnitude-tercile analysis
    Split each class into 3 brightness terciles (bright/medium/faint, 1000/class
    each). Train HardCBM + RF separately on each tercile (6000 sources each).
    Performance degradation from bright -> faint quantifies sensitivity to
    the brightness bias.

S2: Leave-bright-out analysis
    Remove the brightest 1000/class, train on the remaining 2000/class (12000
    total). Compare with full 3000/class baseline (18000 total). This isolates
    the contribution of the easiest (brightest) sources.

S3: Random sub-sampling analysis
    Randomly sample 2000/class from the full 3000 (12000 total). Run 3 independent
    random draws to estimate variance. This controls for sample-size effects
    independently of brightness.

S4: Magnitude-distribution summary
    Report per-class magnitude statistics for each subset to contextualize
    the performance numbers.

Usage
-----
    python run_random_sampling_experiment.py                    # Run all
    python run_random_sampling_experiment.py --experiments S1 S2 S3 S4
    python run_random_sampling_experiment.py --experiments S1   # Single experiment
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_NAMES_12,
    RANDOM_SEED,
    N_CV_FOLDS,
)
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.training.cross_val import run_cross_validation
from cbm_variable_stars.baselines.random_forest import train_random_forest
from cbm_variable_stars.data.splits import create_full_split
from pipeline_utils import compute_imputation_stats, apply_imputation

# ======================================================================
# Configuration
# ======================================================================
DATA_PATH = PROJECT_ROOT / "data" / "real" / "gaia_all_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "sampling_bias"
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 256
N_RANDOM_DRAWS = 10  # Number of independent random sub-samples in S3


# ======================================================================
# Shared utilities
# ======================================================================

def to_native(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


def load_full_data():
    """Load the full Gaia parquet file and return DataFrame + arrays."""
    df = pd.read_parquet(DATA_PATH)
    features = df[CONCEPT_NAMES_12].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)
    return df, features, labels


def prepare_cv_data(features: np.ndarray, labels: np.ndarray):
    """
    Apply the standard train/test split and imputation pipeline.

    Returns (cv_features, cv_labels) ready for cross-validation.
    The 15% hold-out test set is discarded here because we only need CV
    results for the sensitivity comparison.
    """
    split = create_full_split(labels, test_ratio=0.15)
    cv_idx = split["cv_indices"]
    cv_labels = labels[cv_idx]

    imp_stats = compute_imputation_stats(features[cv_idx], cv_labels)
    cv_features = apply_imputation(
        features[cv_idx], cv_labels, imp_stats, use_class_labels=True,
    )
    return cv_features, cv_labels


def run_hardcbm_cv(features, labels, tag=""):
    """Run HardCBM 5-fold CV on the given features/labels."""
    result = run_cross_validation(
        features=features,
        labels=labels,
        model_name="hard_cbm",
        model_kwargs=None,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        output_dir=str(RESULTS_DIR / tag),
    )
    return result


def run_rf_cv(features, labels, tag=""):
    """Run Random Forest 5-fold CV on the given features/labels."""
    result = train_random_forest(
        features=features,
        labels=labels,
        n_folds=N_CV_FOLDS,
        random_seed=RANDOM_SEED,
        output_dir=str(RESULTS_DIR / tag / "rf"),
    )
    return result


def extract_summary(result: dict, model_name: str) -> dict:
    """Extract a compact summary from a CV result dict."""
    agg = result["aggregated"]
    return {
        "model": model_name,
        "accuracy_mean": agg["accuracy_mean"],
        "accuracy_std": agg["accuracy_std"],
        "macro_f1_mean": agg["macro_f1_mean"],
        "macro_f1_std": agg["macro_f1_std"],
        "per_class_f1_mean": agg.get("per_class_f1_mean", []),
        "per_class_f1_std": agg.get("per_class_f1_std", []),
    }


def print_result(agg: dict, label: str):
    """Pretty-print aggregated results."""
    print(f"    [{label}]  Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}  "
          f"F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")


def create_subset_by_magnitude(
    df: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    tercile: int,
    n_terciles: int = 3,
) -> tuple:
    """
    Select a magnitude tercile from each class.

    For each class, sorts by mean_mag (brightest first = smallest value)
    and takes the k-th tercile (0=bright, 1=medium, 2=faint).

    Returns (subset_features, subset_labels, subset_indices).
    """
    indices = []
    for cls_idx in range(len(CLASS_NAMES)):
        cls_mask = labels == cls_idx
        cls_indices = np.where(cls_mask)[0]
        cls_mags = df["mean_mag"].values[cls_indices]

        # Sort by magnitude (ascending = brightest first)
        sort_order = np.argsort(cls_mags)
        sorted_indices = cls_indices[sort_order]

        n_per_class = len(sorted_indices)
        tercile_size = n_per_class // n_terciles

        start = tercile * tercile_size
        if tercile == n_terciles - 1:
            # Last tercile gets any remainder
            end = n_per_class
        else:
            end = start + tercile_size

        indices.extend(sorted_indices[start:end].tolist())

    indices = np.array(indices)
    return features[indices], labels[indices], indices


def create_leave_bright_out_subset(
    df: pd.DataFrame,
    features: np.ndarray,
    labels: np.ndarray,
    n_remove_per_class: int = 1000,
) -> tuple:
    """
    Remove the brightest n_remove_per_class sources from each class.

    Returns (subset_features, subset_labels, subset_indices).
    """
    indices = []
    for cls_idx in range(len(CLASS_NAMES)):
        cls_mask = labels == cls_idx
        cls_indices = np.where(cls_mask)[0]
        cls_mags = df["mean_mag"].values[cls_indices]

        # Sort by magnitude (ascending = brightest first)
        sort_order = np.argsort(cls_mags)
        sorted_indices = cls_indices[sort_order]

        # Skip the brightest n_remove_per_class
        remaining = sorted_indices[n_remove_per_class:]
        indices.extend(remaining.tolist())

    indices = np.array(indices)
    return features[indices], labels[indices], indices


def create_random_subsample(
    features: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 2000,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Randomly sample n_per_class sources from each class.

    Returns (subset_features, subset_labels, subset_indices).
    """
    rng = np.random.RandomState(seed)
    indices = []
    for cls_idx in range(len(CLASS_NAMES)):
        cls_mask = labels == cls_idx
        cls_indices = np.where(cls_mask)[0]
        chosen = rng.choice(cls_indices, size=min(n_per_class, len(cls_indices)),
                            replace=False)
        indices.extend(chosen.tolist())

    indices = np.array(indices)
    return features[indices], labels[indices], indices


# ======================================================================
# S1: Magnitude-Tercile Analysis
# ======================================================================

def run_s1_tercile_analysis(df, features, labels):
    """
    Split each class into 3 brightness terciles and train separately.

    Tercile 0 = brightest 1/3 (smallest mean_mag)
    Tercile 1 = medium 1/3
    Tercile 2 = faintest 1/3 (largest mean_mag)
    """
    print("\n" + "=" * 80)
    print("  S1: Magnitude-Tercile Analysis")
    print("  Train HardCBM and RF on each brightness tercile (1000/class each)")
    print("=" * 80)
    t_total = time.time()

    tercile_names = ["bright", "medium", "faint"]
    tercile_results = {}

    for t_idx, t_name in enumerate(tercile_names):
        print(f"\n  --- Tercile {t_idx}: {t_name} ---")
        t0 = time.time()

        sub_features, sub_labels, sub_indices = create_subset_by_magnitude(
            df, features, labels, tercile=t_idx, n_terciles=3,
        )

        # Report magnitude range
        sub_mags = df["mean_mag"].values[sub_indices]
        print(f"    N={len(sub_labels)}, "
              f"mag range=[{np.nanmin(sub_mags):.2f}, {np.nanmax(sub_mags):.2f}], "
              f"median={np.nanmedian(sub_mags):.2f}")

        # Class distribution check
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            n_cls = np.sum(sub_labels == cls_idx)
            cls_mask_local = sub_labels == cls_idx
            cls_mags_local = df["mean_mag"].values[sub_indices[cls_mask_local]]
            print(f"      {cls_name}: n={n_cls}, "
                  f"mag=[{np.nanmin(cls_mags_local):.2f}, {np.nanmax(cls_mags_local):.2f}]")

        # Prepare data
        cv_features, cv_labels = prepare_cv_data(sub_features, sub_labels)

        # HardCBM
        print(f"\n    Running HardCBM 5-fold CV...")
        hcbm_result = run_hardcbm_cv(cv_features, cv_labels,
                                      tag=f"S1_tercile_{t_name}")
        hcbm_summary = extract_summary(hcbm_result, "hard_cbm")
        print_result(hcbm_result["aggregated"], f"HardCBM {t_name}")

        # RF
        print(f"    Running RF 5-fold CV...")
        rf_result = run_rf_cv(cv_features, cv_labels,
                              tag=f"S1_tercile_{t_name}")
        rf_summary = extract_summary(rf_result, "random_forest")
        print_result(rf_result["aggregated"], f"RF {t_name}")

        tercile_results[t_name] = {
            "tercile_index": t_idx,
            "n_samples": int(len(sub_labels)),
            "mag_min": float(np.nanmin(sub_mags)),
            "mag_max": float(np.nanmax(sub_mags)),
            "mag_median": float(np.nanmedian(sub_mags)),
            "hard_cbm": hcbm_summary,
            "random_forest": rf_summary,
            "time_seconds": time.time() - t0,
        }

    # Summary table
    print(f"\n  S1 Summary:")
    print(f"  {'Tercile':<10} {'Mag Range':<20} {'HardCBM Acc':>14} {'RF Acc':>14} "
          f"{'HardCBM F1':>14} {'RF F1':>14}")
    print(f"  {'=' * 10} {'=' * 20} {'=' * 14} {'=' * 14} {'=' * 14} {'=' * 14}")
    for t_name in tercile_names:
        r = tercile_results[t_name]
        hcbm_acc = r["hard_cbm"]["accuracy_mean"]
        rf_acc = r["random_forest"]["accuracy_mean"]
        hcbm_f1 = r["hard_cbm"]["macro_f1_mean"]
        rf_f1 = r["random_forest"]["macro_f1_mean"]
        mag_range = f"[{r['mag_min']:.1f}, {r['mag_max']:.1f}]"
        print(f"  {t_name:<10} {mag_range:<20} {hcbm_acc:>14.4f} {rf_acc:>14.4f} "
              f"{hcbm_f1:>14.4f} {rf_f1:>14.4f}")

    # Compute degradation bright -> faint
    if "bright" in tercile_results and "faint" in tercile_results:
        bright_acc = tercile_results["bright"]["hard_cbm"]["accuracy_mean"]
        faint_acc = tercile_results["faint"]["hard_cbm"]["accuracy_mean"]
        delta_cbm = faint_acc - bright_acc
        bright_rf = tercile_results["bright"]["random_forest"]["accuracy_mean"]
        faint_rf = tercile_results["faint"]["random_forest"]["accuracy_mean"]
        delta_rf = faint_rf - bright_rf
        print(f"\n  Bright->Faint degradation:")
        print(f"    HardCBM: {delta_cbm:+.4f}")
        print(f"    RF:      {delta_rf:+.4f}")
        tercile_results["degradation_bright_to_faint"] = {
            "hard_cbm_delta_acc": delta_cbm,
            "rf_delta_acc": delta_rf,
        }

    print(f"\n  S1 total time: {time.time() - t_total:.1f}s")
    return tercile_results


# ======================================================================
# S2: Leave-Bright-Out Analysis
# ======================================================================

def run_s2_leave_bright_out(df, features, labels):
    """
    Compare full dataset (3000/class) with leave-bright-out (2000/class).
    """
    print("\n" + "=" * 80)
    print("  S2: Leave-Bright-Out Analysis")
    print("  Compare full (3000/class) vs removing brightest 1000/class")
    print("=" * 80)
    t_total = time.time()

    results = {}

    # ---- Full dataset baseline ----
    print(f"\n  --- Full dataset (3000/class, N={len(labels)}) ---")
    t0 = time.time()
    cv_features_full, cv_labels_full = prepare_cv_data(features, labels)

    print(f"    Running HardCBM 5-fold CV...")
    hcbm_full = run_hardcbm_cv(cv_features_full, cv_labels_full,
                                tag="S2_full_baseline")
    print_result(hcbm_full["aggregated"], "HardCBM full")

    print(f"    Running RF 5-fold CV...")
    rf_full = run_rf_cv(cv_features_full, cv_labels_full,
                        tag="S2_full_baseline")
    print_result(rf_full["aggregated"], "RF full")

    full_mags = df["mean_mag"].values
    results["full_3000"] = {
        "n_samples": int(len(labels)),
        "n_per_class": 3000,
        "mag_min": float(np.nanmin(full_mags)),
        "mag_max": float(np.nanmax(full_mags)),
        "mag_median": float(np.nanmedian(full_mags)),
        "hard_cbm": extract_summary(hcbm_full, "hard_cbm"),
        "random_forest": extract_summary(rf_full, "random_forest"),
        "time_seconds": time.time() - t0,
    }

    # ---- Leave-bright-out ----
    print(f"\n  --- Leave-bright-out (remove brightest 1000/class) ---")
    t0 = time.time()
    sub_features, sub_labels, sub_indices = create_leave_bright_out_subset(
        df, features, labels, n_remove_per_class=1000,
    )
    sub_mags = df["mean_mag"].values[sub_indices]
    print(f"    N={len(sub_labels)}, "
          f"mag range=[{np.nanmin(sub_mags):.2f}, {np.nanmax(sub_mags):.2f}], "
          f"median={np.nanmedian(sub_mags):.2f}")

    cv_features_sub, cv_labels_sub = prepare_cv_data(sub_features, sub_labels)

    print(f"    Running HardCBM 5-fold CV...")
    hcbm_sub = run_hardcbm_cv(cv_features_sub, cv_labels_sub,
                               tag="S2_leave_bright_out")
    print_result(hcbm_sub["aggregated"], "HardCBM leave-bright-out")

    print(f"    Running RF 5-fold CV...")
    rf_sub = run_rf_cv(cv_features_sub, cv_labels_sub,
                       tag="S2_leave_bright_out")
    print_result(rf_sub["aggregated"], "RF leave-bright-out")

    results["leave_bright_out_2000"] = {
        "n_samples": int(len(sub_labels)),
        "n_per_class": 2000,
        "n_removed_per_class": 1000,
        "mag_min": float(np.nanmin(sub_mags)),
        "mag_max": float(np.nanmax(sub_mags)),
        "mag_median": float(np.nanmedian(sub_mags)),
        "hard_cbm": extract_summary(hcbm_sub, "hard_cbm"),
        "random_forest": extract_summary(rf_sub, "random_forest"),
        "time_seconds": time.time() - t0,
    }

    # ---- Delta ----
    delta_hcbm_acc = (results["leave_bright_out_2000"]["hard_cbm"]["accuracy_mean"]
                      - results["full_3000"]["hard_cbm"]["accuracy_mean"])
    delta_rf_acc = (results["leave_bright_out_2000"]["random_forest"]["accuracy_mean"]
                    - results["full_3000"]["random_forest"]["accuracy_mean"])
    delta_hcbm_f1 = (results["leave_bright_out_2000"]["hard_cbm"]["macro_f1_mean"]
                     - results["full_3000"]["hard_cbm"]["macro_f1_mean"])
    delta_rf_f1 = (results["leave_bright_out_2000"]["random_forest"]["macro_f1_mean"]
                   - results["full_3000"]["random_forest"]["macro_f1_mean"])

    results["delta_leave_bright_out"] = {
        "hard_cbm_delta_acc": delta_hcbm_acc,
        "hard_cbm_delta_f1": delta_hcbm_f1,
        "rf_delta_acc": delta_rf_acc,
        "rf_delta_f1": delta_rf_f1,
    }

    print(f"\n  S2 Summary:")
    print(f"  {'Subset':<25} {'N':>7} {'HardCBM Acc':>14} {'RF Acc':>14} "
          f"{'HardCBM F1':>14} {'RF F1':>14}")
    print(f"  {'=' * 25} {'=' * 7} {'=' * 14} {'=' * 14} {'=' * 14} {'=' * 14}")
    for key, label, n in [("full_3000", "Full (3000/class)", len(labels)),
                          ("leave_bright_out_2000", "Leave-bright-out (2000/cls)", len(sub_labels))]:
        r = results[key]
        print(f"  {label:<25} {n:>7} {r['hard_cbm']['accuracy_mean']:>14.4f} "
              f"{r['random_forest']['accuracy_mean']:>14.4f} "
              f"{r['hard_cbm']['macro_f1_mean']:>14.4f} "
              f"{r['random_forest']['macro_f1_mean']:>14.4f}")
    print(f"\n  Delta (leave-bright-out minus full):")
    print(f"    HardCBM: Acc={delta_hcbm_acc:+.4f}, F1={delta_hcbm_f1:+.4f}")
    print(f"    RF:      Acc={delta_rf_acc:+.4f}, F1={delta_rf_f1:+.4f}")

    print(f"\n  S2 total time: {time.time() - t_total:.1f}s")
    return results


# ======================================================================
# S3: Random Sub-Sampling Analysis
# ======================================================================

def run_s3_random_subsampling(df, features, labels):
    """
    Randomly sample 2000/class (3 independent draws) and compare with full.

    This controls for sample-size effects: any performance difference between
    S2 (leave-bright-out) and S3 (random 2000) isolates the brightness effect
    from the sample-size effect.
    """
    print("\n" + "=" * 80)
    print("  S3: Random Sub-Sampling Analysis (2000/class, 3 draws)")
    print("=" * 80)
    t_total = time.time()

    draw_results = []

    for draw_idx in range(N_RANDOM_DRAWS):
        draw_seed = RANDOM_SEED + draw_idx * 1000
        print(f"\n  --- Draw {draw_idx + 1}/{N_RANDOM_DRAWS} (seed={draw_seed}) ---")
        t0 = time.time()

        sub_features, sub_labels, sub_indices = create_random_subsample(
            features, labels, n_per_class=2000, seed=draw_seed,
        )
        sub_mags = df["mean_mag"].values[sub_indices]
        print(f"    N={len(sub_labels)}, "
              f"mag range=[{np.nanmin(sub_mags):.2f}, {np.nanmax(sub_mags):.2f}], "
              f"median={np.nanmedian(sub_mags):.2f}")

        cv_features_sub, cv_labels_sub = prepare_cv_data(sub_features, sub_labels)

        # HardCBM
        print(f"    Running HardCBM 5-fold CV...")
        hcbm_result = run_hardcbm_cv(cv_features_sub, cv_labels_sub,
                                      tag=f"S3_random_draw{draw_idx}")
        print_result(hcbm_result["aggregated"], f"HardCBM draw {draw_idx + 1}")

        # RF
        print(f"    Running RF 5-fold CV...")
        rf_result = run_rf_cv(cv_features_sub, cv_labels_sub,
                              tag=f"S3_random_draw{draw_idx}")
        print_result(rf_result["aggregated"], f"RF draw {draw_idx + 1}")

        draw_results.append({
            "draw_index": draw_idx,
            "seed": draw_seed,
            "n_samples": int(len(sub_labels)),
            "mag_min": float(np.nanmin(sub_mags)),
            "mag_max": float(np.nanmax(sub_mags)),
            "mag_median": float(np.nanmedian(sub_mags)),
            "hard_cbm": extract_summary(hcbm_result, "hard_cbm"),
            "random_forest": extract_summary(rf_result, "random_forest"),
            "time_seconds": time.time() - t0,
        })

    # Aggregate across draws
    hcbm_accs = [d["hard_cbm"]["accuracy_mean"] for d in draw_results]
    rf_accs = [d["random_forest"]["accuracy_mean"] for d in draw_results]
    hcbm_f1s = [d["hard_cbm"]["macro_f1_mean"] for d in draw_results]
    rf_f1s = [d["random_forest"]["macro_f1_mean"] for d in draw_results]

    aggregate = {
        "hard_cbm_acc_mean": float(np.mean(hcbm_accs)),
        "hard_cbm_acc_std": float(np.std(hcbm_accs, ddof=1)) if len(hcbm_accs) > 1 else 0.0,
        "hard_cbm_f1_mean": float(np.mean(hcbm_f1s)),
        "hard_cbm_f1_std": float(np.std(hcbm_f1s, ddof=1)) if len(hcbm_f1s) > 1 else 0.0,
        "rf_acc_mean": float(np.mean(rf_accs)),
        "rf_acc_std": float(np.std(rf_accs, ddof=1)) if len(rf_accs) > 1 else 0.0,
        "rf_f1_mean": float(np.mean(rf_f1s)),
        "rf_f1_std": float(np.std(rf_f1s, ddof=1)) if len(rf_f1s) > 1 else 0.0,
    }

    # Summary
    print(f"\n  S3 Summary (random 2000/class, {N_RANDOM_DRAWS} draws):")
    print(f"    HardCBM:  Acc={aggregate['hard_cbm_acc_mean']:.4f}+/-{aggregate['hard_cbm_acc_std']:.4f}  "
          f"F1={aggregate['hard_cbm_f1_mean']:.4f}+/-{aggregate['hard_cbm_f1_std']:.4f}")
    print(f"    RF:       Acc={aggregate['rf_acc_mean']:.4f}+/-{aggregate['rf_acc_std']:.4f}  "
          f"F1={aggregate['rf_f1_mean']:.4f}+/-{aggregate['rf_f1_std']:.4f}")

    print(f"\n  S3 total time: {time.time() - t_total:.1f}s")

    return {
        "n_draws": N_RANDOM_DRAWS,
        "n_per_class": 2000,
        "draws": draw_results,
        "aggregate": aggregate,
    }


# ======================================================================
# S4: Magnitude Distribution Summary
# ======================================================================

def run_s4_magnitude_summary(df, features, labels):
    """
    Produce per-class magnitude statistics for all subsets.

    This provides the context needed to interpret S1-S3 results:
    how large is the magnitude spread within each class, and how do
    tercile boundaries compare to the overall Gaia magnitude distribution.
    """
    print("\n" + "=" * 80)
    print("  S4: Magnitude Distribution Summary")
    print("=" * 80)

    results = {"per_class": {}, "overall": {}}

    # Overall
    mags = df["mean_mag"].values
    results["overall"] = {
        "n": int(len(mags)),
        "min": float(np.nanmin(mags)),
        "max": float(np.nanmax(mags)),
        "mean": float(np.nanmean(mags)),
        "median": float(np.nanmedian(mags)),
        "std": float(np.nanstd(mags)),
        "q25": float(np.nanpercentile(mags, 25)),
        "q75": float(np.nanpercentile(mags, 75)),
    }

    print(f"\n  Overall: N={results['overall']['n']}, "
          f"mag=[{results['overall']['min']:.2f}, {results['overall']['max']:.2f}], "
          f"median={results['overall']['median']:.2f}")

    # Per-class
    print(f"\n  {'Class':<14} {'N':>6} {'Min':>8} {'Q25':>8} {'Median':>8} "
          f"{'Q75':>8} {'Max':>8} {'Std':>8}")
    print(f"  {'=' * 14} {'=' * 6} {'=' * 8} {'=' * 8} {'=' * 8} {'=' * 8} {'=' * 8} {'=' * 8}")

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_mask = labels == cls_idx
        cls_mags = mags[cls_mask]

        cls_stats = {
            "n": int(len(cls_mags)),
            "min": float(np.nanmin(cls_mags)),
            "max": float(np.nanmax(cls_mags)),
            "mean": float(np.nanmean(cls_mags)),
            "median": float(np.nanmedian(cls_mags)),
            "std": float(np.nanstd(cls_mags)),
            "q25": float(np.nanpercentile(cls_mags, 25)),
            "q75": float(np.nanpercentile(cls_mags, 75)),
        }
        results["per_class"][cls_name] = cls_stats

        print(f"  {cls_name:<14} {cls_stats['n']:>6} {cls_stats['min']:>8.2f} "
              f"{cls_stats['q25']:>8.2f} {cls_stats['median']:>8.2f} "
              f"{cls_stats['q75']:>8.2f} {cls_stats['max']:>8.2f} "
              f"{cls_stats['std']:>8.2f}")

    # Tercile boundaries per class
    print(f"\n  Tercile boundaries (mean_mag thresholds per class):")
    print(f"  {'Class':<14} {'T0/T1 boundary':>16} {'T1/T2 boundary':>16}")
    print(f"  {'=' * 14} {'=' * 16} {'=' * 16}")

    tercile_boundaries = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_mask = labels == cls_idx
        cls_mags = np.sort(mags[cls_mask])
        n = len(cls_mags)
        b1 = cls_mags[n // 3]
        b2 = cls_mags[2 * n // 3]
        tercile_boundaries[cls_name] = {"t0_t1": float(b1), "t1_t2": float(b2)}
        print(f"  {cls_name:<14} {b1:>16.2f} {b2:>16.2f}")

    results["tercile_boundaries"] = tercile_boundaries

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Brightness Bias Sensitivity Experiment (S1-S4)"
    )
    parser.add_argument(
        "--experiments", nargs="*",
        default=["S1", "S2", "S3", "S4"],
        help="Which experiments to run (e.g., --experiments S1 S2)"
    )
    args = parser.parse_args()
    experiments = [e.upper() for e in args.experiments]

    set_global_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  CBM Variable Star Classification")
    print("  Brightness Bias Sensitivity Experiment")
    print(f"  Experiments to run: {experiments}")
    print(f"  Data: {DATA_PATH}")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 80)

    # Load data
    print("\n  Loading data...")
    df, features, labels = load_full_data()
    print(f"  Loaded {len(df)} sources, {len(CLASS_NAMES)} classes, "
          f"{len(CONCEPT_NAMES_12)} concepts")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        n_cls = np.sum(labels == cls_idx)
        print(f"    {cls_name}: {n_cls}")

    all_results = {}
    t_grand = time.time()

    # S4 first (cheap, provides context)
    if "S4" in experiments:
        all_results["S4_magnitude_summary"] = run_s4_magnitude_summary(
            df, features, labels)

    # S1: Tercile analysis
    if "S1" in experiments:
        all_results["S1_tercile_analysis"] = run_s1_tercile_analysis(
            df, features, labels)

    # S2: Leave-bright-out
    if "S2" in experiments:
        all_results["S2_leave_bright_out"] = run_s2_leave_bright_out(
            df, features, labels)

    # S3: Random sub-sampling
    if "S3" in experiments:
        all_results["S3_random_subsampling"] = run_s3_random_subsampling(
            df, features, labels)

    # ---- Final comparison table ----
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON")
    print("=" * 80)

    comparison_rows = []

    if "S2_leave_bright_out" in all_results:
        s2 = all_results["S2_leave_bright_out"]
        if "full_3000" in s2:
            r = s2["full_3000"]
            comparison_rows.append(("Full (3000/cls)", len(labels),
                                    r["hard_cbm"]["accuracy_mean"],
                                    r["random_forest"]["accuracy_mean"],
                                    r["hard_cbm"]["macro_f1_mean"],
                                    r["random_forest"]["macro_f1_mean"]))
        if "leave_bright_out_2000" in s2:
            r = s2["leave_bright_out_2000"]
            comparison_rows.append(("Leave-bright (2000/cls)", r["n_samples"],
                                    r["hard_cbm"]["accuracy_mean"],
                                    r["random_forest"]["accuracy_mean"],
                                    r["hard_cbm"]["macro_f1_mean"],
                                    r["random_forest"]["macro_f1_mean"]))

    if "S3_random_subsampling" in all_results:
        s3 = all_results["S3_random_subsampling"]
        agg = s3["aggregate"]
        comparison_rows.append(("Random (2000/cls avg)", 12000,
                                agg["hard_cbm_acc_mean"],
                                agg["rf_acc_mean"],
                                agg["hard_cbm_f1_mean"],
                                agg["rf_f1_mean"]))

    if "S1_tercile_analysis" in all_results:
        s1 = all_results["S1_tercile_analysis"]
        for t_name in ["bright", "medium", "faint"]:
            if t_name in s1:
                r = s1[t_name]
                comparison_rows.append((f"Tercile: {t_name}", r["n_samples"],
                                        r["hard_cbm"]["accuracy_mean"],
                                        r["random_forest"]["accuracy_mean"],
                                        r["hard_cbm"]["macro_f1_mean"],
                                        r["random_forest"]["macro_f1_mean"]))

    if comparison_rows:
        print(f"\n  {'Subset':<26} {'N':>7} {'CBM Acc':>10} {'RF Acc':>10} "
              f"{'CBM F1':>10} {'RF F1':>10}")
        print(f"  {'=' * 26} {'=' * 7} {'=' * 10} {'=' * 10} {'=' * 10} {'=' * 10}")
        for label, n, cbm_acc, rf_acc, cbm_f1, rf_f1 in comparison_rows:
            print(f"  {label:<26} {n:>7} {cbm_acc:>10.4f} {rf_acc:>10.4f} "
                  f"{cbm_f1:>10.4f} {rf_f1:>10.4f}")

    # ---- Save all results ----
    results_path = RESULTS_DIR / "sampling_bias_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(to_native(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n  All results saved to: {results_path}")

    # Per-experiment files
    for key, data in all_results.items():
        exp_path = RESULTS_DIR / f"{key}.json"
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(to_native(data), f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_grand
    print(f"\n{'=' * 80}")
    print(f"  Brightness bias experiment complete!")
    print(f"  Total wall time: {total_time / 60:.1f} minutes")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
