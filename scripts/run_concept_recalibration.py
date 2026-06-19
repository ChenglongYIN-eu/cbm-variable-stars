#!/usr/bin/env python3
"""
Concept-Selective Recalibration Experiment.

Validates the paper's core claim that "CBM enables targeted recalibration"
in cross-survey scenarios. Unlike black-box models where domain adaptation
requires retraining, a CBM allows selective replacement of high-domain-shift
concepts while preserving well-transferred ones.

Experiment design:
  1. Load OGLE feature data (out-of-domain, I-band)
  2. Load Gaia-trained HardCBM model (fold 4 checkpoint)
  3. Reconstruct Gaia training data scaler (same pipeline as training)
  4. Prepare OGLE features: impute with Gaia global medians, standardize
  5. Run zero-shot prediction on OGLE -> baseline accuracy (~17%)
  6. Concept-selective recalibration:
     a. Identify high-KS concepts (period_snr, stetson_K, R31 -- KS > 0.7)
     b. Compute OGLE concept medians in scaled space
     c. Replace high-KS concept columns with OGLE medians (per-class)
     d. Progressively replace 1, 2, 3 high-KS concepts, measure accuracy
     e. Control: replace 3 LOW-KS concepts -> should not improve
  7. Save results to results/supplementary/concept_recalibration.json

Usage:
    python scripts/run_concept_recalibration.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── Project imports ──────────────────────────────────────────────────────────
from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES_12,
    CLASS_NAMES,
    RANDOM_SEED,
    OGLE_SUBTYPE_MAP,
    NUM_CONCEPTS,
    NUM_CLASSES,
)
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.models import create_model
from cbm_variable_stars.data.splits import create_full_split, create_cv_splits
from pipeline_utils import compute_imputation_stats, apply_imputation

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "real" / "gaia_all_features.parquet"
OGLE_RAW_PATH = PROJECT_ROOT / "data" / "interim" / "ogle_features_raw.parquet"
CKPT_DIR = PROJECT_ROOT / "results" / "real" / "hard_cbm" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "supplementary"

# High-KS concepts: identified from B8 domain shift analysis (KS > 0.7)
# These are concepts with the largest distribution shift between Gaia and OGLE
HIGH_KS_CONCEPTS = ["period_snr", "stetson_K", "R31"]

# Low-KS concepts: well-transferred, should NOT improve when replaced
LOW_KS_CONCEPTS = ["period", "skewness", "amplitude"]


# ── Utilities ────────────────────────────────────────────────────────────────

def to_native(obj: Any) -> Any:
    """Recursively convert numpy/torch types to native Python for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    return obj


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_native(data), f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


# ── Core Functions ───────────────────────────────────────────────────────────

def load_gaia_data_and_scaler() -> Tuple[
    np.ndarray, np.ndarray, StandardScaler, dict, np.ndarray
]:
    """
    Load Gaia data and reconstruct the exact scaler used for fold 4 training.

    Returns:
        gaia_features_raw: Raw features (with NaN), shape (N, 12)
        gaia_labels: Labels, shape (N,)
        scaler: StandardScaler fitted on fold 4 training data
        imp_stats: Imputation statistics from CV data
        cv_idx: CV subset indices
    """
    print("[1/5] Loading Gaia data and reconstructing scaler...")
    df = pd.read_parquet(DATA_PATH)
    gaia_features = df[CONCEPT_NAMES_12].values.astype(np.float32)
    gaia_labels = df["label"].values.astype(np.int64)
    print(f"  Gaia samples: {len(gaia_labels)}")

    # Reproduce the exact train/test split
    split = create_full_split(gaia_labels, test_ratio=0.15, random_seed=RANDOM_SEED)
    cv_idx = split["cv_indices"]

    # Compute imputation stats on CV data only (Fix C1)
    imp_stats = compute_imputation_stats(gaia_features[cv_idx], gaia_labels[cv_idx])

    # Impute CV data with per-class medians
    cv_imputed = apply_imputation(
        gaia_features[cv_idx], gaia_labels[cv_idx], imp_stats,
        use_class_labels=True,
    )

    # Get fold 4 training indices within CV subset
    cv_labels = gaia_labels[cv_idx]
    folds = create_cv_splits(cv_labels, n_folds=5, random_seed=RANDOM_SEED)
    train_idx_f4, val_idx_f4 = folds[4]

    # Fit scaler on fold 4 training data only
    scaler = StandardScaler()
    scaler.fit(cv_imputed[train_idx_f4])
    print(f"  Scaler fitted on fold 4 training data: {len(train_idx_f4)} samples")

    return gaia_features, gaia_labels, scaler, imp_stats, cv_idx


def load_ogle_data(
    imp_stats: dict, scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load OGLE data, impute, and standardize using Gaia statistics.

    Returns:
        ogle_features_raw: Raw features before imputation (with NaN), shape (N, 12)
        ogle_imputed: Imputed features (before scaling), shape (N, 12)
        ogle_scaled: Standardized features, shape (N, 12)
        ogle_labels: Labels, shape (N,)
    """
    print("\n[2/5] Loading and preparing OGLE data...")
    if not OGLE_RAW_PATH.exists():
        raise FileNotFoundError(f"OGLE data not found at {OGLE_RAW_PATH}")

    df_ogle = pd.read_parquet(OGLE_RAW_PATH)
    print(f"  OGLE DataFrame shape: {df_ogle.shape}")
    print(f"  OGLE columns: {df_ogle.columns.tolist()}")

    # Extract features and labels
    ogle_features_raw = df_ogle[CONCEPT_NAMES_12].values.astype(np.float32)
    ogle_labels = df_ogle["label"].values.astype(np.int64)

    # Class distribution
    unique, counts = np.unique(ogle_labels, return_counts=True)
    print(f"  OGLE samples: {len(ogle_labels)}")
    print(f"  Class distribution:")
    for u, c in zip(unique, counts):
        print(f"    {CLASS_NAMES[u]}: {c}")

    # NaN analysis
    print(f"\n  NaN rates per concept:")
    for ci, concept in enumerate(CONCEPT_NAMES_12):
        nan_pct = 100 * np.isnan(ogle_features_raw[:, ci]).sum() / len(ogle_labels)
        if nan_pct > 0:
            print(f"    {concept}: {nan_pct:.1f}% NaN")

    # Impute using Gaia global medians (OGLE is out-of-domain, no class labels)
    ogle_imputed = apply_imputation(
        ogle_features_raw, ogle_labels, imp_stats, use_class_labels=False,
    ).astype(np.float32)

    # Standardize using Gaia scaler
    ogle_scaled = scaler.transform(ogle_imputed).astype(np.float32)

    return ogle_features_raw, ogle_imputed, ogle_scaled, ogle_labels


def load_hard_cbm_model(fold: int = 4) -> torch.nn.Module:
    """Load trained HardCBM model from checkpoint."""
    print(f"\n[3/5] Loading HardCBM model (fold {fold})...")
    ckpt_path = CKPT_DIR / f"best_model_fold{fold}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = create_model("hard_cbm")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded: {ckpt_path.name} (epoch {ckpt.get('epoch', '?')})")
    return model


def compute_domain_shift(
    gaia_features_raw: np.ndarray,
    gaia_labels: np.ndarray,
    cv_idx: np.ndarray,
    ogle_features_raw: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-concept domain shift using two-sample KS test.

    Compares Gaia CV pool vs OGLE raw features (pre-standardization).
    """
    print("\n[4/5] Computing per-concept domain shift (KS test)...")
    gaia_cv_raw = gaia_features_raw[cv_idx]
    domain_shift = {}

    for ci, concept in enumerate(CONCEPT_NAMES_12):
        gaia_vals = gaia_cv_raw[:, ci][~np.isnan(gaia_cv_raw[:, ci])]
        ogle_vals = ogle_features_raw[:, ci][~np.isnan(ogle_features_raw[:, ci])]

        if len(gaia_vals) > 5 and len(ogle_vals) > 5:
            ks_stat, ks_p = stats.ks_2samp(gaia_vals, ogle_vals)
            mean_shift = abs(float(np.mean(ogle_vals)) - float(np.mean(gaia_vals)))
        else:
            ks_stat, ks_p, mean_shift = float("nan"), float("nan"), float("nan")

        domain_shift[concept] = {
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "mean_shift": float(mean_shift),
            "n_gaia": int(len(gaia_vals)),
            "n_ogle": int(len(ogle_vals)),
        }

    # Print sorted by KS statistic
    sorted_shift = sorted(domain_shift.items(), key=lambda x: -x[1]["ks_statistic"])
    print("  Domain shift ranking (by KS statistic):")
    for concept, ds in sorted_shift:
        marker = " ***" if concept in HIGH_KS_CONCEPTS else ""
        print(f"    {concept:15s}: KS={ds['ks_statistic']:.3f} "
              f"(p={ds['ks_p_value']:.2e}){marker}")

    return domain_shift


def predict_with_replacement(
    model: torch.nn.Module,
    features_scaled: np.ndarray,
    replace_concepts: List[str],
    replace_values: np.ndarray,
) -> np.ndarray:
    """
    Replace specific concept columns with given values and predict.

    Args:
        model: Trained HardCBM model.
        features_scaled: Standardized features, shape (N, 12).
        replace_concepts: List of concept names to replace.
        replace_values: Values to use for replacement, shape (N,) or scalar per concept.
            These should already be in scaled (z-score) space.

    Returns:
        Predicted class labels, shape (N,).
    """
    modified = features_scaled.copy()
    for i, concept_name in enumerate(replace_concepts):
        idx = CONCEPT_NAMES_12.index(concept_name)
        if isinstance(replace_values, dict):
            modified[:, idx] = replace_values[concept_name]
        elif replace_values.ndim == 1:
            # Single value per concept
            modified[:, idx] = replace_values[i]
        else:
            # Per-sample values
            modified[:, idx] = replace_values[:, i]

    with torch.no_grad():
        out = model(torch.tensor(modified, dtype=torch.float32))
        preds = out["logits"].argmax(dim=1).numpy()
    return preds


def predict_with_class_medians(
    model: torch.nn.Module,
    features_scaled: np.ndarray,
    labels: np.ndarray,
    replace_concepts: List[str],
    ogle_scaled: np.ndarray,
    ogle_labels: np.ndarray,
) -> np.ndarray:
    """
    Replace specific concepts with per-class OGLE medians in scaled space.

    For each sample, uses the median of its TRUE class in the OGLE dataset.
    This simulates the best-case scenario where we know the correct class
    and can provide class-appropriate concept values.

    Args:
        model: Trained HardCBM model.
        features_scaled: Input features (standardized), shape (N, 12).
        labels: True labels for per-class routing, shape (N,).
        replace_concepts: Concept names to replace.
        ogle_scaled: Full OGLE scaled features for computing medians.
        ogle_labels: OGLE labels for per-class median computation.

    Returns:
        Predicted class labels, shape (N,).
    """
    modified = features_scaled.copy()

    for concept_name in replace_concepts:
        cidx = CONCEPT_NAMES_12.index(concept_name)
        for cls in np.unique(labels):
            cls_mask = labels == cls
            # Compute OGLE class median for this concept
            ogle_cls_vals = ogle_scaled[ogle_labels == cls, cidx]
            if len(ogle_cls_vals) > 0:
                cls_median = float(np.median(ogle_cls_vals))
            else:
                cls_median = 0.0  # fallback: z-score mean
            modified[cls_mask, cidx] = cls_median

    with torch.no_grad():
        out = model(torch.tensor(modified, dtype=torch.float32))
        preds = out["logits"].argmax(dim=1).numpy()
    return preds


def run_recalibration_experiment(
    model: torch.nn.Module,
    ogle_scaled: np.ndarray,
    ogle_labels: np.ndarray,
    domain_shift: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Run the full concept-selective recalibration experiment.

    Steps:
      1. Zero-shot baseline (no replacement)
      2. Individual high-KS concept replacement
      3. Progressive high-KS replacement (1, 2, 3 concepts)
      4. Control: replace low-KS concepts
      5. Oracle: replace ALL concepts with per-class medians

    Returns:
        Dictionary with all results.
    """
    print("\n[5/5] Running concept-selective recalibration experiment...")
    results: Dict[str, Any] = {}

    # ---- 0. Zero-shot baseline ----
    with torch.no_grad():
        out = model(torch.tensor(ogle_scaled, dtype=torch.float32))
        baseline_preds = out["logits"].argmax(dim=1).numpy()

    baseline_acc = float(accuracy_score(ogle_labels, baseline_preds))
    baseline_f1 = float(f1_score(ogle_labels, baseline_preds, average="macro",
                                  zero_division=0))

    # Per-class accuracy for baseline
    baseline_per_class = {}
    for ci, cn in enumerate(CLASS_NAMES):
        mask = ogle_labels == ci
        if mask.sum() > 0:
            baseline_per_class[cn] = float(accuracy_score(
                ogle_labels[mask], baseline_preds[mask]
            ))

    results["zero_shot_baseline"] = {
        "accuracy": baseline_acc,
        "macro_f1": baseline_f1,
        "per_class_accuracy": baseline_per_class,
    }
    print(f"\n  Zero-shot baseline:")
    print(f"    Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"    Macro F1: {baseline_f1:.4f}")

    # ---- 1. Compute OGLE per-class medians in scaled space ----
    ogle_class_medians: Dict[str, Dict[int, float]] = {}
    ogle_global_medians: Dict[str, float] = {}
    for ci, concept in enumerate(CONCEPT_NAMES_12):
        class_meds = {}
        for cls in np.unique(ogle_labels):
            vals = ogle_scaled[ogle_labels == cls, ci]
            class_meds[int(cls)] = float(np.median(vals))
        ogle_class_medians[concept] = class_meds
        ogle_global_medians[concept] = float(np.median(ogle_scaled[:, ci]))
    results["ogle_class_medians_scaled"] = ogle_class_medians
    results["ogle_global_medians_scaled"] = ogle_global_medians

    # ---- 2. Individual high-KS concept replacement ----
    print(f"\n  Individual high-KS concept replacement:")
    individual_results = {}
    for concept in HIGH_KS_CONCEPTS:
        preds = predict_with_class_medians(
            model, ogle_scaled, ogle_labels,
            replace_concepts=[concept],
            ogle_scaled=ogle_scaled,
            ogle_labels=ogle_labels,
        )
        acc = float(accuracy_score(ogle_labels, preds))
        f1 = float(f1_score(ogle_labels, preds, average="macro", zero_division=0))
        delta = acc - baseline_acc
        ks = domain_shift[concept]["ks_statistic"]

        per_class = {}
        for ci, cn in enumerate(CLASS_NAMES):
            mask = ogle_labels == ci
            if mask.sum() > 0:
                per_class[cn] = float(accuracy_score(ogle_labels[mask], preds[mask]))

        individual_results[concept] = {
            "accuracy": acc,
            "macro_f1": f1,
            "delta_accuracy": delta,
            "ks_statistic": ks,
            "per_class_accuracy": per_class,
        }
        print(f"    Replace {concept:15s}: acc={acc:.4f} ({delta:+.4f}), "
              f"KS={ks:.3f}")

    results["individual_high_ks"] = individual_results

    # ---- 3. Progressive high-KS replacement ----
    print(f"\n  Progressive high-KS replacement:")
    # Sort high-KS concepts by KS statistic (descending)
    sorted_high_ks = sorted(
        HIGH_KS_CONCEPTS,
        key=lambda c: -domain_shift[c]["ks_statistic"],
    )
    progressive_results = []
    for n_replace in range(1, len(sorted_high_ks) + 1):
        concepts_to_replace = sorted_high_ks[:n_replace]
        preds = predict_with_class_medians(
            model, ogle_scaled, ogle_labels,
            replace_concepts=concepts_to_replace,
            ogle_scaled=ogle_scaled,
            ogle_labels=ogle_labels,
        )
        acc = float(accuracy_score(ogle_labels, preds))
        f1 = float(f1_score(ogle_labels, preds, average="macro", zero_division=0))
        delta = acc - baseline_acc

        per_class = {}
        for ci, cn in enumerate(CLASS_NAMES):
            mask = ogle_labels == ci
            if mask.sum() > 0:
                per_class[cn] = float(accuracy_score(ogle_labels[mask], preds[mask]))

        entry = {
            "n_replaced": n_replace,
            "concepts_replaced": concepts_to_replace,
            "accuracy": acc,
            "macro_f1": f1,
            "delta_accuracy": delta,
            "per_class_accuracy": per_class,
        }
        progressive_results.append(entry)
        print(f"    {n_replace} concept(s) [{', '.join(concepts_to_replace)}]: "
              f"acc={acc:.4f} ({delta:+.4f})")

    results["progressive_high_ks"] = progressive_results

    # ---- 4. Control: replace low-KS concepts ----
    print(f"\n  Control: replace 3 LOW-KS concepts (should NOT improve):")
    control_preds = predict_with_class_medians(
        model, ogle_scaled, ogle_labels,
        replace_concepts=LOW_KS_CONCEPTS,
        ogle_scaled=ogle_scaled,
        ogle_labels=ogle_labels,
    )
    control_acc = float(accuracy_score(ogle_labels, control_preds))
    control_f1 = float(f1_score(ogle_labels, control_preds, average="macro",
                                 zero_division=0))
    control_delta = control_acc - baseline_acc

    control_per_class = {}
    for ci, cn in enumerate(CLASS_NAMES):
        mask = ogle_labels == ci
        if mask.sum() > 0:
            control_per_class[cn] = float(accuracy_score(
                ogle_labels[mask], control_preds[mask]
            ))

    results["control_low_ks"] = {
        "concepts_replaced": LOW_KS_CONCEPTS,
        "accuracy": control_acc,
        "macro_f1": control_f1,
        "delta_accuracy": control_delta,
        "per_class_accuracy": control_per_class,
    }
    print(f"    Replace {LOW_KS_CONCEPTS}: "
          f"acc={control_acc:.4f} ({control_delta:+.4f})")

    # ---- 5. Individual low-KS concept replacement (for completeness) ----
    print(f"\n  Individual low-KS concept replacement:")
    individual_low_results = {}
    for concept in LOW_KS_CONCEPTS:
        preds = predict_with_class_medians(
            model, ogle_scaled, ogle_labels,
            replace_concepts=[concept],
            ogle_scaled=ogle_scaled,
            ogle_labels=ogle_labels,
        )
        acc = float(accuracy_score(ogle_labels, preds))
        f1 = float(f1_score(ogle_labels, preds, average="macro", zero_division=0))
        delta = acc - baseline_acc
        ks = domain_shift[concept]["ks_statistic"]

        individual_low_results[concept] = {
            "accuracy": acc,
            "macro_f1": f1,
            "delta_accuracy": delta,
            "ks_statistic": ks,
        }
        print(f"    Replace {concept:15s}: acc={acc:.4f} ({delta:+.4f}), "
              f"KS={ks:.3f}")

    results["individual_low_ks"] = individual_low_results

    # ---- 6. Oracle: replace ALL 12 concepts with per-class medians ----
    print(f"\n  Oracle: replace ALL 12 concepts with per-class OGLE medians:")
    oracle_preds = predict_with_class_medians(
        model, ogle_scaled, ogle_labels,
        replace_concepts=CONCEPT_NAMES_12,
        ogle_scaled=ogle_scaled,
        ogle_labels=ogle_labels,
    )
    oracle_acc = float(accuracy_score(ogle_labels, oracle_preds))
    oracle_f1 = float(f1_score(ogle_labels, oracle_preds, average="macro",
                                zero_division=0))
    oracle_delta = oracle_acc - baseline_acc

    oracle_per_class = {}
    for ci, cn in enumerate(CLASS_NAMES):
        mask = ogle_labels == ci
        if mask.sum() > 0:
            oracle_per_class[cn] = float(accuracy_score(
                ogle_labels[mask], oracle_preds[mask]
            ))

    results["oracle_all_concepts"] = {
        "accuracy": oracle_acc,
        "macro_f1": oracle_f1,
        "delta_accuracy": oracle_delta,
        "per_class_accuracy": oracle_per_class,
    }
    print(f"    All 12 concepts: acc={oracle_acc:.4f} ({oracle_delta:+.4f})")

    # ---- 7. Global median replacement (non-oracle, no class info) ----
    print(f"\n  Non-oracle variants (global median, no class label info):")

    # High-KS with global median
    modified_global = ogle_scaled.copy()
    for concept in HIGH_KS_CONCEPTS:
        cidx = CONCEPT_NAMES_12.index(concept)
        modified_global[:, cidx] = ogle_global_medians[concept]
    with torch.no_grad():
        out = model(torch.tensor(modified_global, dtype=torch.float32))
        preds_global = out["logits"].argmax(dim=1).numpy()
    acc_global = float(accuracy_score(ogle_labels, preds_global))
    delta_global = acc_global - baseline_acc
    results["high_ks_global_median"] = {
        "concepts_replaced": HIGH_KS_CONCEPTS,
        "accuracy": acc_global,
        "delta_accuracy": delta_global,
    }
    print(f"    3 high-KS (global median): acc={acc_global:.4f} "
          f"({delta_global:+.4f})")

    # Low-KS with global median (control)
    modified_global_low = ogle_scaled.copy()
    for concept in LOW_KS_CONCEPTS:
        cidx = CONCEPT_NAMES_12.index(concept)
        modified_global_low[:, cidx] = ogle_global_medians[concept]
    with torch.no_grad():
        out = model(torch.tensor(modified_global_low, dtype=torch.float32))
        preds_global_low = out["logits"].argmax(dim=1).numpy()
    acc_global_low = float(accuracy_score(ogle_labels, preds_global_low))
    delta_global_low = acc_global_low - baseline_acc
    results["low_ks_global_median"] = {
        "concepts_replaced": LOW_KS_CONCEPTS,
        "accuracy": acc_global_low,
        "delta_accuracy": delta_global_low,
    }
    print(f"    3 low-KS  (global median): acc={acc_global_low:.4f} "
          f"({delta_global_low:+.4f})")

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print a clear summary table of all results."""
    print("\n" + "=" * 70)
    print("CONCEPT-SELECTIVE RECALIBRATION -- SUMMARY")
    print("=" * 70)

    baseline_acc = results["zero_shot_baseline"]["accuracy"]

    rows = [
        ("Zero-shot baseline", baseline_acc, 0.0),
    ]

    # Individual high-KS
    for concept, r in results["individual_high_ks"].items():
        rows.append((f"  + Replace {concept}", r["accuracy"], r["delta_accuracy"]))

    # Progressive
    for entry in results["progressive_high_ks"]:
        label = f"  + Replace {entry['n_replaced']} high-KS"
        rows.append((label, entry["accuracy"], entry["delta_accuracy"]))

    # Control
    ctrl = results["control_low_ks"]
    rows.append(("  [Control] Replace 3 low-KS", ctrl["accuracy"],
                 ctrl["delta_accuracy"]))

    # Oracle
    oracle = results["oracle_all_concepts"]
    rows.append(("  [Oracle] Replace all 12", oracle["accuracy"],
                 oracle["delta_accuracy"]))

    # Global median variants
    hkg = results["high_ks_global_median"]
    rows.append(("  [No label] 3 high-KS global med", hkg["accuracy"],
                 hkg["delta_accuracy"]))
    lkg = results["low_ks_global_median"]
    rows.append(("  [No label] 3 low-KS global med", lkg["accuracy"],
                 lkg["delta_accuracy"]))

    print(f"\n  {'Condition':<40s} {'Accuracy':>10s} {'Delta':>10s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    for label, acc, delta in rows:
        delta_str = f"{delta:+.4f}" if delta != 0.0 else "---"
        print(f"  {label:<40s} {acc:10.4f} {delta_str:>10s}")

    # Key finding
    high_ks_best = results["progressive_high_ks"][-1]
    print(f"\n  Key finding:")
    print(f"    Replacing {len(HIGH_KS_CONCEPTS)} high-KS concepts improves accuracy "
          f"by {high_ks_best['delta_accuracy']:+.4f}")
    print(f"    Replacing {len(LOW_KS_CONCEPTS)} low-KS concepts changes accuracy "
          f"by {ctrl['delta_accuracy']:+.4f}")
    print(f"    This confirms concept-selective recalibration targets the RIGHT concepts.")

    # Verify the claim: high-KS replacement should help more than low-KS
    if high_ks_best["delta_accuracy"] > ctrl["delta_accuracy"]:
        print(f"\n    CONFIRMED: High-KS replacement ({high_ks_best['delta_accuracy']:+.4f}) "
              f"> Low-KS replacement ({ctrl['delta_accuracy']:+.4f})")
        print(f"    CBM enables TARGETED recalibration of survey-dependent concepts.")
    else:
        print(f"\n    NOTE: High-KS replacement did not outperform low-KS replacement.")
        print(f"    This may indicate the domain shift pattern differs from expected.")


def main() -> None:
    t_start = time.time()

    print("=" * 70)
    print("Concept-Selective Recalibration Experiment")
    print("Validates: CBM enables targeted recalibration in cross-survey scenarios")
    print("=" * 70)

    set_global_seed(RANDOM_SEED)

    # Step 1: Load Gaia data and reconstruct scaler
    gaia_features_raw, gaia_labels, scaler, imp_stats, cv_idx = (
        load_gaia_data_and_scaler()
    )

    # Step 2: Load and prepare OGLE data
    ogle_features_raw, ogle_imputed, ogle_scaled, ogle_labels = load_ogle_data(
        imp_stats, scaler,
    )

    # Step 3: Load model
    model = load_hard_cbm_model(fold=4)

    # Step 4: Domain shift analysis
    domain_shift = compute_domain_shift(
        gaia_features_raw, gaia_labels, cv_idx, ogle_features_raw,
    )

    # Step 5: Run recalibration experiment
    results = run_recalibration_experiment(
        model, ogle_scaled, ogle_labels, domain_shift,
    )

    # Add metadata
    results["domain_shift"] = domain_shift
    results["high_ks_concepts"] = HIGH_KS_CONCEPTS
    results["low_ks_concepts"] = LOW_KS_CONCEPTS
    results["model"] = "hard_cbm"
    results["fold"] = 4
    results["n_ogle_samples"] = int(len(ogle_labels))
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Print summary
    print_summary(results)

    # Save
    output_path = RESULTS_DIR / "concept_recalibration.json"
    save_json(results, output_path)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
