#!/usr/bin/env python3
"""Compare SHAP feature importance from Random Forest with CBM concept importance rankings.

This script:
  1. Loads Gaia data and prepares train/val split (same as main experiments)
  2. Trains a Random Forest classifier on the last CV fold
  3. Computes SHAP TreeExplainer values on the validation set
  4. Computes global SHAP importance (mean |SHAP value| per concept)
  5. Loads CBM leave-one-out (LOO) importance from ablation results
  6. Loads CBM noise sensitivity from B3 importance consensus (if available)
  7. Computes Kendall tau correlation between SHAP and CBM rankings
  8. Saves results to results/supplementary/shap_comparison.json
  9. Prints a comparison table

Usage:
    python run_shap_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── Project imports ───────────────────────────────────────────────────────────
from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES_12,
    RANDOM_SEED,
    N_CV_FOLDS,
)
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.data.splits import create_full_split, create_cv_splits
from pipeline_utils import compute_imputation_stats, apply_imputation

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "real" / "gaia_all_features.parquet"
ABLATION_PATH = (
    PROJECT_ROOT / "results" / "ablation_detailed" / "all_ablation_results.json"
)
B3_PATH = PROJECT_ROOT / "results" / "supplementary" / "B3_importance_consensus.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "supplementary"
OUTPUT_PATH = RESULTS_DIR / "shap_comparison.json"


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Gaia data and return train/val splits for the last CV fold.

    Returns:
        (X_train, X_val, y_train, y_val) — standardised feature arrays.
    """
    print("[1/5] Loading data from", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    features = df[CONCEPT_NAMES_12].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)
    print(f"       Dataset: {len(df)} samples, {len(CONCEPT_NAMES_12)} concepts")

    # Train/test split (same as main experiments)
    split = create_full_split(labels, test_ratio=0.15, random_seed=RANDOM_SEED)
    cv_idx = split["cv_indices"]
    cv_features_raw = features[cv_idx]
    cv_labels = labels[cv_idx]
    print(f"       CV subset: {len(cv_idx)} samples")

    # Imputation (per-class median, fitted on CV data only)
    imp_stats = compute_imputation_stats(cv_features_raw, cv_labels)
    cv_features = apply_imputation(
        cv_features_raw, cv_labels, imp_stats, use_class_labels=True
    )

    # Use last fold (fold index 4) like main experiments
    folds = create_cv_splits(cv_labels, n_folds=N_CV_FOLDS, random_seed=RANDOM_SEED)
    train_idx, val_idx = folds[N_CV_FOLDS - 1]
    print(f"       Fold {N_CV_FOLDS}: train={len(train_idx)}, val={len(val_idx)}")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(cv_features[train_idx])
    X_val = scaler.transform(cv_features[val_idx])
    y_train = cv_labels[train_idx]
    y_val = cv_labels[val_idx]

    return X_train, X_val, y_train, y_val


def train_rf(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    """Train a Random Forest classifier with the same hyper-parameters as the baseline."""
    print("[2/5] Training Random Forest (500 trees)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"       Done in {elapsed:.1f}s  |  OOB-like train acc: "
          f"{rf.score(X_train, y_train):.4f}")
    return rf


def compute_shap_importance(
    rf: RandomForestClassifier, X_val: np.ndarray
) -> np.ndarray:
    """Compute global SHAP importance (mean |SHAP value| per concept).

    Returns:
        1-D array of shape (n_concepts,) with mean absolute SHAP values.
    """
    import shap

    print("[3/5] Computing SHAP values (TreeExplainer)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_val)
    elapsed = time.time() - t0
    print(f"       Done in {elapsed:.1f}s")

    # Debug shape
    if isinstance(shap_values, list):
        print(f"       SHAP type: list of {len(shap_values)} arrays, each shape {shap_values[0].shape}")
    else:
        print(f"       SHAP type: ndarray, shape {shap_values.shape}")

    # For multi-class RF, shap_values may be:
    #   - list of arrays: one (n_samples, n_features) per class
    #   - 3D array: (n_samples, n_features, n_classes) [newer shap] or (n_classes, n_samples, n_features)
    #   - 2D array: (n_samples, n_features) for binary/regression
    n_features = X_val.shape[1]

    if isinstance(shap_values, list):
        shap_importance = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    elif shap_values.ndim == 3:
        # Determine axis layout: (n_samples, n_features, n_classes) vs (n_classes, n_samples, n_features)
        if shap_values.shape[1] == n_features:
            # shape is (n_samples, n_features, n_classes) — new shap format
            shap_importance = np.abs(shap_values).mean(axis=(0, 2))
        elif shap_values.shape[2] == n_features:
            # shape is (n_classes, n_samples, n_features) — old format
            shap_importance = np.abs(shap_values).mean(axis=(0, 1))
        else:
            # fallback: average over all but feature dim
            shap_importance = np.abs(shap_values).reshape(-1, n_features).mean(axis=0)
    else:
        shap_importance = np.abs(shap_values).mean(axis=0)

    # Ensure 1D with correct length
    shap_importance = np.asarray(shap_importance).ravel()
    assert len(shap_importance) == n_features, f"SHAP importance length {len(shap_importance)} != {n_features}"
    return shap_importance


def load_loo_importance() -> Dict[str, float]:
    """Load CBM leave-one-out importance from ablation results.

    Returns:
        Dict mapping concept name -> accuracy drop when that concept is removed.
    """
    print("[4/5] Loading CBM LOO importance from", ABLATION_PATH)
    with open(ABLATION_PATH, "r") as f:
        ablation = json.load(f)

    baseline_acc = ablation["A0_baseline"]["aggregated"]["accuracy_mean"]
    loo = ablation.get("A2b_leave_one_out", {})

    loo_importance: Dict[str, float] = {}
    for concept, data in loo.items():
        loo_importance[concept] = baseline_acc - data["accuracy_mean"]

    print(f"       Baseline accuracy: {baseline_acc:.4f}")
    print(f"       Loaded LOO deltas for {len(loo_importance)} concepts")
    return loo_importance


def load_noise_sensitivity() -> Dict[str, float] | None:
    """Load CBM noise sensitivity from B3 importance consensus (if available).

    Returns:
        Dict mapping concept name -> noise sensitivity, or None if file missing.
    """
    if not B3_PATH.exists():
        print("       B3 importance consensus file not found — skipping noise sensitivity")
        return None

    with open(B3_PATH, "r") as f:
        b3 = json.load(f)

    noise_sens = b3.get("rankings", {}).get("noise_sensitivity", None)
    if noise_sens is None:
        print("       noise_sensitivity not found in B3 results — skipping")
        return None

    print(f"       Loaded noise sensitivity for {len(noise_sens)} concepts")
    return noise_sens


def rank_dict(d: Dict[str, float], concepts: List[str]) -> Dict[str, int]:
    """Rank concepts by value (highest value = rank 1)."""
    vals = [(c, d.get(c, 0.0)) for c in concepts]
    vals.sort(key=lambda x: x[1], reverse=True)
    return {c: rank + 1 for rank, (c, _) in enumerate(vals)}


def compute_kendall_tau(
    ranks_a: Dict[str, int], ranks_b: Dict[str, int], concepts: List[str]
) -> Tuple[float, float]:
    """Compute Kendall tau between two rank dictionaries."""
    a = [ranks_a[c] for c in concepts]
    b = [ranks_b[c] for c in concepts]
    tau, p = kendalltau(a, b)
    return float(tau), float(p)


def main() -> None:
    set_global_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1-3: Data, RF, SHAP ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = load_data()
    rf = train_rf(X_train, y_train)

    val_acc = rf.score(X_val, y_val)
    print(f"       RF validation accuracy: {val_acc:.4f}")

    shap_importance = compute_shap_importance(rf, X_val)

    # ── Step 4: LOO and noise sensitivity ─────────────────────────────────────
    loo_importance = load_loo_importance()
    noise_sensitivity = load_noise_sensitivity()

    # ── Step 5: Build rankings ────────────────────────────────────────────────
    print("[5/5] Computing rankings and Kendall tau correlations...")

    concepts = CONCEPT_NAMES_12

    # SHAP ranking
    shap_dict = {c: float(shap_importance[i]) for i, c in enumerate(concepts)}
    shap_ranks = rank_dict(shap_dict, concepts)

    # LOO ranking
    loo_ranks = rank_dict(loo_importance, concepts)

    # Kendall tau: SHAP vs LOO
    tau_shap_loo, p_shap_loo = compute_kendall_tau(shap_ranks, loo_ranks, concepts)
    print(f"       Kendall tau (SHAP vs LOO): tau={tau_shap_loo:.4f}, p={p_shap_loo:.4f}")

    # Kendall tau: SHAP vs noise sensitivity (if available)
    tau_shap_noise, p_shap_noise = None, None
    noise_ranks = None
    if noise_sensitivity is not None:
        noise_ranks = rank_dict(noise_sensitivity, concepts)
        tau_shap_noise, p_shap_noise = compute_kendall_tau(
            shap_ranks, noise_ranks, concepts
        )
        print(f"       Kendall tau (SHAP vs Noise): tau={tau_shap_noise:.4f}, "
              f"p={p_shap_noise:.4f}")

    # ── Build output ──────────────────────────────────────────────────────────
    shap_ranking = sorted(
        [
            {"concept": c, "shap_importance": shap_dict[c], "rank": shap_ranks[c]}
            for c in concepts
        ],
        key=lambda x: x["rank"],
    )

    loo_ranking = sorted(
        [
            {
                "concept": c,
                "loo_delta": loo_importance.get(c, 0.0),
                "rank": loo_ranks[c],
            }
            for c in concepts
        ],
        key=lambda x: x["rank"],
    )

    comparison_table = sorted(
        [
            {
                "concept": c,
                "shap_rank": shap_ranks[c],
                "loo_rank": loo_ranks[c],
                "noise_rank": noise_ranks[c] if noise_ranks else None,
            }
            for c in concepts
        ],
        key=lambda x: x["shap_rank"],
    )

    results: Dict[str, Any] = {
        "rf_val_accuracy": val_acc,
        "shap_ranking": shap_ranking,
        "loo_ranking": loo_ranking,
        "kendall_tau_shap_vs_loo": {"tau": tau_shap_loo, "p_value": p_shap_loo},
        "comparison_table": comparison_table,
    }

    if tau_shap_noise is not None:
        results["kendall_tau_shap_vs_noise"] = {
            "tau": tau_shap_noise,
            "p_value": p_shap_noise,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SHAP (RF) vs CBM Concept Importance Comparison")
    print("=" * 72)

    header = f"{'Concept':<18} {'SHAP val':>10} {'SHAP rank':>10} {'LOO delta':>10} {'LOO rank':>10}"
    if noise_ranks:
        header += f" {'Noise rank':>11}"
    print(header)
    print("-" * len(header))

    for row in comparison_table:
        c = row["concept"]
        sv = shap_dict[c]
        sr = row["shap_rank"]
        ld = loo_importance.get(c, 0.0)
        lr = row["loo_rank"]
        line = f"{c:<18} {sv:>10.6f} {sr:>10d} {ld:>10.6f} {lr:>10d}"
        if noise_ranks:
            nr = row["noise_rank"]
            line += f" {nr:>11d}"
        print(line)

    print("-" * len(header))
    print(f"\nKendall tau (SHAP vs LOO):   tau = {tau_shap_loo:.4f},  p = {p_shap_loo:.4f}")
    if tau_shap_noise is not None:
        print(f"Kendall tau (SHAP vs Noise): tau = {tau_shap_noise:.4f},  "
              f"p = {p_shap_noise:.4f}")

    # Interpretation
    print("\nInterpretation:")
    if abs(tau_shap_loo) >= 0.6:
        print("  Strong agreement between RF-SHAP and CBM-LOO importance rankings.")
    elif abs(tau_shap_loo) >= 0.4:
        print("  Moderate agreement between RF-SHAP and CBM-LOO importance rankings.")
    elif abs(tau_shap_loo) >= 0.2:
        print("  Weak agreement between RF-SHAP and CBM-LOO importance rankings.")
    else:
        print("  No meaningful agreement between RF-SHAP and CBM-LOO importance rankings.")

    if p_shap_loo < 0.05:
        print(f"  The correlation is statistically significant (p = {p_shap_loo:.4f} < 0.05).")
    else:
        print(f"  The correlation is NOT statistically significant (p = {p_shap_loo:.4f} >= 0.05).")

    print()


if __name__ == "__main__":
    main()
