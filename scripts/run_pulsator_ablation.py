#!/usr/bin/env python
"""
Pulsator-Only Concept Ablation for CBM Variable Star Classification.

Motivation:
  The full 6-class ablation shows R31 as the most important concept
  (-2.72% accuracy drop when removed). However, R31 is NaN for 50%
  of sources (ECL, DSCT/SXPhe, MIRA_SR) and is filled via per-class
  median imputation. Its importance may therefore be inflated by acting
  as a binary "has Fourier tables" signal rather than reflecting genuine
  astrophysical diagnostic power.

  By restricting ablation to the 3 pulsator classes (RRAB, RRC, DCEP)
  -- where R31 is always available from Gaia type-specific Fourier
  tables -- we can isolate R31's genuine diagnostic contribution.

Experiments:
  P0:  Pulsator baseline (all 12 concepts, 3 classes)
  P1:  Leave-one-out concept ablation (12 runs)
  P2:  Forward greedy selection
  P3:  Backward greedy elimination
  P4:  Comparison with full 6-class LOO rankings

Usage:
  python scripts/run_pulsator_ablation.py
  python scripts/run_pulsator_ablation.py --experiments P0 P1 P4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_NAMES_12,
    RANDOM_SEED,
)
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.training.cross_val import run_cross_validation
from pipeline_utils import compute_imputation_stats, apply_imputation
from cbm_variable_stars.data.splits import create_full_split

# ======================================================================
# Configuration
# ======================================================================
DATA_PATH = PROJECT_ROOT / "data" / "real" / "gaia_all_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "pulsator_ablation"
PREV_RESULTS_PATH = PROJECT_ROOT / "results" / "ablation_detailed" / "all_ablation_results.json"
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 256
N_FOLDS = 5

PULSATOR_CLASSES = ["RRAB", "RRC", "DCEP"]
PULSATOR_LABELS = [0, 1, 2]  # First 3 classes in CLASS_NAMES
N_PULSATOR_CLASSES = 3

# A2/A3 early stopping thresholds
FORWARD_STOP_RATIO = 0.99
BACKWARD_STOP_DROP = 0.05


# ======================================================================
# Shared utilities
# ======================================================================

def load_pulsator_data():
    """
    Load data and filter to pulsator classes only (RRAB, RRC, DCEP).

    Returns imputed CV features, CV labels, test features, test labels.
    Labels are remapped to 0, 1, 2 for the 3-class problem.
    """
    df = pd.read_parquet(DATA_PATH)
    features = df[CONCEPT_NAMES_12].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)

    # Filter to pulsator classes only
    mask = np.isin(labels, PULSATOR_LABELS)
    features = features[mask]
    labels = labels[mask]

    # Remap labels to 0, 1, 2 (they already are 0, 1, 2 since these
    # are the first 3 classes, but be explicit for safety)
    label_remap = {0: 0, 1: 1, 2: 2}
    labels = np.array([label_remap[l] for l in labels], dtype=np.int64)

    n_total = len(labels)
    for cls_idx, cls_name in enumerate(PULSATOR_CLASSES):
        n_cls = np.sum(labels == cls_idx)
        print(f"    {cls_name} (label {cls_idx}): {n_cls} sources")
    print(f"    Total pulsator sources: {n_total}")

    # Verify NaN pattern: pulsators should have Fourier values
    r31_col_idx = CONCEPT_NAMES_12.index("R31")
    n_r31_nan = np.sum(np.isnan(features[:, r31_col_idx]))
    pct_r31_nan = 100.0 * n_r31_nan / n_total
    print(f"    R31 NaN count: {n_r31_nan}/{n_total} ({pct_r31_nan:.1f}%)")

    # Split into CV/test (85%/15%)
    split = create_full_split(labels, test_ratio=0.15)
    cv_idx = split["cv_indices"]
    test_idx = split["test_indices"]

    # Fit imputation on CV data only
    imp_stats = compute_imputation_stats(features[cv_idx], labels[cv_idx])
    cv_features = apply_imputation(
        features[cv_idx], labels[cv_idx], imp_stats, use_class_labels=True,
    )
    test_features = apply_imputation(
        features[test_idx], labels[test_idx], imp_stats, use_class_labels=False,
    )
    cv_labels = labels[cv_idx]
    test_labels = labels[test_idx]

    print(f"    CV set: {len(cv_labels)} sources")
    print(f"    Test set: {len(test_labels)} sources")

    # Verify no NaN remains after imputation
    n_nan_cv = np.sum(np.isnan(cv_features))
    n_nan_test = np.sum(np.isnan(test_features))
    print(f"    NaN remaining after imputation: CV={n_nan_cv}, Test={n_nan_test}")

    return cv_features, cv_labels, test_features, test_labels


def subset_features(features: np.ndarray, concept_names: list[str]) -> np.ndarray:
    """Extract subset of features by concept name."""
    indices = [CONCEPT_NAMES_12.index(c) for c in concept_names]
    return features[:, indices]


def run_cv(features, labels, model_name="hard_cbm", model_kwargs=None, tag=""):
    """Run 5-fold CV on pulsator data (3 classes) and return results."""
    if model_kwargs is None:
        model_kwargs = {}
    # Ensure num_classes=3 for the pulsator subset
    if "num_classes" not in model_kwargs:
        model_kwargs["num_classes"] = N_PULSATOR_CLASSES
    result = run_cross_validation(
        features=features,
        labels=labels,
        model_name=model_name,
        model_kwargs=model_kwargs,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        output_dir=str(RESULTS_DIR / tag),
    )
    return result


def print_detailed_result(result, label=""):
    """Print detailed per-fold and per-class results."""
    agg = result["aggregated"]
    folds = result.get("fold_results", [])

    print(f"\n  [{label}] Aggregated (5-fold CV):")
    print(f"    Accuracy:    {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")
    print(f"    Macro F1:    {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")

    if "per_class_f1_mean" in agg:
        print(f"\n    Per-class F1 (mean +/- std):")
        for i, cls in enumerate(PULSATOR_CLASSES):
            if i < len(agg["per_class_f1_mean"]):
                mean_f1 = agg["per_class_f1_mean"][i]
                std_f1 = agg["per_class_f1_std"][i]
                print(f"      {cls:<14} {mean_f1:.4f} +/- {std_f1:.4f}")

    print(f"\n    Per-fold breakdown:")
    print(f"    {'Fold':<6} {'Acc':>8} {'F1':>8} {'BestEp':>8} {'Time(s)':>8}")
    print(f"    {'----':<6} {'----':>8} {'----':>8} {'------':>8} {'-------':>8}")
    for fr in folds:
        m = fr["metrics"]
        print(f"    {fr['fold']+1:<6} {m['val_accuracy']:>8.4f} {m['val_macro_f1']:>8.4f} "
              f"{fr['best_epoch']:>8} {fr['training_time']:>8.1f}")

    return agg


def load_previous_full_results() -> dict:
    """Load LOO results from the full 6-class ablation for comparison."""
    if PREV_RESULTS_PATH.exists():
        with open(PREV_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  WARNING: Previous full 6-class results not found at {PREV_RESULTS_PATH}")
    return {}


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


# ======================================================================
# P0: Pulsator Baseline
# ======================================================================

def run_p0_baseline(cv_features, cv_labels):
    """Train HardCBM with all 12 concepts on the 3-class pulsator subset."""
    print("\n" + "=" * 80)
    print("  P0: Pulsator Baseline — All 12 concepts, 3 classes (RRAB, RRC, DCEP)")
    print("=" * 80)
    t0 = time.time()

    result = run_cv(
        cv_features, cv_labels, "hard_cbm",
        model_kwargs={"num_concepts": 12, "num_classes": N_PULSATOR_CLASSES},
        tag="P0_baseline",
    )
    agg = print_detailed_result(result, "P0 baseline")
    print(f"\n  P0 time: {time.time() - t0:.1f}s")

    return result, agg


# ======================================================================
# P1: Leave-One-Out Concept Ablation
# ======================================================================

def run_p1_leave_one_out(cv_features, cv_labels, baseline_agg):
    """
    For each of 12 concepts, train HardCBM on remaining 11 and record accuracy.
    This is the core experiment to isolate R31's genuine importance in pulsators.
    """
    print("\n" + "=" * 80)
    print("  P1: Leave-One-Out Concept Ablation — Pulsators Only")
    print("=" * 80)
    t_total = time.time()

    baseline_acc = baseline_agg["accuracy_mean"]
    loo_results = {}

    for concept_to_remove in CONCEPT_NAMES_12:
        remaining = [c for c in CONCEPT_NAMES_12 if c != concept_to_remove]
        n_remaining = len(remaining)

        print(f"\n  --- Removing: {concept_to_remove} ({n_remaining} remaining) ---")
        t0 = time.time()

        features_sub = subset_features(cv_features, remaining)
        result = run_cv(
            features_sub, cv_labels, "hard_cbm",
            model_kwargs={"num_concepts": n_remaining, "num_classes": N_PULSATOR_CLASSES},
            tag=f"P1_no_{concept_to_remove}",
        )
        agg = result["aggregated"]

        delta_acc = agg["accuracy_mean"] - baseline_acc
        delta_f1 = agg["macro_f1_mean"] - baseline_agg["macro_f1_mean"]

        print(f"    Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
              f"F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")
        print(f"    Delta vs baseline: Acc={delta_acc:+.4f}, F1={delta_f1:+.4f}")
        print(f"    Time: {time.time()-t0:.1f}s")

        loo_results[concept_to_remove] = {
            "remaining_concepts": remaining,
            "n_remaining": n_remaining,
            "accuracy_mean": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"],
            "macro_f1_mean": agg["macro_f1_mean"],
            "macro_f1_std": agg["macro_f1_std"],
            "delta_accuracy": delta_acc,
            "delta_macro_f1": delta_f1,
            "per_class_f1_mean": agg.get("per_class_f1_mean", []),
            "per_class_f1_std": agg.get("per_class_f1_std", []),
        }

    # Summary sorted by impact (most negative delta = most important)
    print(f"\n  Leave-One-Out Summary (sorted by impact):")
    print(f"  {'Rank':<6} {'Removed':<18} {'Accuracy':>10} {'Delta Acc':>10} {'F1':>10} {'Delta F1':>10}")
    print(f"  {'=' * 6} {'=' * 18} {'=' * 10} {'=' * 10} {'=' * 10} {'=' * 10}")
    sorted_loo = sorted(loo_results.items(), key=lambda x: x[1]["delta_accuracy"])
    for rank, (concept, r) in enumerate(sorted_loo, 1):
        marker = " <-- R31" if concept == "R31" else ""
        print(f"  {rank:<6} {concept:<18} {r['accuracy_mean']:>10.4f} "
              f"{r['delta_accuracy']:>+10.4f} {r['macro_f1_mean']:>10.4f} "
              f"{r['delta_macro_f1']:>+10.4f}{marker}")

    print(f"\n  P1 total time: {time.time() - t_total:.1f}s")
    return loo_results


# ======================================================================
# P2: Forward Greedy Selection
# ======================================================================

def run_p2_forward_selection(cv_features, cv_labels, baseline_agg):
    """
    Starting from 0 concepts, greedily add the concept that maximizes
    accuracy on the pulsator 3-class problem.
    """
    print("\n" + "=" * 80)
    print("  P2: Forward Greedy Selection — Pulsators Only")
    print("=" * 80)
    t_total = time.time()

    baseline_acc = baseline_agg["accuracy_mean"]
    stop_threshold = baseline_acc * FORWARD_STOP_RATIO

    selected = []
    remaining_pool = list(CONCEPT_NAMES_12)
    steps = []
    cv_run_count = 0

    for step in range(1, 13):
        print(f"\n  === Step {step}/12: selecting concept #{step} ===")
        print(f"      Current set: {selected if selected else '(empty)'}")
        t_step = time.time()

        best_concept = None
        best_acc = -1.0
        best_result_agg = None
        candidate_scores = {}

        for candidate in remaining_pool:
            trial_set = selected + [candidate]
            features_sub = subset_features(cv_features, trial_set)
            result = run_cv(
                features_sub, cv_labels, "hard_cbm",
                model_kwargs={"num_concepts": len(trial_set), "num_classes": N_PULSATOR_CLASSES},
                tag=f"P2_step{step}_{candidate}",
            )
            agg = result["aggregated"]
            cv_run_count += 1
            acc = agg["accuracy_mean"]
            candidate_scores[candidate] = acc

            print(f"      + {candidate}: Acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_concept = candidate
                best_result_agg = agg

        selected.append(best_concept)
        remaining_pool.remove(best_concept)

        marginal_gain = best_acc - (steps[-1]["accuracy"] if steps else 0.0)

        step_info = {
            "step": step,
            "added_concept": best_concept,
            "selected_concepts": list(selected),
            "n_concepts": len(selected),
            "accuracy": best_acc,
            "accuracy_std": best_result_agg["accuracy_std"],
            "macro_f1": best_result_agg["macro_f1_mean"],
            "macro_f1_std": best_result_agg["macro_f1_std"],
            "marginal_gain": marginal_gain,
            "retained_pct": best_acc / baseline_acc * 100 if baseline_acc > 0 else 0.0,
            "candidate_scores": candidate_scores,
            "per_class_f1_mean": best_result_agg.get("per_class_f1_mean", []),
        }
        steps.append(step_info)

        print(f"\n      Selected: {best_concept} (Acc={best_acc:.4f}, +{marginal_gain:+.4f})")
        print(f"      Retained: {step_info['retained_pct']:.1f}% of baseline")
        print(f"      Step time: {time.time()-t_step:.1f}s, CV runs so far: {cv_run_count}")

        if best_acc >= stop_threshold and step < 12:
            print(f"\n      Reached {FORWARD_STOP_RATIO*100:.0f}% of baseline accuracy")
            step_info["reached_threshold"] = True

    # Summary
    print(f"\n  Forward Selection Order (Pulsators):")
    print(f"  {'Step':<6} {'Added':<18} {'#Concepts':>10} {'Accuracy':>10} {'Marginal':>10} {'Retained%':>10}")
    print(f"  {'=' * 6} {'=' * 18} {'=' * 10} {'=' * 10} {'=' * 10} {'=' * 10}")
    for s in steps:
        marker = " *" if s["added_concept"] == "R31" else ""
        print(f"  {s['step']:<6} {s['added_concept']:<18} {s['n_concepts']:>10} "
              f"{s['accuracy']:>10.4f} {s['marginal_gain']:>+10.4f} {s['retained_pct']:>10.1f}{marker}")

    r31_step = next((s["step"] for s in steps if s["added_concept"] == "R31"), None)
    print(f"\n  R31 added at step: {r31_step}/12")
    print(f"  Total CV runs: {cv_run_count}")
    print(f"  P2 total time: {time.time() - t_total:.1f}s")

    return {
        "selection_order": [s["added_concept"] for s in steps],
        "steps": steps,
        "total_cv_runs": cv_run_count,
        "r31_forward_rank": r31_step,
    }


# ======================================================================
# P3: Backward Greedy Elimination
# ======================================================================

def run_p3_backward_elimination(cv_features, cv_labels, baseline_agg):
    """
    Starting from all 12 concepts, greedily remove the least important
    concept on the pulsator 3-class problem.
    """
    print("\n" + "=" * 80)
    print("  P3: Backward Greedy Elimination — Pulsators Only")
    print("=" * 80)
    t_total = time.time()

    baseline_acc = baseline_agg["accuracy_mean"]
    stop_threshold = baseline_acc - BACKWARD_STOP_DROP

    remaining = list(CONCEPT_NAMES_12)
    steps = []
    cv_run_count = 0

    # Step 0 is the full baseline
    steps.append({
        "step": 0,
        "removed_concept": None,
        "remaining_concepts": list(remaining),
        "n_concepts": 12,
        "accuracy": baseline_acc,
        "accuracy_std": baseline_agg["accuracy_std"],
        "macro_f1": baseline_agg["macro_f1_mean"],
        "macro_f1_std": baseline_agg["macro_f1_std"],
        "drop_from_previous": 0.0,
    })

    for step in range(1, 12):
        print(f"\n  === Step {step}/11: removing concept (currently {len(remaining)} remaining) ===")
        t_step = time.time()

        best_to_remove = None
        best_acc_after = -1.0
        best_result_agg = None
        candidate_scores = {}

        for candidate in remaining:
            trial_set = [c for c in remaining if c != candidate]
            features_sub = subset_features(cv_features, trial_set)
            result = run_cv(
                features_sub, cv_labels, "hard_cbm",
                model_kwargs={"num_concepts": len(trial_set), "num_classes": N_PULSATOR_CLASSES},
                tag=f"P3_step{step}_no_{candidate}",
            )
            agg = result["aggregated"]
            cv_run_count += 1
            acc = agg["accuracy_mean"]
            candidate_scores[candidate] = acc

            print(f"      - {candidate}: Acc={acc:.4f}")

            # Best to remove = concept whose removal causes least damage
            if acc > best_acc_after:
                best_acc_after = acc
                best_to_remove = candidate
                best_result_agg = agg

        remaining.remove(best_to_remove)

        drop_from_prev = best_acc_after - steps[-1]["accuracy"]

        step_info = {
            "step": step,
            "removed_concept": best_to_remove,
            "remaining_concepts": list(remaining),
            "n_concepts": len(remaining),
            "accuracy": best_acc_after,
            "accuracy_std": best_result_agg["accuracy_std"],
            "macro_f1": best_result_agg["macro_f1_mean"],
            "macro_f1_std": best_result_agg["macro_f1_std"],
            "drop_from_previous": drop_from_prev,
            "drop_from_baseline": best_acc_after - baseline_acc,
            "retained_pct": best_acc_after / baseline_acc * 100 if baseline_acc > 0 else 0.0,
            "candidate_scores": candidate_scores,
            "per_class_f1_mean": best_result_agg.get("per_class_f1_mean", []),
        }
        steps.append(step_info)

        marker = " <-- R31" if best_to_remove == "R31" else ""
        print(f"\n      Removed: {best_to_remove} (Acc={best_acc_after:.4f}, drop={drop_from_prev:+.4f}){marker}")
        print(f"      Remaining ({len(remaining)}): {remaining}")
        print(f"      Step time: {time.time()-t_step:.1f}s, CV runs so far: {cv_run_count}")

        if best_acc_after < stop_threshold:
            print(f"\n      Accuracy dropped below threshold ({stop_threshold:.4f}) — stopping")
            break

    # Summary
    print(f"\n  Backward Elimination Order (Pulsators):")
    print(f"  {'Step':<6} {'Removed':<18} {'#Remain':>8} {'Accuracy':>10} {'Drop':>10} {'Retained%':>10}")
    print(f"  {'=' * 6} {'=' * 18} {'=' * 8} {'=' * 10} {'=' * 10} {'=' * 10}")
    for s in steps:
        removed = s["removed_concept"] or "(baseline)"
        marker = " *" if s["removed_concept"] == "R31" else ""
        print(f"  {s['step']:<6} {removed:<18} {s['n_concepts']:>8} "
              f"{s['accuracy']:>10.4f} {s['drop_from_previous']:>+10.4f} "
              f"{s.get('retained_pct', 100.0):>10.1f}{marker}")

    elimination_order = [s["removed_concept"] for s in steps[1:]]
    r31_elim_step = None
    for s in steps[1:]:
        if s["removed_concept"] == "R31":
            r31_elim_step = s["step"]
            break

    print(f"\n  R31 eliminated at step: {r31_elim_step}/11 (later = more important)")
    print(f"  Total CV runs: {cv_run_count}")
    print(f"  P3 total time: {time.time() - t_total:.1f}s")

    return {
        "elimination_order": elimination_order,
        "steps": steps,
        "total_cv_runs": cv_run_count,
        "r31_elimination_step": r31_elim_step,
    }


# ======================================================================
# P4: Comparison with Full 6-Class Results
# ======================================================================

def run_p4_comparison(pulsator_loo, pulsator_baseline_agg):
    """
    Compare concept importance rankings between pulsator-only and full 6-class.
    This is the key analysis: does R31 ranking change when the imputation
    confound is removed?
    """
    print("\n" + "=" * 80)
    print("  P4: Pulsator-Only vs Full 6-Class — R31 Confound Analysis")
    print("=" * 80)

    # Load full 6-class results
    prev = load_previous_full_results()
    full_baseline_agg = prev.get("A0_baseline", {}).get("aggregated", {})
    full_loo = prev.get("A2b_leave_one_out", {})

    if not full_loo:
        print("  WARNING: Full 6-class LOO results not found. Cannot compare.")
        print("  Run scripts/run_ablation_detailed.py first to generate A2b results.")
        return {"error": "full_6class_results_not_found"}

    full_baseline_acc = full_baseline_agg.get("accuracy_mean", 0.0)
    pulsator_baseline_acc = pulsator_baseline_agg["accuracy_mean"]

    # Build ranking tables
    # Full 6-class ranking (delta accuracy when concept removed)
    full_ranking = []
    for concept in CONCEPT_NAMES_12:
        if concept in full_loo:
            data = full_loo[concept]
            # Handle both key formats (delta_accuracy or compute from accuracy_mean)
            if "delta_accuracy" in data:
                delta = data["delta_accuracy"]
            else:
                delta = data.get("accuracy_mean", full_baseline_acc) - full_baseline_acc
            full_ranking.append((concept, delta))
    full_ranking.sort(key=lambda x: x[1])  # Most negative = most important

    # Pulsator ranking
    pulsator_ranking = []
    for concept in CONCEPT_NAMES_12:
        if concept in pulsator_loo:
            data = pulsator_loo[concept]
            delta = data["delta_accuracy"]
            pulsator_ranking.append((concept, delta))
    pulsator_ranking.sort(key=lambda x: x[1])

    # Print side-by-side comparison
    print(f"\n  Baseline accuracies:")
    print(f"    Full 6-class:    {full_baseline_acc:.4f}")
    print(f"    Pulsator 3-class: {pulsator_baseline_acc:.4f}")

    print(f"\n  {'':=<90}")
    print(f"  Concept Importance Ranking Comparison")
    print(f"  {'':=<90}")
    print(f"  {'Rank':<6} {'Full 6-Class':<25}{'':>5} {'Pulsator 3-Class':<25}")
    print(f"  {'':<6} {'Concept':<15} {'Delta':>8}{'':>5} {'Concept':<15} {'Delta':>8}")
    print(f"  {'=' * 6} {'=' * 15} {'=' * 8}{'':>5} {'=' * 15} {'=' * 8}")

    max_len = max(len(full_ranking), len(pulsator_ranking))
    for rank in range(max_len):
        full_str = ""
        puls_str = ""
        if rank < len(full_ranking):
            c, d = full_ranking[rank]
            marker = " ***" if c == "R31" else ""
            full_str = f"{c:<15} {d:>+8.4f}{marker}"
        if rank < len(pulsator_ranking):
            c, d = pulsator_ranking[rank]
            marker = " ***" if c == "R31" else ""
            puls_str = f"{c:<15} {d:>+8.4f}{marker}"
        print(f"  {rank+1:<6} {full_str:<29} {puls_str}")

    # R31 specific analysis
    r31_full_rank = None
    r31_full_delta = None
    for rank, (concept, delta) in enumerate(full_ranking):
        if concept == "R31":
            r31_full_rank = rank + 1
            r31_full_delta = delta
            break

    r31_puls_rank = None
    r31_puls_delta = None
    for rank, (concept, delta) in enumerate(pulsator_ranking):
        if concept == "R31":
            r31_puls_rank = rank + 1
            r31_puls_delta = delta
            break

    print(f"\n  {'':=<70}")
    print(f"  R31 Confound Analysis")
    print(f"  {'':=<70}")
    print(f"  Full 6-class:     rank {r31_full_rank}/12, delta_acc = {r31_full_delta:+.4f}")
    print(f"  Pulsator 3-class: rank {r31_puls_rank}/12, delta_acc = {r31_puls_delta:+.4f}")

    if r31_full_rank is not None and r31_puls_rank is not None:
        rank_change = r31_puls_rank - r31_full_rank
        if rank_change > 0:
            print(f"\n  R31 dropped {rank_change} positions (rank {r31_full_rank} -> {r31_puls_rank})")
            print(f"  INTERPRETATION: R31's importance in the full 6-class setting is PARTIALLY")
            print(f"  inflated by imputation confound. When restricted to pulsators (where R31")
            print(f"  is always available), its diagnostic power is lower than the full-sample")
            print(f"  ranking suggests.")
        elif rank_change < 0:
            print(f"\n  R31 rose {-rank_change} positions (rank {r31_full_rank} -> {r31_puls_rank})")
            print(f"  INTERPRETATION: R31's genuine astrophysical importance for pulsator")
            print(f"  classification is even HIGHER than the full-sample ranking suggests.")
            print(f"  The imputation confound does not inflate R31's importance.")
        else:
            print(f"\n  R31 rank unchanged (rank {r31_full_rank} in both settings)")
            print(f"  INTERPRETATION: R31's importance is robust. Imputation confound has")
            print(f"  minimal effect on its ranking.")

        delta_ratio = abs(r31_puls_delta / r31_full_delta) if r31_full_delta != 0 else float("inf")
        print(f"\n  Accuracy drop ratio (pulsator/full): {delta_ratio:.2f}")
        if delta_ratio < 0.5:
            print(f"  The pulsator-only drop ({r31_puls_delta:+.4f}) is < 50% of the full drop "
                  f"({r31_full_delta:+.4f}),")
            print(f"  suggesting substantial confound inflation in the 6-class setting.")
        elif delta_ratio > 1.5:
            print(f"  The pulsator-only drop ({r31_puls_delta:+.4f}) exceeds the full drop "
                  f"({r31_full_delta:+.4f}),")
            print(f"  confirming R31 has genuine diagnostic power for pulsator subclassification.")
        else:
            print(f"  Drops are comparable ({r31_puls_delta:+.4f} vs {r31_full_delta:+.4f}),")
            print(f"  suggesting moderate confound effect.")

    # Compute Kendall tau correlation between rankings
    try:
        from scipy.stats import kendalltau
        full_rank_order = [c for c, _ in full_ranking]
        puls_rank_order = [c for c, _ in pulsator_ranking]
        # Align to same concept set
        common_concepts = [c for c in full_rank_order if c in puls_rank_order]
        rank1 = [full_rank_order.index(c) for c in common_concepts]
        rank2 = [puls_rank_order.index(c) for c in common_concepts]
        tau, p_value = kendalltau(rank1, rank2)
        print(f"\n  Kendall tau correlation between rankings: tau={tau:.4f}, p={p_value:.4f}")
        if tau > 0.6:
            print(f"  Rankings are significantly correlated (tau > 0.6) — concept importance")
            print(f"  is largely consistent between pulsator-only and full settings.")
        elif tau > 0.3:
            print(f"  Rankings show moderate correlation — some concepts change importance")
            print(f"  when the imputation confound is removed.")
        else:
            print(f"  Rankings show weak/no correlation — the imputation confound")
            print(f"  substantially distorts concept importance in the full 6-class setting.")
    except ImportError:
        tau, p_value = None, None
        print("\n  scipy not available — skipping Kendall tau computation")

    comparison = {
        "full_6class_baseline_acc": full_baseline_acc,
        "pulsator_3class_baseline_acc": pulsator_baseline_acc,
        "full_6class_ranking": [{"concept": c, "delta_accuracy": d, "rank": i+1}
                                 for i, (c, d) in enumerate(full_ranking)],
        "pulsator_3class_ranking": [{"concept": c, "delta_accuracy": d, "rank": i+1}
                                     for i, (c, d) in enumerate(pulsator_ranking)],
        "r31_analysis": {
            "full_rank": r31_full_rank,
            "full_delta": r31_full_delta,
            "pulsator_rank": r31_puls_rank,
            "pulsator_delta": r31_puls_delta,
            "rank_change": (r31_puls_rank - r31_full_rank) if r31_full_rank and r31_puls_rank else None,
            "delta_ratio": float(delta_ratio) if r31_full_rank and r31_puls_rank and r31_full_delta != 0 else None,
        },
        "kendall_tau": {
            "tau": float(tau) if tau is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
        },
    }

    return comparison


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pulsator-Only Concept Ablation for CBM Variable Star Classification"
    )
    parser.add_argument(
        "--experiments", nargs="*",
        default=["P0", "P1", "P2", "P3", "P4"],
        help="Which experiments to run (e.g., --experiments P0 P1 P4)"
    )
    args = parser.parse_args()
    experiments = [e.upper() for e in args.experiments]

    set_global_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  CBM Variable Star Classification — Pulsator-Only Concept Ablation")
    print(f"  Purpose: Decouple R31 imputation confound from genuine astrophysical importance")
    print(f"  Classes: {PULSATOR_CLASSES} (N~9000, 3000 per class)")
    print(f"  Experiments to run: {experiments}")
    print("=" * 80)

    # Load pulsator data
    print("\n  Loading pulsator data...")
    cv_features, cv_labels, test_features, test_labels = load_pulsator_data()

    all_results = {}
    t_grand = time.time()

    # P0: Baseline
    baseline_agg = None
    if "P0" in experiments or any(e in experiments for e in ["P1", "P2", "P3", "P4"]):
        baseline_result, baseline_agg = run_p0_baseline(cv_features, cv_labels)
        all_results["P0_baseline"] = {
            "aggregated": baseline_agg,
            "fold_results": [
                {k: v for k, v in fr.items() if k != "predictions"}
                for fr in baseline_result["fold_results"]
            ],
        }

    # P1: Leave-One-Out
    pulsator_loo = None
    if "P1" in experiments:
        pulsator_loo = run_p1_leave_one_out(cv_features, cv_labels, baseline_agg)
        all_results["P1_leave_one_out"] = pulsator_loo

    # P2: Forward Selection
    if "P2" in experiments:
        all_results["P2_forward_selection"] = run_p2_forward_selection(
            cv_features, cv_labels, baseline_agg
        )

    # P3: Backward Elimination
    if "P3" in experiments:
        all_results["P3_backward_elimination"] = run_p3_backward_elimination(
            cv_features, cv_labels, baseline_agg
        )

    # P4: Comparison with full 6-class
    if "P4" in experiments:
        if pulsator_loo is None:
            # Try to load from previous run
            prev_pulsator_path = RESULTS_DIR / "pulsator_ablation_results.json"
            if prev_pulsator_path.exists():
                with open(prev_pulsator_path, "r", encoding="utf-8") as f:
                    prev_pulsator = json.load(f)
                pulsator_loo = prev_pulsator.get("P1_leave_one_out", {})
                if not baseline_agg:
                    baseline_agg = prev_pulsator.get("P0_baseline", {}).get("aggregated", {})
            if not pulsator_loo:
                print("  WARNING: P4 requires P1 results. Run with --experiments P1 P4")

        if pulsator_loo and baseline_agg:
            all_results["P4_comparison"] = run_p4_comparison(pulsator_loo, baseline_agg)

    # Save all results
    results_path = RESULTS_DIR / "pulsator_ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(to_native(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n  All results saved to: {results_path}")

    # Per-experiment result files
    for key, data in all_results.items():
        exp_path = RESULTS_DIR / f"{key}.json"
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(to_native(data), f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_grand
    print(f"\n{'=' * 80}")
    print(f"  Pulsator-only ablation complete!")
    print(f"  Total wall time: {total_time/60:.1f} minutes")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
