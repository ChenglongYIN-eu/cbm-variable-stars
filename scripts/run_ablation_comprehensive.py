#!/usr/bin/env python
"""
Comprehensive Ablation Experiments (A6–A12) for CBM Variable Star Classification.

Extends the detailed ablation study (A0–A5b) with:
  A6:  Group-level ablation (remove entire physical concept groups)
  A7:  Forward greedy selection (add concepts one by one)
  A8:  Backward greedy elimination (remove concepts one by one)
  A9:  Pairwise concept synergy matrix
  A10: Per-class sensitivity analysis (no new training)
  A11: Critical subset verification
  A12: Cross-architecture stability (LOO on multiple architectures)

Usage:
  python run_ablation_comprehensive.py                    # Run all
  python run_ablation_comprehensive.py --experiments A6 A10  # Run specific
  python run_ablation_comprehensive.py --experiments A6 A7 A8 A9 A10 A11 A12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_GROUPS,
    CONCEPT_NAMES_12,
    CONCEPTS_CROSS_SURVEY_10,
    LABEL_TO_IDX,
    MINIMAL_CONCEPTS,
    N_CLASSES,
    NUM_CONCEPTS,
    RANDOM_SEED,
)
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.training.cross_val import run_cross_validation

# ======================================================================
# Configuration
# ======================================================================
DATA_PATH = PROJECT_ROOT / "data" / "real" / "gaia_all_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation_comprehensive"
PREV_RESULTS_PATH = PROJECT_ROOT / "results" / "ablation_detailed" / "all_ablation_results.json"
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 256
N_FOLDS = 5

# A7/A8 early stopping thresholds
FORWARD_STOP_RATIO = 0.99      # stop when accuracy reaches 99% of baseline
BACKWARD_STOP_DROP = 0.05      # stop when accuracy drops > 5% from baseline

# A9: only compute synergy for top-N concepts by LOO importance
A9_TOP_N_CONCEPTS = 8


# ======================================================================
# Shared utilities
# ======================================================================

def load_data():
    """Load cached real data, impute, split, and standardize (consistent with run_ablation_detailed.py)."""
    df = pd.read_parquet(DATA_PATH)
    features = df[CONCEPT_NAMES_12].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)

    from pipeline_utils import compute_imputation_stats, apply_imputation
    from cbm_variable_stars.data.splits import create_full_split

    split = create_full_split(labels, test_ratio=0.15)
    cv_idx = split["cv_indices"]
    test_idx = split["test_indices"]

    # Fit imputation on CV data only, apply separately to CV and test
    # [Fix M7/C4] Test data uses global medians only (no per-class label routing)
    imp_stats = compute_imputation_stats(features[cv_idx], labels[cv_idx])
    cv_features = apply_imputation(
        features[cv_idx], labels[cv_idx], imp_stats, use_class_labels=True,
    )
    test_features = apply_imputation(
        features[test_idx], labels[test_idx], imp_stats, use_class_labels=False,
    )
    cv_labels = labels[cv_idx]
    test_labels = labels[test_idx]

    # [Fix] Return raw imputed features (NOT standardized).
    # run_cross_validation() applies per-fold StandardScaler internally.
    return cv_features, cv_labels, test_features, test_labels


def subset_features(features: np.ndarray, concept_names: list[str]) -> np.ndarray:
    """Extract subset of features by concept name."""
    indices = [CONCEPT_NAMES_12.index(c) for c in concept_names]
    return features[:, indices]


def run_cv(features, labels, model_name="hard_cbm", model_kwargs=None, tag=""):
    """Run 5-fold CV and return results."""
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
        for i, cls in enumerate(CLASS_NAMES):
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


def load_previous_results() -> dict:
    """Load results from the detailed ablation (A0–A5b) for reuse."""
    if PREV_RESULTS_PATH.exists():
        with open(PREV_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  WARNING: Previous results not found at {PREV_RESULTS_PATH}")
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
# A6: Group-Level Ablation
# ======================================================================

def run_a6_group_ablation(cv_features, cv_labels, baseline_agg):
    """
    Remove each physical concept group and measure impact.

    Groups (from constants.py CONCEPT_GROUPS):
      timing:      [period, rise_fraction, period_snr]
      fourier:     [R21, R31, phi21]
      amplitude:   [amplitude]
      statistics:  [skewness, kurtosis, stetson_K]
      photometric: [color_bp_rp, mean_mag]
    """
    print("\n" + "=" * 80)
    print("  A6: Group-Level Ablation — Remove entire physical concept groups")
    print("=" * 80)
    t_total = time.time()

    baseline_acc = baseline_agg["accuracy_mean"]
    group_results = {}

    for group_name, group_concepts in CONCEPT_GROUPS.items():
        remaining = [c for c in CONCEPT_NAMES_12 if c not in group_concepts]
        n_remaining = len(remaining)
        removed_str = ", ".join(group_concepts)

        print(f"\n  --- Remove group '{group_name}' ({len(group_concepts)} concepts: {removed_str}) ---")
        print(f"      Remaining: {n_remaining} concepts")
        t0 = time.time()

        features_sub = subset_features(cv_features, remaining)
        result = run_cv(features_sub, cv_labels, "hard_cbm",
                        model_kwargs={"num_concepts": n_remaining},
                        tag=f"A6_no_{group_name}")
        agg = result["aggregated"]

        delta_acc = agg["accuracy_mean"] - baseline_acc
        delta_f1 = agg["macro_f1_mean"] - baseline_agg["macro_f1_mean"]

        print(f"    Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
              f"F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")
        print(f"    Delta vs baseline: Acc={delta_acc:+.4f}, F1={delta_f1:+.4f}")
        print(f"    Time: {time.time()-t0:.1f}s")

        group_results[group_name] = {
            "removed_concepts": group_concepts,
            "remaining_concepts": remaining,
            "n_removed": len(group_concepts),
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

    # Summary sorted by impact
    print(f"\n  Group Ablation Summary (sorted by impact):")
    print(f"  {'Group':<14} {'#Removed':>8} {'Acc':>10} {'Delta Acc':>10} {'F1':>10} {'Delta F1':>10}")
    print(f"  {'=' * 14} {'=' * 8} {'=' * 10} {'=' * 10} {'=' * 10} {'=' * 10}")
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]["delta_accuracy"])
    for group_name, r in sorted_groups:
        print(f"  {group_name:<14} {r['n_removed']:>8} {r['accuracy_mean']:>10.4f} "
              f"{r['delta_accuracy']:>+10.4f} {r['macro_f1_mean']:>10.4f} {r['delta_macro_f1']:>+10.4f}")

    print(f"\n  A6 total time: {time.time() - t_total:.1f}s")
    return group_results


# ======================================================================
# A7: Forward Greedy Selection
# ======================================================================

def run_a7_forward_selection(cv_features, cv_labels, baseline_agg):
    """
    Starting from 0 concepts, greedily add the concept that maximizes accuracy.
    Produces the concept addition order and marginal benefit curve.
    """
    print("\n" + "=" * 80)
    print("  A7: Forward Greedy Selection — Build concept set from scratch")
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
            result = run_cv(features_sub, cv_labels, "hard_cbm",
                            model_kwargs={"num_concepts": len(trial_set)},
                            tag=f"A7_step{step}_{candidate}")
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
            "retained_pct": best_acc / baseline_acc * 100,
            "candidate_scores": candidate_scores,
            "per_class_f1_mean": best_result_agg.get("per_class_f1_mean", []),
        }
        steps.append(step_info)

        print(f"\n      Selected: {best_concept} (Acc={best_acc:.4f}, +{marginal_gain:+.4f})")
        print(f"      Retained: {step_info['retained_pct']:.1f}% of baseline")
        print(f"      Step time: {time.time()-t_step:.1f}s, CV runs so far: {cv_run_count}")

        if best_acc >= stop_threshold and step < 12:
            print(f"\n      Reached {FORWARD_STOP_RATIO*100:.0f}% of baseline accuracy — early stop possible")
            # Continue anyway to get full curve, but mark the point
            step_info["reached_threshold"] = True

    # Summary
    print(f"\n  Forward Selection Order:")
    print(f"  {'Step':<6} {'Added':<18} {'#Concepts':>10} {'Accuracy':>10} {'Marginal':>10} {'Retained%':>10}")
    print(f"  {'=' * 6} {'=' * 18} {'=' * 10} {'=' * 10} {'=' * 10} {'=' * 10}")
    for s in steps:
        print(f"  {s['step']:<6} {s['added_concept']:<18} {s['n_concepts']:>10} "
              f"{s['accuracy']:>10.4f} {s['marginal_gain']:>+10.4f} {s['retained_pct']:>10.1f}")

    print(f"\n  Total CV runs: {cv_run_count}")
    print(f"  A7 total time: {time.time() - t_total:.1f}s")

    return {
        "selection_order": [s["added_concept"] for s in steps],
        "steps": steps,
        "total_cv_runs": cv_run_count,
    }


# ======================================================================
# A8: Backward Greedy Elimination
# ======================================================================

def run_a8_backward_elimination(cv_features, cv_labels, baseline_agg):
    """
    Starting from all 12 concepts, greedily remove the least important concept.
    Produces the concept elimination order and degradation curve.
    """
    print("\n" + "=" * 80)
    print("  A8: Backward Greedy Elimination — Remove concepts one by one")
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
            result = run_cv(features_sub, cv_labels, "hard_cbm",
                            model_kwargs={"num_concepts": len(trial_set)},
                            tag=f"A8_step{step}_no_{candidate}")
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
            "retained_pct": best_acc_after / baseline_acc * 100,
            "candidate_scores": candidate_scores,
            "per_class_f1_mean": best_result_agg.get("per_class_f1_mean", []),
        }
        steps.append(step_info)

        print(f"\n      Removed: {best_to_remove} (Acc={best_acc_after:.4f}, drop={drop_from_prev:+.4f})")
        print(f"      Remaining ({len(remaining)}): {remaining}")
        print(f"      Step time: {time.time()-t_step:.1f}s, CV runs so far: {cv_run_count}")

        if best_acc_after < stop_threshold:
            print(f"\n      Accuracy dropped below threshold ({stop_threshold:.4f}) — stopping")
            break

    # Summary
    print(f"\n  Backward Elimination Order:")
    print(f"  {'Step':<6} {'Removed':<18} {'#Remain':>8} {'Accuracy':>10} {'Drop':>10} {'Retained%':>10}")
    print(f"  {'=' * 6} {'=' * 18} {'=' * 8} {'=' * 10} {'=' * 10} {'=' * 10}")
    for s in steps:
        removed = s["removed_concept"] or "(baseline)"
        print(f"  {s['step']:<6} {removed:<18} {s['n_concepts']:>8} "
              f"{s['accuracy']:>10.4f} {s['drop_from_previous']:>+10.4f} "
              f"{s.get('retained_pct', 100.0):>10.1f}")

    print(f"\n  Total CV runs: {cv_run_count}")
    print(f"  A8 total time: {time.time() - t_total:.1f}s")

    elimination_order = [s["removed_concept"] for s in steps[1:]]
    return {
        "elimination_order": elimination_order,
        "steps": steps,
        "total_cv_runs": cv_run_count,
    }


# ======================================================================
# A9: Pairwise Concept Synergy Matrix
# ======================================================================

def run_a9_pairwise_synergy(cv_features, cv_labels, baseline_agg, loo_results):
    """
    Compute synergy/redundancy between concept pairs.

    synergy(i,j) = [drop(i) + drop(j)] - drop(i,j)
      > 0 → synergistic (joint removal hurts more than sum of individual)
      < 0 → redundant (overlapping information)
      ≈ 0 → independent
    """
    print("\n" + "=" * 80)
    print("  A9: Pairwise Concept Synergy Matrix")
    print("=" * 80)
    t_total = time.time()

    baseline_acc = baseline_agg["accuracy_mean"]

    # Get single-concept drops from LOO results
    single_drops = {}
    for concept, data in loo_results.items():
        single_drops[concept] = baseline_acc - data["accuracy_mean"]

    # Select top N concepts by LOO importance for pairwise analysis
    sorted_by_importance = sorted(single_drops.items(), key=lambda x: -x[1])
    top_concepts = [c for c, _ in sorted_by_importance[:A9_TOP_N_CONCEPTS]]

    print(f"  Analyzing top {A9_TOP_N_CONCEPTS} concepts by LOO importance:")
    for c, d in sorted_by_importance[:A9_TOP_N_CONCEPTS]:
        print(f"    {c:<18} drop={d:+.4f}")

    pairs = list(combinations(top_concepts, 2))
    print(f"\n  Computing {len(pairs)} pairwise removal experiments...")

    pair_results = {}
    cv_run_count = 0

    for idx, (c1, c2) in enumerate(pairs):
        remaining = [c for c in CONCEPT_NAMES_12 if c not in (c1, c2)]
        n_remaining = len(remaining)

        print(f"\n  [{idx+1}/{len(pairs)}] Remove ({c1}, {c2}) — {n_remaining} remaining")
        t0 = time.time()

        features_sub = subset_features(cv_features, remaining)
        result = run_cv(features_sub, cv_labels, "hard_cbm",
                        model_kwargs={"num_concepts": n_remaining},
                        tag=f"A9_no_{c1}_{c2}")
        agg = result["aggregated"]
        cv_run_count += 1

        pair_drop = baseline_acc - agg["accuracy_mean"]
        synergy = (single_drops[c1] + single_drops[c2]) - pair_drop

        pair_key = f"{c1}+{c2}"
        pair_results[pair_key] = {
            "concept_1": c1,
            "concept_2": c2,
            "remaining_concepts": remaining,
            "accuracy_mean": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"],
            "macro_f1_mean": agg["macro_f1_mean"],
            "pair_drop": pair_drop,
            "individual_drop_sum": single_drops[c1] + single_drops[c2],
            "synergy": synergy,
            "interpretation": "synergistic" if synergy > 0.005 else ("redundant" if synergy < -0.005 else "independent"),
            "per_class_f1_mean": agg.get("per_class_f1_mean", []),
        }

        print(f"    Acc={agg['accuracy_mean']:.4f}, pair_drop={pair_drop:+.4f}")
        print(f"    Individual sum={single_drops[c1]+single_drops[c2]:.4f}, synergy={synergy:+.4f} "
              f"({pair_results[pair_key]['interpretation']})")
        print(f"    Time: {time.time()-t0:.1f}s")

    # Build synergy matrix (full 12x12 for visualization, NaN for non-computed pairs)
    n = len(CONCEPT_NAMES_12)
    synergy_matrix = np.full((n, n), np.nan)
    np.fill_diagonal(synergy_matrix, 0.0)

    for pair_key, data in pair_results.items():
        c1, c2 = data["concept_1"], data["concept_2"]
        i = CONCEPT_NAMES_12.index(c1)
        j = CONCEPT_NAMES_12.index(c2)
        synergy_matrix[i, j] = data["synergy"]
        synergy_matrix[j, i] = data["synergy"]

    # Summary
    print(f"\n  Synergy Matrix (top pairs by |synergy|):")
    print(f"  {'Pair':<36} {'Synergy':>10} {'Type':>14}")
    print(f"  {'=' * 36} {'=' * 10} {'=' * 14}")
    sorted_pairs = sorted(pair_results.items(), key=lambda x: abs(x[1]["synergy"]), reverse=True)
    for pair_key, data in sorted_pairs[:15]:
        print(f"  {pair_key:<36} {data['synergy']:>+10.4f} {data['interpretation']:>14}")

    print(f"\n  Total CV runs: {cv_run_count}")
    print(f"  A9 total time: {time.time() - t_total:.1f}s")

    return {
        "top_concepts_analyzed": top_concepts,
        "pair_results": pair_results,
        "synergy_matrix": synergy_matrix.tolist(),
        "concept_order": CONCEPT_NAMES_12,
        "total_cv_runs": cv_run_count,
    }


# ======================================================================
# A10: Per-Class Sensitivity Analysis (no new training)
# ======================================================================

def run_a10_per_class_sensitivity(baseline_agg, loo_results):
    """
    Analyze per-concept, per-class F1 impact using existing LOO data.
    Produces a 12×6 sensitivity matrix (concepts × classes).
    """
    print("\n" + "=" * 80)
    print("  A10: Per-Class Sensitivity Analysis (from existing LOO data)")
    print("=" * 80)

    baseline_per_class = baseline_agg.get("per_class_f1_mean", [])
    if not baseline_per_class:
        print("  ERROR: baseline per_class_f1_mean not available")
        return {}

    # Build sensitivity matrix: delta F1 when removing each concept
    n_concepts = len(CONCEPT_NAMES_12)
    n_classes = len(CLASS_NAMES)
    sensitivity_matrix = np.zeros((n_concepts, n_classes))

    for i, concept in enumerate(CONCEPT_NAMES_12):
        loo_data = loo_results.get(concept, {})
        per_class = loo_data.get("per_class_f1_mean", [])
        if len(per_class) == n_classes:
            for j in range(n_classes):
                sensitivity_matrix[i, j] = per_class[j] - baseline_per_class[j]

    # Print matrix
    print(f"\n  Sensitivity Matrix (delta F1 when concept removed):")
    header = f"  {'Concept':<18}" + "".join(f"{cls:>14}" for cls in CLASS_NAMES)
    print(header)
    print(f"  {'=' * 18}" + "=" * 14 * n_classes)

    for i, concept in enumerate(CONCEPT_NAMES_12):
        row_str = f"  {concept:<18}"
        for j in range(n_classes):
            val = sensitivity_matrix[i, j]
            row_str += f"{val:>+14.4f}"
        print(row_str)

    # Identify class-specific concepts (large impact on specific class)
    print(f"\n  Class-Specific Concepts (|delta F1| > 0.01 for a specific class):")
    for j, cls in enumerate(CLASS_NAMES):
        col = sensitivity_matrix[:, j]
        important = [(CONCEPT_NAMES_12[i], col[i]) for i in range(n_concepts) if abs(col[i]) > 0.01]
        important.sort(key=lambda x: x[1])
        if important:
            concepts_str = ", ".join(f"{c}({d:+.4f})" for c, d in important)
            print(f"    {cls:<14}: {concepts_str}")

    # Concept specificity: how concentrated is the impact across classes?
    print(f"\n  Concept Impact Distribution (std across classes):")
    print(f"  {'Concept':<18} {'Mean Delta':>12} {'Std Delta':>12} {'Max |Delta|':>12} {'Most Affected':>14}")
    print(f"  {'=' * 18} {'=' * 12} {'=' * 12} {'=' * 12} {'=' * 14}")
    for i, concept in enumerate(CONCEPT_NAMES_12):
        row = sensitivity_matrix[i, :]
        mean_d = np.mean(row)
        std_d = np.std(row)
        max_abs_idx = np.argmax(np.abs(row))
        max_abs_val = row[max_abs_idx]
        print(f"  {concept:<18} {mean_d:>+12.4f} {std_d:>12.4f} {max_abs_val:>+12.4f} {CLASS_NAMES[max_abs_idx]:>14}")

    return {
        "sensitivity_matrix": sensitivity_matrix.tolist(),
        "concepts": CONCEPT_NAMES_12,
        "classes": CLASS_NAMES,
        "baseline_per_class_f1": baseline_per_class,
    }


# ======================================================================
# A11: Critical Subset Verification
# ======================================================================

def run_a11_critical_subsets(cv_features, cv_labels, baseline_agg,
                             a7_results=None, a8_results=None):
    """
    Verify specific concept subsets identified by A7/A8 and compare with
    astronomically-motivated subsets.
    """
    print("\n" + "=" * 80)
    print("  A11: Critical Subset Verification")
    print("=" * 80)
    t_total = time.time()

    baseline_acc = baseline_agg["accuracy_mean"]

    # Build list of subsets to test
    subsets_to_test = {}

    # Predefined subsets
    subsets_to_test["MINIMAL_4"] = MINIMAL_CONCEPTS  # Already tested in A2, but re-verify
    subsets_to_test["CROSS_SURVEY_10"] = CONCEPTS_CROSS_SURVEY_10

    # LOO top 3
    subsets_to_test["LOO_top3"] = ["R31", "period_snr", "color_bp_rp"]

    # LOO top 6
    subsets_to_test["LOO_top6"] = ["R31", "period_snr", "color_bp_rp", "mean_mag", "period", "amplitude"]

    # A7 forward selection subsets (if available)
    if a7_results and "selection_order" in a7_results:
        order = a7_results["selection_order"]
        for k in [4, 6, 8]:
            if k <= len(order):
                subsets_to_test[f"A7_top{k}"] = order[:k]

    # A8 backward elimination subsets (if available)
    if a8_results and "elimination_order" in a8_results:
        elim_order = a8_results["elimination_order"]
        for k in [4, 6, 8]:
            # After eliminating (12-k) concepts, k remain
            n_to_remove = 12 - k
            if n_to_remove <= len(elim_order):
                removed = set(elim_order[:n_to_remove])
                remaining = [c for c in CONCEPT_NAMES_12 if c not in removed]
                subsets_to_test[f"A8_top{k}"] = remaining

    # Run each subset
    subset_results = {}
    cv_run_count = 0

    for name, concepts in subsets_to_test.items():
        n = len(concepts)
        print(f"\n  --- {name} ({n} concepts): {concepts} ---")
        t0 = time.time()

        features_sub = subset_features(cv_features, concepts)
        result = run_cv(features_sub, cv_labels, "hard_cbm",
                        model_kwargs={"num_concepts": n},
                        tag=f"A11_{name}")
        agg = result["aggregated"]
        cv_run_count += 1

        delta_acc = agg["accuracy_mean"] - baseline_acc
        retained = agg["accuracy_mean"] / baseline_acc * 100

        print(f"    Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
              f"F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")
        print(f"    Delta={delta_acc:+.4f}, Retained={retained:.1f}%")
        print(f"    Time: {time.time()-t0:.1f}s")

        subset_results[name] = {
            "concepts": concepts,
            "n_concepts": n,
            "accuracy_mean": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"],
            "macro_f1_mean": agg["macro_f1_mean"],
            "macro_f1_std": agg["macro_f1_std"],
            "delta_accuracy": delta_acc,
            "retained_pct": retained,
            "per_class_f1_mean": agg.get("per_class_f1_mean", []),
        }

    # Summary
    print(f"\n  Critical Subset Summary:")
    print(f"  {'Subset':<20} {'#Concepts':>10} {'Accuracy':>10} {'Delta':>10} {'Retained%':>10}")
    print(f"  {'=' * 20} {'=' * 10} {'=' * 10} {'=' * 10} {'=' * 10}")
    sorted_subsets = sorted(subset_results.items(), key=lambda x: -x[1]["accuracy_mean"])
    for name, r in sorted_subsets:
        print(f"  {name:<20} {r['n_concepts']:>10} {r['accuracy_mean']:>10.4f} "
              f"{r['delta_accuracy']:>+10.4f} {r['retained_pct']:>10.1f}")

    print(f"\n  Total CV runs: {cv_run_count}")
    print(f"  A11 total time: {time.time() - t_total:.1f}s")

    return subset_results


# ======================================================================
# A12: Cross-Architecture Stability
# ======================================================================

def run_a12_cross_architecture(cv_features, cv_labels, baseline_agg, loo_results):
    """
    Repeat LOO experiment on HardCBM_Linear and SoftCBM to check if
    concept importance ranking is architecture-independent.
    """
    print("\n" + "=" * 80)
    print("  A12: Cross-Architecture Stability — LOO on multiple architectures")
    print("=" * 80)
    t_total = time.time()

    architectures = {
        "hard_cbm": loo_results,  # reuse existing
        "hard_cbm_linear": None,
        "soft_cbm": None,
    }

    baseline_results_per_arch = {
        "hard_cbm": baseline_agg,
    }

    # Run baselines for other architectures
    for arch in ["hard_cbm_linear", "soft_cbm"]:
        print(f"\n  --- {arch} baseline (12 concepts) ---")
        t0 = time.time()
        result = run_cv(cv_features, cv_labels, arch, tag=f"A12_{arch}_baseline")
        agg = print_detailed_result(result, f"A12 baseline: {arch}")
        baseline_results_per_arch[arch] = agg
        print(f"    Time: {time.time()-t0:.1f}s")

    # Run LOO for non-hard_cbm architectures
    cv_run_count = 0
    for arch in ["hard_cbm_linear", "soft_cbm"]:
        print(f"\n  ===== LOO for {arch} =====")
        arch_loo = {}
        arch_baseline_acc = baseline_results_per_arch[arch]["accuracy_mean"]

        for concept_to_remove in CONCEPT_NAMES_12:
            remaining = [c for c in CONCEPT_NAMES_12 if c != concept_to_remove]
            n_remaining = len(remaining)

            print(f"\n  [{arch}] Removing: {concept_to_remove}")
            t0 = time.time()

            features_sub = subset_features(cv_features, remaining)
            result = run_cv(features_sub, cv_labels, arch,
                            model_kwargs={"num_concepts": n_remaining},
                            tag=f"A12_{arch}_no_{concept_to_remove}")
            agg = result["aggregated"]
            cv_run_count += 1

            delta_acc = agg["accuracy_mean"] - arch_baseline_acc
            print(f"    Acc={agg['accuracy_mean']:.4f}, Delta={delta_acc:+.4f}, Time={time.time()-t0:.1f}s")

            arch_loo[concept_to_remove] = {
                "accuracy_mean": agg["accuracy_mean"],
                "accuracy_std": agg["accuracy_std"],
                "macro_f1_mean": agg["macro_f1_mean"],
                "macro_f1_std": agg["macro_f1_std"],
                "delta_accuracy": delta_acc,
                "delta_macro_f1": agg["macro_f1_mean"] - baseline_results_per_arch[arch]["macro_f1_mean"],
                "per_class_f1_mean": agg.get("per_class_f1_mean", []),
            }

        architectures[arch] = arch_loo

    # Compute rankings and Kendall tau correlation
    rankings = {}
    for arch, loo_data in architectures.items():
        if loo_data:
            sorted_concepts = sorted(loo_data.items(), key=lambda x: x[1]["delta_accuracy"])
            rankings[arch] = [c for c, _ in sorted_concepts]

    print(f"\n  Concept Importance Rankings by Architecture:")
    print(f"  {'Rank':<6}" + "".join(f"{arch:<20}" for arch in rankings.keys()))
    print(f"  {'=' * 6}" + "=" * 20 * len(rankings))
    for rank in range(12):
        row = f"  {rank+1:<6}"
        for arch in rankings:
            concept = rankings[arch][rank]
            delta = architectures[arch][concept]["delta_accuracy"]
            row += f"{concept} ({delta:+.3f})   "
        print(row)

    # Compute Kendall tau between each pair
    from scipy.stats import kendalltau
    arch_names = list(rankings.keys())
    tau_results = {}

    print(f"\n  Kendall Tau Correlations:")
    for i, a1 in enumerate(arch_names):
        for j, a2 in enumerate(arch_names):
            if i < j:
                # Convert rankings to numeric ranks
                rank1 = [rankings[a1].index(c) for c in CONCEPT_NAMES_12]
                rank2 = [rankings[a2].index(c) for c in CONCEPT_NAMES_12]
                tau, p_value = kendalltau(rank1, rank2)
                pair_key = f"{a1}_vs_{a2}"
                tau_results[pair_key] = {"tau": float(tau), "p_value": float(p_value)}
                print(f"    {pair_key}: tau={tau:.4f}, p={p_value:.4f}")

    print(f"\n  Total CV runs: {cv_run_count}")
    print(f"  A12 total time: {time.time() - t_total:.1f}s")

    return {
        "baseline_per_arch": {
            arch: {
                "accuracy_mean": agg["accuracy_mean"],
                "accuracy_std": agg["accuracy_std"],
                "macro_f1_mean": agg["macro_f1_mean"],
                "macro_f1_std": agg["macro_f1_std"],
            }
            for arch, agg in baseline_results_per_arch.items()
        },
        "loo_per_arch": {
            arch: data for arch, data in architectures.items()
        },
        "rankings": rankings,
        "kendall_tau": tau_results,
        "total_cv_runs": cv_run_count,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive CBM Ablation Experiments (A6–A12)")
    parser.add_argument(
        "--experiments", nargs="*",
        default=["A6", "A7", "A8", "A9", "A10", "A11", "A12"],
        help="Which experiments to run (e.g., --experiments A6 A10)"
    )
    args = parser.parse_args()
    experiments = [e.upper() for e in args.experiments]

    set_global_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  CBM Variable Star Classification — Comprehensive Ablation Experiments")
    print(f"  Experiments to run: {experiments}")
    print("=" * 80)

    # Load previous results (A0–A5b)
    prev = load_previous_results()
    baseline_agg = prev.get("A0_baseline", {}).get("aggregated", {})
    loo_results = prev.get("A2b_leave_one_out", {})

    if not baseline_agg:
        print("  WARNING: No baseline results found. Running A0 baseline first...")
        cv_features, cv_labels, _, _ = load_data()
        result = run_cv(cv_features, cv_labels, "hard_cbm", tag="A0_baseline")
        baseline_agg = result["aggregated"]
    else:
        print(f"  Loaded baseline: Acc={baseline_agg['accuracy_mean']:.4f}, F1={baseline_agg['macro_f1_mean']:.4f}")

    # Load data (needed for all experiments except A10)
    needs_data = any(e in experiments for e in ["A6", "A7", "A8", "A9", "A11", "A12"])
    if needs_data:
        print("\n  Loading data...")
        cv_features, cv_labels, _, _ = load_data()
    else:
        cv_features = cv_labels = None

    all_results = {}
    t_grand = time.time()

    # A6: Group-Level Ablation
    if "A6" in experiments:
        all_results["A6_group_ablation"] = run_a6_group_ablation(cv_features, cv_labels, baseline_agg)

    # A7: Forward Greedy Selection
    if "A7" in experiments:
        all_results["A7_forward_selection"] = run_a7_forward_selection(cv_features, cv_labels, baseline_agg)

    # A8: Backward Greedy Elimination
    if "A8" in experiments:
        all_results["A8_backward_elimination"] = run_a8_backward_elimination(cv_features, cv_labels, baseline_agg)

    # A9: Pairwise Synergy
    if "A9" in experiments:
        if not loo_results:
            print("  WARNING: A9 requires LOO results (A2b). Skipping.")
        else:
            all_results["A9_pairwise_synergy"] = run_a9_pairwise_synergy(
                cv_features, cv_labels, baseline_agg, loo_results)

    # A10: Per-Class Sensitivity (no new training)
    if "A10" in experiments:
        if not loo_results:
            print("  WARNING: A10 requires LOO results (A2b). Skipping.")
        else:
            all_results["A10_per_class_sensitivity"] = run_a10_per_class_sensitivity(
                baseline_agg, loo_results)

    # A11: Critical Subsets
    if "A11" in experiments:
        all_results["A11_critical_subsets"] = run_a11_critical_subsets(
            cv_features, cv_labels, baseline_agg,
            a7_results=all_results.get("A7_forward_selection"),
            a8_results=all_results.get("A8_backward_elimination"),
        )

    # A12: Cross-Architecture Stability
    if "A12" in experiments:
        if not loo_results:
            print("  WARNING: A12 requires LOO results (A2b). Skipping.")
        else:
            all_results["A12_cross_architecture"] = run_a12_cross_architecture(
                cv_features, cv_labels, baseline_agg, loo_results)

    # Save all results
    results_path = RESULTS_DIR / "comprehensive_ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(to_native(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n  All results saved to: {results_path}")

    # Per-experiment result files for convenience
    for key, data in all_results.items():
        exp_path = RESULTS_DIR / f"{key}.json"
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(to_native(data), f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_grand
    print(f"\n{'=' * 80}")
    print(f"  Comprehensive ablation complete!")
    print(f"  Total wall time: {total_time/60:.1f} minutes")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
