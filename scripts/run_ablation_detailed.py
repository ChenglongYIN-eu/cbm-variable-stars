#!/usr/bin/env python
"""
Detailed Ablation Experiments — Run one by one with full per-class results.

Experiments:
  A1:   Remove C11 (color_bp_rp)
  A2:   Minimal 4-concept set {period, amplitude, R21, phi21}
  A2b:  Leave-One-Concept-Out (remove each of 12 concepts individually)
  A4:   Architecture comparison (HardCBM / HardCBM_Cal / SoftCBM / CEM)
  A5a:  Label predictor complexity (Linear vs MLP)
"""
from __future__ import annotations

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
    CLASS_NAMES, CONCEPT_NAMES_12, LABEL_TO_IDX, NUM_CONCEPTS,
    RANDOM_SEED, N_CLASSES, MINIMAL_CONCEPTS, CONCEPTS_NO_COLOR,
)
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.training.cross_val import run_cross_validation

# ======================================================================
DATA_PATH = PROJECT_ROOT / "data" / "real" / "gaia_all_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation_detailed"
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 256
N_FOLDS = 5


def load_data():
    """Load cached real data, split, and standardize."""
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


def run_cv(features, labels, model_name, model_kwargs=None, tag=""):
    """Run 5-fold CV and return results with detailed printing."""
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

    # Per-class F1
    if "per_class_f1_mean" in agg:
        print(f"\n    Per-class F1 (mean +/- std):")
        for i, cls in enumerate(CLASS_NAMES):
            mean_f1 = agg["per_class_f1_mean"][i]
            std_f1 = agg["per_class_f1_std"][i]
            print(f"      {cls:<14} {mean_f1:.4f} +/- {std_f1:.4f}")

    # Per-fold summary
    print(f"\n    Per-fold breakdown:")
    print(f"    {'Fold':<6} {'Acc':>8} {'F1':>8} {'BestEp':>8} {'Time(s)':>8}")
    print(f"    {'----':<6} {'----':>8} {'----':>8} {'------':>8} {'-------':>8}")
    for fr in folds:
        m = fr["metrics"]
        print(f"    {fr['fold']+1:<6} {m['val_accuracy']:>8.4f} {m['val_macro_f1']:>8.4f} "
              f"{fr['best_epoch']:>8} {fr['training_time']:>8.1f}")

    return agg


def subset_features(features, concept_names):
    """Extract subset of features by concept name."""
    indices = [CONCEPT_NAMES_12.index(c) for c in concept_names]
    return features[:, indices]


def main():
    set_global_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cv_features, cv_labels, test_features, test_labels = load_data()
    all_ablation_results = {}

    # ==================================================================
    # A0: BASELINE — Full 12-concept HardCBM
    # ==================================================================
    print("\n" + "=" * 80)
    print("  A0: BASELINE — Full 12-concept HardCBM")
    print("=" * 80)
    t0 = time.time()

    result_baseline = run_cv(cv_features, cv_labels, "hard_cbm", tag="A0_baseline")
    agg_baseline = print_detailed_result(result_baseline, "Baseline 12-concept HardCBM")
    all_ablation_results["A0_baseline"] = {
        "concepts": CONCEPT_NAMES_12,
        "n_concepts": 12,
        "aggregated": agg_baseline,
    }
    print(f"\n  A0 completed in {time.time() - t0:.1f}s")

    # ==================================================================
    # A1: Remove C11 (color_bp_rp)
    # ==================================================================
    print("\n" + "=" * 80)
    print("  A1: Remove C11 (color_bp_rp) — 11 concepts")
    print("=" * 80)
    t0 = time.time()

    features_no_color = subset_features(cv_features, CONCEPTS_NO_COLOR)
    result_a1 = run_cv(features_no_color, cv_labels, "hard_cbm",
                       model_kwargs={"num_concepts": 11}, tag="A1_no_color")
    agg_a1 = print_detailed_result(result_a1, "A1: No color_bp_rp (11-dim)")

    delta_acc = agg_a1["accuracy_mean"] - agg_baseline["accuracy_mean"]
    delta_f1 = agg_a1["macro_f1_mean"] - agg_baseline["macro_f1_mean"]
    print(f"\n  Delta vs baseline: Acc={delta_acc:+.4f}, F1={delta_f1:+.4f}")
    all_ablation_results["A1_remove_color"] = {
        "concepts": CONCEPTS_NO_COLOR,
        "n_concepts": 11,
        "aggregated": agg_a1,
        "delta_accuracy": delta_acc,
        "delta_macro_f1": delta_f1,
    }
    print(f"  A1 completed in {time.time() - t0:.1f}s")

    # ==================================================================
    # A2: Minimal 4-concept set
    # ==================================================================
    print("\n" + "=" * 80)
    print("  A2: Minimal 4-concept set {period, amplitude, R21, phi21}")
    print("=" * 80)
    t0 = time.time()

    features_minimal = subset_features(cv_features, MINIMAL_CONCEPTS)
    result_a2 = run_cv(features_minimal, cv_labels, "hard_cbm",
                       model_kwargs={"num_concepts": 4}, tag="A2_minimal")
    agg_a2 = print_detailed_result(result_a2, "A2: Minimal 4-concept")

    delta_acc = agg_a2["accuracy_mean"] - agg_baseline["accuracy_mean"]
    delta_f1 = agg_a2["macro_f1_mean"] - agg_baseline["macro_f1_mean"]
    retained_pct = agg_a2["accuracy_mean"] / agg_baseline["accuracy_mean"] * 100
    print(f"\n  Delta vs baseline: Acc={delta_acc:+.4f}, F1={delta_f1:+.4f}")
    print(f"  Retained accuracy: {retained_pct:.1f}%")
    all_ablation_results["A2_minimal"] = {
        "concepts": MINIMAL_CONCEPTS,
        "n_concepts": 4,
        "aggregated": agg_a2,
        "delta_accuracy": delta_acc,
        "delta_macro_f1": delta_f1,
        "retained_pct": retained_pct,
    }
    print(f"  A2 completed in {time.time() - t0:.1f}s")

    # ==================================================================
    # A2b: Leave-One-Concept-Out (12 individual ablations)
    # ==================================================================
    print("\n" + "=" * 80)
    print("  A2b: Leave-One-Concept-Out — Remove each concept individually")
    print("=" * 80)
    t_total = time.time()

    loco_results = {}
    for concept_to_remove in CONCEPT_NAMES_12:
        remaining = [c for c in CONCEPT_NAMES_12 if c != concept_to_remove]
        n_remaining = len(remaining)

        print(f"\n  --- Removing: {concept_to_remove} ({n_remaining} remaining) ---")
        t0 = time.time()

        features_sub = subset_features(cv_features, remaining)
        result = run_cv(features_sub, cv_labels, "hard_cbm",
                        model_kwargs={"num_concepts": n_remaining},
                        tag=f"A2b_no_{concept_to_remove}")
        agg = result["aggregated"]

        delta_acc = agg["accuracy_mean"] - agg_baseline["accuracy_mean"]
        delta_f1 = agg["macro_f1_mean"] - agg_baseline["macro_f1_mean"]

        print(f"    Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
              f"F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")
        print(f"    Delta vs baseline: Acc={delta_acc:+.4f}, F1={delta_f1:+.4f}")
        print(f"    Time: {time.time()-t0:.1f}s")

        loco_results[concept_to_remove] = {
            "accuracy_mean": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"],
            "macro_f1_mean": agg["macro_f1_mean"],
            "macro_f1_std": agg["macro_f1_std"],
            "delta_accuracy": delta_acc,
            "delta_macro_f1": delta_f1,
            "per_class_f1_mean": agg.get("per_class_f1_mean", []),
        }

    # Sort by impact (most negative delta = most important concept)
    print(f"\n  Leave-One-Concept-Out Summary (sorted by impact):")
    print(f"  {'Removed Concept':<18} {'Acc':>10} {'Delta Acc':>10} {'F1':>10} {'Delta F1':>10}")
    print(f"  {'=' * 18} {'=' * 10} {'=' * 10} {'=' * 10} {'=' * 10}")
    sorted_concepts = sorted(loco_results.items(), key=lambda x: x[1]["delta_accuracy"])
    for concept, r in sorted_concepts:
        print(f"  {concept:<18} {r['accuracy_mean']:>10.4f} {r['delta_accuracy']:>+10.4f} "
              f"{r['macro_f1_mean']:>10.4f} {r['delta_macro_f1']:>+10.4f}")

    all_ablation_results["A2b_leave_one_out"] = loco_results
    print(f"\n  A2b total time: {time.time() - t_total:.1f}s")

    # ==================================================================
    # A4: Architecture Comparison
    # ==================================================================
    print("\n" + "=" * 80)
    print("  A4: Architecture Comparison — HardCBM / HardCBM_Cal / SoftCBM / CEM")
    print("=" * 80)
    t_total = time.time()

    arch_results = {}
    for model_name in ["hard_cbm", "hard_cbm_cal", "soft_cbm", "cem"]:
        print(f"\n  --- {model_name} ---")
        t0 = time.time()
        result = run_cv(cv_features, cv_labels, model_name, tag=f"A4_{model_name}")
        agg = print_detailed_result(result, f"A4: {model_name}")
        arch_results[model_name] = agg
        print(f"    Time: {time.time()-t0:.1f}s")

    print(f"\n  Architecture Comparison Summary:")
    print(f"  {'Model':<20} {'Accuracy':>14} {'Macro F1':>14} {'Params':>10}")
    print(f"  {'=' * 20} {'=' * 14} {'=' * 14} {'=' * 10}")

    # Get param counts
    from cbm_variable_stars.models import create_model
    for model_name, agg in arch_results.items():
        m = create_model(model_name)
        n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        acc_str = f"{agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}"
        f1_str = f"{agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}"
        print(f"  {model_name:<20} {acc_str:>14} {f1_str:>14} {n_params:>10,}")

    all_ablation_results["A4_architecture"] = {
        name: {
            "accuracy_mean": a["accuracy_mean"],
            "accuracy_std": a["accuracy_std"],
            "macro_f1_mean": a["macro_f1_mean"],
            "macro_f1_std": a["macro_f1_std"],
            "per_class_f1_mean": a.get("per_class_f1_mean", []),
            "per_class_f1_std": a.get("per_class_f1_std", []),
        }
        for name, a in arch_results.items()
    }
    print(f"\n  A4 total time: {time.time() - t_total:.1f}s")

    # ==================================================================
    # A5a: Label Predictor Complexity (Linear vs MLP)
    # ==================================================================
    print("\n" + "=" * 80)
    print("  A5a: Label Predictor Complexity — HardCBM_Linear vs HardCBM (MLP)")
    print("=" * 80)
    t_total = time.time()

    a5a_results = {}
    for model_name in ["hard_cbm_linear", "hard_cbm"]:
        print(f"\n  --- {model_name} ---")
        t0 = time.time()
        result = run_cv(cv_features, cv_labels, model_name, tag=f"A5a_{model_name}")
        agg = print_detailed_result(result, f"A5a: {model_name}")
        a5a_results[model_name] = agg
        print(f"    Time: {time.time()-t0:.1f}s")

    delta_acc = a5a_results["hard_cbm"]["accuracy_mean"] - a5a_results["hard_cbm_linear"]["accuracy_mean"]
    delta_f1 = a5a_results["hard_cbm"]["macro_f1_mean"] - a5a_results["hard_cbm_linear"]["macro_f1_mean"]

    print(f"\n  MLP vs Linear:")
    print(f"    Accuracy gain:  {delta_acc:+.4f}")
    print(f"    Macro F1 gain:  {delta_f1:+.4f}")

    # Per-class comparison
    if "per_class_f1_mean" in a5a_results["hard_cbm"] and "per_class_f1_mean" in a5a_results["hard_cbm_linear"]:
        print(f"\n    Per-class F1 comparison:")
        print(f"    {'Class':<14} {'Linear':>8} {'MLP':>8} {'Delta':>8}")
        print(f"    {'=' * 14} {'=' * 8} {'=' * 8} {'=' * 8}")
        for i, cls in enumerate(CLASS_NAMES):
            lin_f1 = a5a_results["hard_cbm_linear"]["per_class_f1_mean"][i]
            mlp_f1 = a5a_results["hard_cbm"]["per_class_f1_mean"][i]
            d = mlp_f1 - lin_f1
            print(f"    {cls:<14} {lin_f1:>8.4f} {mlp_f1:>8.4f} {d:>+8.4f}")

    all_ablation_results["A5a_predictor_complexity"] = {
        name: {
            "accuracy_mean": a["accuracy_mean"],
            "accuracy_std": a["accuracy_std"],
            "macro_f1_mean": a["macro_f1_mean"],
            "macro_f1_std": a["macro_f1_std"],
            "per_class_f1_mean": a.get("per_class_f1_mean", []),
        }
        for name, a in a5a_results.items()
    }
    all_ablation_results["A5a_predictor_complexity"]["delta"] = {
        "accuracy": delta_acc,
        "macro_f1": delta_f1,
    }
    print(f"\n  A5a total time: {time.time() - t_total:.1f}s")

    # ==================================================================
    # A5b: Baselines comparison (MLP / RF / XGBoost vs HardCBM)
    # ==================================================================
    print("\n" + "=" * 80)
    print("  A5b: CBM vs Black-Box Baselines (MLP / RF / XGBoost)")
    print("=" * 80)
    t_total = time.time()

    # MLP baseline
    print(f"\n  --- mlp ---")
    t0 = time.time()
    result_mlp = run_cv(cv_features, cv_labels, "mlp", tag="A5b_mlp")
    agg_mlp = print_detailed_result(result_mlp, "A5b: MLP baseline")
    print(f"    Time: {time.time()-t0:.1f}s")

    # RF and XGBoost baselines
    from cbm_variable_stars.data.splits import create_cv_splits
    from cbm_variable_stars.training.trainer import train_baseline
    from sklearn.preprocessing import StandardScaler

    baseline_results = {}
    for model_type in ["rf", "xgb"]:
        print(f"\n  --- {model_type} ---")
        t0 = time.time()
        splits = create_cv_splits(cv_labels, n_folds=N_FOLDS)
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Per-fold standardization for baselines (consistent with CBM CV)
            bl_scaler = StandardScaler()
            train_feat_scaled = bl_scaler.fit_transform(cv_features[train_idx])
            val_feat_scaled = bl_scaler.transform(cv_features[val_idx])
            model, metrics = train_baseline(
                model_type=model_type,
                features_train=train_feat_scaled,
                labels_train=cv_labels[train_idx],
                features_val=val_feat_scaled,
                labels_val=cv_labels[val_idx],
            )
            fold_metrics.append(metrics)
            print(f"    Fold {fold_idx+1}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

        mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
        std_acc = np.std([m["accuracy"] for m in fold_metrics], ddof=1)
        mean_f1 = np.mean([m["macro_f1"] for m in fold_metrics])
        std_f1 = np.std([m["macro_f1"] for m in fold_metrics], ddof=1)
        print(f"    Mean: Acc={mean_acc:.4f}+/-{std_acc:.4f}, F1={mean_f1:.4f}+/-{std_f1:.4f}")

        baseline_results[model_type] = {
            "accuracy_mean": float(mean_acc),
            "accuracy_std": float(std_acc),
            "macro_f1_mean": float(mean_f1),
            "macro_f1_std": float(std_f1),
        }
        print(f"    Time: {time.time()-t0:.1f}s")

    # Summary table
    print(f"\n  CBM vs Baselines Summary:")
    print(f"  {'Model':<20} {'Accuracy':>14} {'Macro F1':>14} {'Interpretable':>14}")
    print(f"  {'=' * 20} {'=' * 14} {'=' * 14} {'=' * 14}")

    for name, agg in [
        ("hard_cbm", agg_baseline),
        ("soft_cbm", arch_results.get("soft_cbm", {})),
        ("mlp", agg_mlp),
    ]:
        if agg:
            acc_str = f"{agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}"
            f1_str = f"{agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}"
            interp = "Yes" if name.startswith(("hard", "soft")) else "No"
            print(f"  {name:<20} {acc_str:>14} {f1_str:>14} {interp:>14}")

    for name, r in baseline_results.items():
        acc_str = f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}"
        f1_str = f"{r['macro_f1_mean']:.4f}+/-{r['macro_f1_std']:.4f}"
        print(f"  {name:<20} {acc_str:>14} {f1_str:>14} {'No':>14}")

    all_ablation_results["A5b_baselines"] = baseline_results
    all_ablation_results["A5b_baselines"]["mlp"] = {
        "accuracy_mean": agg_mlp["accuracy_mean"],
        "accuracy_std": agg_mlp["accuracy_std"],
        "macro_f1_mean": agg_mlp["macro_f1_mean"],
        "macro_f1_std": agg_mlp["macro_f1_std"],
    }
    print(f"\n  A5b total time: {time.time() - t_total:.1f}s")

    # ==================================================================
    # Save all results
    # ==================================================================
    def to_native(obj):
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

    results_path = RESULTS_DIR / "all_ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(to_native(all_ablation_results), f, indent=2, ensure_ascii=False)
    print(f"\n  All results saved to: {results_path}")
    print("\n  Ablation experiments complete!")


if __name__ == "__main__":
    main()
