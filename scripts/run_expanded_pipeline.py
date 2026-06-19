#!/usr/bin/env python
"""
Expanded Data Pipeline: Download more data from Gaia DR3 and run comprehensive tests.

Compared to the original pipeline (3000/class = 18K total):
  - Downloads up to 5000+ sources per class
  - Balances classes to the minimum available count
  - Runs full model training + ablation experiments
  - Compares expanded results with original 18K baseline

Usage:
    python scripts/run_expanded_pipeline.py
    python scripts/run_expanded_pipeline.py --max-per-class 8000
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES, CONCEPT_NAMES_12, NUM_CONCEPTS, NUM_CLASSES,
    RANDOM_SEED,
)
from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.reproducibility import set_global_seed

# Add scripts dir to path for pipeline_utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (
    QUERIES, HAS_FOURIER, query_gaia_class,
    compute_derived_features, build_feature_matrix,
    compute_imputation_stats, apply_imputation,
)


def balance_classes(features, labels, max_per_class=None):
    """Balance classes by undersampling to the minimum class count."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()
    if max_per_class is not None:
        min_count = min(min_count, max_per_class)

    balanced_idx = []
    rng = np.random.RandomState(42)
    for cls in unique_labels:
        cls_idx = np.where(labels == cls)[0]
        selected = rng.choice(cls_idx, size=min_count, replace=False)
        balanced_idx.extend(selected)

    balanced_idx = np.sort(balanced_idx)
    return features[balanced_idx], labels[balanced_idx], min_count


def run_cv(cv_features, cv_labels, model_name, n_folds=5, max_epochs=100,
           patience=15, batch_size=256, output_dir=None, model_kwargs=None,
           tag=None, device="cpu", raw_features=None):
    """Run cross-validation for a single model."""
    from cbm_variable_stars.training.cross_val import run_cross_validation

    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "results" / "expanded")

    result = run_cross_validation(
        features=cv_features,
        labels=cv_labels,
        model_name=model_name,
        n_folds=n_folds,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        output_dir=output_dir,
        model_kwargs=model_kwargs or {},
        device=device,
        raw_features=raw_features,
    )
    return result


def subset_features(features, concept_subset):
    """Select a subset of concept columns from the feature matrix."""
    indices = [CONCEPT_NAMES_12.index(c) for c in concept_subset if c in CONCEPT_NAMES_12]
    return features[:, indices]


def print_result(tag, result, baseline_acc=None):
    """Print a concise result summary."""
    agg = result["aggregated"]
    acc_m = agg["accuracy_mean"]
    acc_s = agg["accuracy_std"]
    f1_m = agg["macro_f1_mean"]
    f1_s = agg["macro_f1_std"]
    print(f"  [{tag}] Acc: {acc_m:.4f}+/-{acc_s:.4f}, F1: {f1_m:.4f}+/-{f1_s:.4f}", end="")
    if baseline_acc is not None:
        delta = acc_m - baseline_acc
        print(f", delta={delta:+.4f}", end="")
    print()

    if "per_class_f1_mean" in agg:
        for i, cls in enumerate(CLASS_NAMES):
            cls_f1 = agg["per_class_f1_mean"][i]
            cls_std = agg["per_class_f1_std"][i]
            print(f"    {cls:15s} F1={cls_f1:.4f}+/-{cls_std:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Expanded data pipeline")
    parser.add_argument("--max-per-class", type=int, default=5000,
                        help="Max sources to query per class (default: 5000)")
    parser.add_argument("--output-dir", default="data/expanded",
                        help="Output directory for expanded data files")
    parser.add_argument("--results-dir", default="results/expanded",
                        help="Output directory for results")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Gaia download, use cached data")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation experiments")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(RANDOM_SEED)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cached_path = output_dir / "gaia_expanded_features.parquet"

    t_start = time.time()

    # ================================================================
    # STEP 1: Download expanded data from Gaia DR3
    # ================================================================
    if args.skip_download and cached_path.exists():
        print("\n" + "=" * 70)
        print("  Loading cached expanded data...")
        print("=" * 70)
        combined = pd.read_parquet(cached_path)
        print(f"  Loaded {len(combined)} sources from cache")
    else:
        print("\n" + "=" * 70)
        print(f"  STEP 1: Download Gaia DR3 data (up to {args.max_per_class}/class)")
        print("=" * 70)

        all_dfs = []
        for var_type in CLASS_NAMES:
            try:
                df = query_gaia_class(var_type, max_rows=args.max_per_class)
                if len(df) > 0:
                    all_dfs.append(df)
                    print(f"  {var_type}: {len(df)} sources retrieved")
                else:
                    print(f"  {var_type}: NO DATA")
            except Exception as e:
                print(f"  {var_type}: FAILED - {e}")
                traceback.print_exc()

        if not all_dfs:
            print("ERROR: No data downloaded!")
            sys.exit(1)

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n  Total raw sources: {len(combined)}")
        for cls in CLASS_NAMES:
            n = (combined["label_name"] == cls).sum()
            print(f"    {cls}: {n}")

        # Save raw
        combined.to_parquet(output_dir / "gaia_raw_expanded.parquet", index=False)

    # ================================================================
    # STEP 2: Feature engineering
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Compute derived features")
    print("=" * 70)

    combined = compute_derived_features(combined)
    features, labels, combined = build_feature_matrix(combined)
    print(f"  Feature matrix: {features.shape}")

    # Report per-class counts before balancing
    print("\n  Per-class counts (before balancing):")
    for i, cls in enumerate(CLASS_NAMES):
        n = (labels == i).sum()
        print(f"    {cls}: {n}")

    # Save raw features to cache (before imputation)
    combined.to_parquet(cached_path, index=False)

    # ================================================================
    # STEP 3: Balance classes + split + impute
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 3: Balance classes & train/test split")
    print("=" * 70)

    features_bal, labels_bal, per_class = balance_classes(features, labels)
    n_total = len(features_bal)
    print(f"  Balanced to {per_class}/class x {NUM_CLASSES} = {n_total} total")

    # Split: 15% holdout + 85% CV
    from cbm_variable_stars.data.splits import create_full_split
    from sklearn.preprocessing import StandardScaler

    split = create_full_split(labels_bal, test_ratio=0.15)
    cv_idx = split["cv_indices"]
    test_idx = split["test_indices"]

    # Fix C1: Impute AFTER split — compute stats on CV only, apply to both
    # [Fix C4] Test data uses global medians only (no per-class label routing)
    cv_features_raw = features_bal[cv_idx].copy()  # raw with NaN — for per-fold imputation (Fix C4)
    cv_labels_raw = labels_bal[cv_idx].copy()
    impute_stats = compute_imputation_stats(features_bal[cv_idx], labels_bal[cv_idx])
    cv_features_imputed = apply_imputation(features_bal[cv_idx], labels_bal[cv_idx], impute_stats, use_class_labels=True)
    test_features_imputed = apply_imputation(features_bal[test_idx], labels_bal[test_idx], impute_stats, use_class_labels=False)
    cv_labels = labels_bal[cv_idx]
    test_labels = labels_bal[test_idx]

    # Remove any remaining NaN rows
    cv_valid = ~np.any(np.isnan(cv_features_imputed), axis=1)
    test_valid = ~np.any(np.isnan(test_features_imputed), axis=1)
    cv_features_imputed = cv_features_imputed[cv_valid]
    cv_features_raw = cv_features_raw[cv_valid]  # Fix C4: sync raw features with NaN-removed labels
    cv_labels = cv_labels[cv_valid]
    test_features_imputed = test_features_imputed[test_valid]
    test_labels = test_labels[test_valid]

    print(f"  NaN after imputation (CV): {np.sum(np.isnan(cv_features_imputed))}")
    print(f"  NaN after imputation (test): {np.sum(np.isnan(test_features_imputed))}")

    # Scaler: fit on CV (imputed), for holdout evaluation path
    # CV path uses per-fold scaling inside cross_val.py (Fix C3)
    scaler = StandardScaler()
    cv_features_scaled = scaler.fit_transform(cv_features_imputed)
    test_features_scaled = scaler.transform(test_features_imputed)

    n_cv = len(cv_features_imputed)
    n_test = len(test_features_imputed)
    print(f"  CV pool:   {n_cv} ({100*n_cv/n_total:.0f}%)")
    print(f"  Test set:  {n_test} ({100*n_test/n_total:.0f}%)")
    print(f"  CV class distribution:")
    for i, cls in enumerate(CLASS_NAMES):
        n = (cv_labels == i).sum()
        print(f"    {cls}: {n}")

    # Save scaler
    import pickle
    with open(output_dir / "scaler_expanded.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ================================================================
    # STEP 4: Train all models (5-fold CV)
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 4: Train all models with 5-fold CV")
    print("=" * 70)

    from cbm_variable_stars.training.cross_val import run_cross_validation
    from cbm_variable_stars.data.splits import create_cv_splits
    from cbm_variable_stars.training.trainer import train_baseline

    cbm_models = ["hard_cbm", "hard_cbm_linear", "hard_cbm_cal", "soft_cbm", "cem", "mlp"]
    all_results = {}

    if args.device.startswith("cuda"):
        # GPU parallel: train all 6 models simultaneously using CUDA Streams
        from cbm_variable_stars.training.parallel_cv import run_parallel_cross_validation
        try:
            t0 = time.time()
            all_results = run_parallel_cross_validation(
                features=cv_features_imputed,  # fallback if raw_features fails
                labels=cv_labels,
                model_names=cbm_models,
                n_folds=args.n_folds,
                batch_size=256,
                max_epochs=args.max_epochs,
                patience=15,
                output_dir=str(results_dir),
                device=args.device,
                raw_features=cv_features_raw,  # Fix C4: per-fold imputation
            )
            dt = time.time() - t0
            for model_name in cbm_models:
                if model_name in all_results:
                    agg = all_results[model_name]["aggregated"]
                    print(f"  {model_name}: Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
                          f"F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")
            print(f"  Parallel time: {dt:.1f}s")
        except Exception as e:
            print(f"  Parallel training FAILED, falling back to serial: {e}")
            traceback.print_exc()
            all_results = {}

    # CPU serial (or fallback if parallel failed)
    for model_name in cbm_models:
        if model_name in all_results:
            continue
        print(f"\n--- Training: {model_name} ---")
        try:
            t0 = time.time()
            result = run_cv(
                cv_features_imputed, cv_labels, model_name,  # fallback if raw_features fails
                n_folds=args.n_folds, max_epochs=args.max_epochs,
                patience=15, output_dir=str(results_dir),
                device=args.device,
                raw_features=cv_features_raw,  # Fix C4: per-fold imputation
            )
            dt = time.time() - t0
            agg = result["aggregated"]
            print(f"  Acc: {agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}")
            print(f"  F1:  {agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")
            print(f"  Time: {dt:.1f}s")
            all_results[model_name] = result
        except Exception as e:
            print(f"  {model_name}: FAILED - {e}")
            traceback.print_exc()

    # Baselines: RF, XGBoost
    for model_type in ["rf", "xgb"]:
        print(f"\n--- Training: {model_type} ---")
        try:
            t0 = time.time()
            splits = create_cv_splits(cv_labels, n_folds=args.n_folds)
            fold_metrics = []
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                model, metrics = train_baseline(
                    model_type=model_type,
                    features_train=cv_features_imputed[train_idx],
                    labels_train=cv_labels[train_idx],
                    features_val=cv_features_imputed[val_idx],
                    labels_val=cv_labels[val_idx],
                )
                fold_metrics.append(metrics)
                print(f"  Fold {fold_idx+1}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

            mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
            std_acc = np.std([m["accuracy"] for m in fold_metrics], ddof=1)
            mean_f1 = np.mean([m["macro_f1"] for m in fold_metrics])
            std_f1 = np.std([m["macro_f1"] for m in fold_metrics], ddof=1)
            dt = time.time() - t0

            all_results[model_type] = {
                "aggregated": {
                    "accuracy_mean": mean_acc, "accuracy_std": std_acc,
                    "macro_f1_mean": mean_f1, "macro_f1_std": std_f1,
                },
            }
            print(f"  Mean: Acc={mean_acc:.4f}+/-{std_acc:.4f}, F1={mean_f1:.4f}+/-{std_f1:.4f}")
            print(f"  Time: {dt:.1f}s")
        except Exception as e:
            print(f"  {model_type}: FAILED - {e}")
            traceback.print_exc()

    # ================================================================
    # STEP 5: Hold-out evaluation
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 5: Hold-out test evaluation (HardCBM)")
    print("=" * 70)

    import torch
    from sklearn.model_selection import train_test_split as sk_split
    from cbm_variable_stars.models import create_model
    from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
    from cbm_variable_stars.training.trainer import evaluate_model, Trainer
    from cbm_variable_stars.losses.cbm_loss import CBMJointLoss, compute_class_weights

    # Fix C2: Split CV into train/val for early stopping (don't use test set)
    train_sub_idx, val_sub_idx = sk_split(
        np.arange(len(cv_labels)), test_size=0.15,
        stratify=cv_labels, random_state=RANDOM_SEED,
    )

    model = create_model("hard_cbm", num_concepts=NUM_CONCEPTS, num_classes=NUM_CLASSES)
    train_dataset = VariableStarDataset(
        features=cv_features_scaled[train_sub_idx],
        labels=cv_labels[train_sub_idx],
    )
    val_dataset = VariableStarDataset(
        features=cv_features_scaled[val_sub_idx],
        labels=cv_labels[val_sub_idx],
    )
    test_dataset = VariableStarDataset(
        features=test_features_scaled, labels=test_labels,
    )
    train_loader = create_dataloader(train_dataset, batch_size=256, shuffle=True, device=args.device)
    val_loader = create_dataloader(val_dataset, batch_size=256, shuffle=False, device=args.device)
    test_loader = create_dataloader(test_dataset, batch_size=256, shuffle=False, device=args.device)

    class_weights = compute_class_weights(torch.tensor(cv_labels[train_sub_idx], dtype=torch.long))
    loss_fn = CBMJointLoss(alpha=0.0, beta=1.0, class_weights=class_weights)

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        max_epochs=args.max_epochs, patience=15,
        device=args.device,
        log_dir=str(results_dir / "eval_logs"),
        checkpoint_dir=str(results_dir / "eval_ckpts"),
    )
    trainer.fit(train_loader, val_loader, fold_id=0)  # val_loader for early stopping, NOT test

    eval_result = evaluate_model(model, test_loader, device=args.device)
    holdout_acc = eval_result["accuracy"]
    holdout_f1 = eval_result["macro_f1"]
    print(f"  Hold-out Accuracy: {holdout_acc:.4f}")
    print(f"  Hold-out Macro F1: {holdout_f1:.4f}")

    if "per_class" in eval_result:
        print("  Per-class F1:")
        for cls_name, cls_metrics in eval_result["per_class"].items():
            if isinstance(cls_metrics, dict) and "f1" in cls_metrics:
                print(f"    {cls_name}: F1={cls_metrics['f1']:.4f}")

    # ================================================================
    # STEP 6: Ablation experiments (if not skipped)
    # ================================================================
    ablation_results = {}

    if not args.skip_ablation:
        print("\n" + "=" * 70)
        print("  STEP 6: Ablation experiments on expanded data")
        print("=" * 70)

        from cbm_variable_stars.shared.constants import MINIMAL_CONCEPTS

        # A0: Baseline
        print("\n--- A0: Baseline 12-concept HardCBM ---")
        baseline = run_cv(cv_features_imputed, cv_labels, "hard_cbm",
                          n_folds=args.n_folds, max_epochs=args.max_epochs,
                          device=args.device)
        baseline_acc = baseline["aggregated"]["accuracy_mean"]
        print_result("A0 Baseline", baseline)
        ablation_results["A0_baseline"] = baseline["aggregated"]

        # A1: Remove color_bp_rp
        print("\n--- A1: Remove color_bp_rp (11 concepts) ---")
        concepts_no_color = [c for c in CONCEPT_NAMES_12 if c != "color_bp_rp"]
        feat_no_color = subset_features(cv_features_imputed, concepts_no_color)
        a1 = run_cv(feat_no_color, cv_labels, "hard_cbm",
                     n_folds=args.n_folds, max_epochs=args.max_epochs,
                     model_kwargs={"num_concepts": 11}, device=args.device)
        print_result("A1 No color", a1, baseline_acc)
        ablation_results["A1_remove_color"] = a1["aggregated"]
        ablation_results["A1_remove_color"]["delta_accuracy"] = a1["aggregated"]["accuracy_mean"] - baseline_acc

        # A2: Minimal 4-concept set
        print("\n--- A2: Minimal 4 concepts {period, amplitude, R21, phi21} ---")
        feat_minimal = subset_features(cv_features_imputed, MINIMAL_CONCEPTS)
        a2 = run_cv(feat_minimal, cv_labels, "hard_cbm",
                     n_folds=args.n_folds, max_epochs=args.max_epochs,
                     model_kwargs={"num_concepts": 4}, device=args.device)
        a2_acc = a2["aggregated"]["accuracy_mean"]
        print_result("A2 Minimal", a2, baseline_acc)
        ablation_results["A2_minimal"] = a2["aggregated"]
        ablation_results["A2_minimal"]["delta_accuracy"] = a2_acc - baseline_acc
        ablation_results["A2_minimal"]["retained_pct"] = 100 * a2_acc / baseline_acc

        # A2b: Leave-one-concept-out
        print("\n--- A2b: Leave-One-Concept-Out ---")
        loo_results = {}
        for concept in CONCEPT_NAMES_12:
            remaining = [c for c in CONCEPT_NAMES_12 if c != concept]
            feat_sub = subset_features(cv_features_imputed, remaining)
            r = run_cv(feat_sub, cv_labels, "hard_cbm",
                       n_folds=args.n_folds, max_epochs=args.max_epochs,
                       model_kwargs={"num_concepts": len(remaining)},
                       device=args.device)
            delta = r["aggregated"]["accuracy_mean"] - baseline_acc
            print(f"  No {concept:15s}: Acc={r['aggregated']['accuracy_mean']:.4f}, delta={delta:+.4f}")
            loo_results[concept] = {
                "accuracy_mean": r["aggregated"]["accuracy_mean"],
                "accuracy_std": r["aggregated"]["accuracy_std"],
                "macro_f1_mean": r["aggregated"]["macro_f1_mean"],
                "delta_accuracy": delta,
            }
        ablation_results["A2b_leave_one_out"] = loo_results

        # Sort by importance
        print("\n  Concept importance ranking (by delta):")
        sorted_concepts = sorted(loo_results.items(), key=lambda x: x[1]["delta_accuracy"])
        for rank, (concept, info) in enumerate(sorted_concepts, 1):
            print(f"    {rank:2d}. {concept:15s}: delta={info['delta_accuracy']:+.4f}")

        # A4: Architecture comparison
        print("\n--- A4: Architecture comparison ---")
        arch_results = {}
        for arch in ["hard_cbm", "hard_cbm_cal", "soft_cbm", "cem"]:
            print(f"  Training {arch}...")
            r = run_cv(cv_features_imputed, cv_labels, arch,
                       n_folds=args.n_folds, max_epochs=args.max_epochs,
                       device=args.device)
            acc = r["aggregated"]["accuracy_mean"]
            f1 = r["aggregated"]["macro_f1_mean"]
            print(f"    {arch}: Acc={acc:.4f}, F1={f1:.4f}")
            arch_results[arch] = r["aggregated"]
        ablation_results["A4_architecture"] = arch_results

        # A5a: Linear vs MLP
        print("\n--- A5a: Linear vs MLP predictor ---")
        r_lin = run_cv(cv_features_imputed, cv_labels, "hard_cbm_linear",
                       n_folds=args.n_folds, max_epochs=args.max_epochs,
                       device=args.device)
        r_mlp = run_cv(cv_features_imputed, cv_labels, "hard_cbm",
                       n_folds=args.n_folds, max_epochs=args.max_epochs,
                       device=args.device)
        delta_acc = r_mlp["aggregated"]["accuracy_mean"] - r_lin["aggregated"]["accuracy_mean"]
        print(f"  Linear: Acc={r_lin['aggregated']['accuracy_mean']:.4f}")
        print(f"  MLP:    Acc={r_mlp['aggregated']['accuracy_mean']:.4f}")
        print(f"  Delta:  {delta_acc:+.4f}")
        ablation_results["A5a_predictor"] = {
            "linear": r_lin["aggregated"],
            "mlp": r_mlp["aggregated"],
            "delta_accuracy": delta_acc,
        }

    # ================================================================
    # STEP 7: Comparison with original 18K results
    # ================================================================
    print("\n" + "=" * 70)
    print("  STEP 7: Comparison with original 18K dataset")
    print("=" * 70)

    original_path = PROJECT_ROOT / "results" / "real" / "summary.json"
    if original_path.exists():
        with open(original_path) as f:
            original = json.load(f)

        print(f"\n  {'Model':<20} {'Original (18K)':>15} {'Expanded':>15} {'Delta':>10}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")

        for model_name in ["hard_cbm", "hard_cbm_linear", "hard_cbm_cal",
                           "soft_cbm", "cem", "mlp", "rf", "xgb"]:
            orig_acc = original.get("model_results", {}).get(model_name, {}).get("accuracy_mean", 0)
            if model_name in all_results:
                new_acc = all_results[model_name]["aggregated"]["accuracy_mean"]
                delta = new_acc - orig_acc
                print(f"  {model_name:<20} {orig_acc:>14.4f} {new_acc:>14.4f} {delta:>+9.4f}")
            else:
                print(f"  {model_name:<20} {orig_acc:>14.4f} {'N/A':>15} {'N/A':>10}")

        orig_holdout = original.get("holdout_accuracy", 0)
        delta_h = holdout_acc - orig_holdout
        print(f"\n  Hold-out:  Original={orig_holdout:.4f}, Expanded={holdout_acc:.4f}, Delta={delta_h:+.4f}")
    else:
        print("  Original results not found, skipping comparison.")

    # ================================================================
    # SAVE ALL RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("  Saving results...")
    print("=" * 70)

    summary = {
        "dataset": {
            "n_total": n_total,
            "n_cv": n_cv,
            "n_test": n_test,
            "per_class": int(per_class),
            "max_per_class_requested": args.max_per_class,
        },
        "model_results": {
            name: {k: float(v) for k, v in result.get("aggregated", {}).items()
                   if isinstance(v, (int, float, np.floating))}
            for name, result in all_results.items()
        },
        "holdout": {
            "accuracy": holdout_acc,
            "macro_f1": holdout_f1,
        },
        "ablation": ablation_results,
        "total_time_seconds": time.time() - t_start,
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    summary = convert_numpy(summary)

    with open(results_dir / "expanded_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Results saved to {results_dir / 'expanded_results.json'}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  COMPLETE ({total_time:.0f}s total)")
    print("=" * 70)
    print(f"\n  Dataset: {n_total} sources ({per_class}/class x {NUM_CLASSES})")
    print(f"  CV: {n_cv} | Test: {n_test}")
    print(f"\n  Model Performance (5-fold CV):")
    print(f"  {'Model':<20} {'Accuracy':>15} {'Macro F1':>15}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    for name in ["hard_cbm", "hard_cbm_linear", "hard_cbm_cal",
                  "soft_cbm", "cem", "mlp", "rf", "xgb"]:
        if name in all_results:
            agg = all_results[name]["aggregated"]
            print(f"  {name:<20} {agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}"
                  f" {agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")

    print(f"\n  Hold-out: Acc={holdout_acc:.4f}, F1={holdout_f1:.4f}")
    print(f"\n  Pipeline complete!")


if __name__ == "__main__":
    main()
