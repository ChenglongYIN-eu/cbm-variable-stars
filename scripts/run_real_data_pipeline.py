#!/usr/bin/env python
"""
Real Data Pipeline: Gaia DR3 Variable Star Classification

Queries Gaia DR3 variability tables directly for pre-computed features:
  - vari_rrlyrae:              RRAB, RRC (period, amplitude, R21, R31, phi21, phi31)
  - vari_cepheid:              DCEP      (period, amplitude, R21, R31, phi21, phi31)
  - vari_short_timescale:      DSCT/SXPHE (frequency, amplitude)
  - vari_eclipsing_binary:     ECL       (frequency)
  - vari_long_period_variable: MIRA/SR   (frequency, amplitude)
  - vari_summary:              ALL       (skewness, kurtosis, stetson_K, iqr, std)
  - gaia_source:               ALL       (bp_rp, phot_g_mean_mag)

For classes without catalog Fourier params (DSCT/SXPHE, ECL, MIRA/SR),
downloads epoch photometry and extracts features from light curves.

Usage:
    python scripts/run_real_data_pipeline.py
    python scripts/run_real_data_pipeline.py --max-per-class 500
    python scripts/run_real_data_pipeline.py --skip-epoch-download
"""

from __future__ import annotations
import argparse
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(description="Real data pipeline for CBM variable stars")
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="Max sources per class (default: 3000)")
    parser.add_argument("--output-dir", default="data/real",
                        help="Output directory for data files")
    parser.add_argument("--results-dir", default="results/real",
                        help="Output directory for model results")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="Max training epochs")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Gaia download, use cached data")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(RANDOM_SEED)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    cached_path = output_dir / "gaia_all_features.parquet"

    # ================================================================
    # STEP 1: Download real data from Gaia DR3
    # ================================================================
    if args.skip_download and cached_path.exists():
        logger.info("Loading cached data...")
        combined = pd.read_parquet(cached_path)
        logger.info(f"Loaded {len(combined)} sources from cache")
    else:
        print("\n" + "=" * 70)
        print("STEP 1: Download real data from Gaia DR3")
        print("=" * 70)

        all_dfs = []
        for var_type in CLASS_NAMES:
            try:
                df = query_gaia_class(var_type, max_rows=args.max_per_class)
                if len(df) > 0:
                    all_dfs.append(df)
                    print(f"  {var_type}: {len(df)} sources")
                else:
                    print(f"  {var_type}: NO DATA")
            except Exception as e:
                print(f"  {var_type}: FAILED - {e}")
                traceback.print_exc()

        if not all_dfs:
            print("ERROR: No data downloaded from any class!")
            sys.exit(1)

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n  Total sources: {len(combined)}")
        print(f"  Class distribution:")
        for cls in CLASS_NAMES:
            n = (combined["label_name"] == cls).sum()
            print(f"    {cls}: {n}")

        # Save raw data
        combined.to_parquet(output_dir / "gaia_raw_metadata.parquet", index=False)

    # ================================================================
    # STEP 2: Compute derived features
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Compute derived features")
    print("=" * 70)

    combined = compute_derived_features(combined)

    # Build feature matrix
    features, labels, combined = build_feature_matrix(combined)
    print(f"  Feature matrix: {features.shape}")
    print(f"  Label vector:   {labels.shape}")
    print(f"  NaN count per concept:")
    for i, name in enumerate(CONCEPT_NAMES_12):
        n_nan = np.sum(np.isnan(features[:, i]))
        if n_nan > 0:
            print(f"    {name}: {n_nan} ({100*n_nan/len(features):.1f}%)")

    # Save raw features to cache (before imputation to preserve NaN structure)
    combined.to_parquet(cached_path, index=False)
    print(f"  Saved raw features to {cached_path}")

    # ================================================================
    # STEP 3: Train/test split + impute + standardize
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Train/test split (15% hold-out + 85% CV)")
    print("=" * 70)

    from cbm_variable_stars.data.splits import create_full_split
    from sklearn.preprocessing import StandardScaler

    split = create_full_split(labels, test_ratio=0.15)
    cv_idx = split["cv_indices"]
    test_idx = split["test_indices"]

    # Fix C1: Impute AFTER split — compute stats on CV only, apply to both
    # [Fix C4] Test data uses global medians only (no per-class label routing)
    cv_features_raw = features[cv_idx].copy()  # raw with NaN — for per-fold imputation (Fix C4)
    cv_labels_raw = labels[cv_idx].copy()
    impute_stats = compute_imputation_stats(features[cv_idx], labels[cv_idx])
    cv_features_imputed = apply_imputation(features[cv_idx], labels[cv_idx], impute_stats, use_class_labels=True)
    test_features_imputed = apply_imputation(features[test_idx], labels[test_idx], impute_stats, use_class_labels=False)
    cv_labels = labels[cv_idx]
    test_labels = labels[test_idx]

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

    n_total_valid = len(cv_features_imputed) + len(test_features_imputed)
    print(f"  Total valid: {n_total_valid}")
    print(f"  CV subset:  {len(cv_features_imputed)} samples")
    print(f"  Test set:   {len(test_features_imputed)} samples")
    print(f"  CV class distribution:")
    for i, cls in enumerate(CLASS_NAMES):
        n = (cv_labels == i).sum()
        print(f"    {cls}: {n}")

    # Save scaler
    import pickle
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ================================================================
    # STEP 4: (Reserved — feature validation done inline above)
    # ================================================================

    # ================================================================
    # STEP 5: Train all models
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Train CBM models with cross-validation")
    print("=" * 70)

    from cbm_variable_stars.training.cross_val import run_cross_validation

    cbm_models = ["hard_cbm", "hard_cbm_linear", "hard_cbm_cal", "soft_cbm", "cem", "mlp"]
    all_results = {}

    if args.device.startswith("cuda"):
        # GPU parallel: train all 6 models simultaneously using CUDA Streams
        from cbm_variable_stars.training.parallel_cv import run_parallel_cross_validation
        try:
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
            for model_name in cbm_models:
                if model_name in all_results:
                    agg = all_results[model_name]["aggregated"]
                    print(f"  {model_name}: Accuracy={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f}, "
                          f"F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}")
        except Exception as e:
            print(f"  Parallel training FAILED, falling back to serial: {e}")
            traceback.print_exc()
            all_results = {}  # reset, fall through to serial

    # CPU serial (or fallback if parallel failed)
    for model_name in cbm_models:
        if model_name in all_results:
            continue  # already trained in parallel
        print(f"\n--- Training: {model_name} ---")
        try:
            result = run_cross_validation(
                features=cv_features_imputed,  # fallback if raw_features fails
                labels=cv_labels,
                model_name=model_name,
                n_folds=args.n_folds,
                batch_size=256,
                max_epochs=args.max_epochs,
                patience=15,
                output_dir=str(results_dir),
                device=args.device,
                raw_features=cv_features_raw,  # Fix C4: per-fold imputation
            )
            agg = result["aggregated"]
            print(f"  Accuracy: {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")
            print(f"  Macro F1: {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")
            all_results[model_name] = result
        except Exception as e:
            print(f"  {model_name}: FAILED - {e}")
            traceback.print_exc()

    # ================================================================
    # STEP 6: Train baselines (RF, XGBoost)
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Train baseline models")
    print("=" * 70)

    from cbm_variable_stars.data.splits import create_cv_splits
    from cbm_variable_stars.training.trainer import train_baseline

    for model_type in ["rf", "xgb"]:
        print(f"\n--- Training: {model_type} ---")
        try:
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
                print(f"  Fold {fold_idx}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

            mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
            mean_f1 = np.mean([m["macro_f1"] for m in fold_metrics])
            all_results[model_type] = {
                "aggregated": {
                    "accuracy_mean": mean_acc,
                    "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics], ddof=1)),
                    "macro_f1_mean": mean_f1,
                    "macro_f1_std": float(np.std([m["macro_f1"] for m in fold_metrics], ddof=1)),
                },
            }
            print(f"  {model_type} Mean: Acc={mean_acc:.4f}, F1={mean_f1:.4f}")
        except Exception as e:
            print(f"  {model_type}: FAILED - {e}")
            traceback.print_exc()

    # ================================================================
    # STEP 7: Hold-out evaluation
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Hold-out test set evaluation")
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

    model = create_model("hard_cbm")
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

    eval_result = evaluate_model(model, test_loader, device=args.device)  # final eval on truly unseen test set
    print(f"\n  Hold-out Accuracy:    {eval_result['accuracy']:.4f}")
    print(f"  Hold-out Macro F1:    {eval_result['macro_f1']:.4f}")
    print(f"  Hold-out Weighted F1: {eval_result['weighted_f1']:.4f}")

    if "per_class" in eval_result:
        print("\n  Per-class F1:")
        for cls_name, cls_metrics in eval_result["per_class"].items():
            if isinstance(cls_metrics, dict) and "f1" in cls_metrics:
                print(f"    {cls_name}: F1={cls_metrics['f1']:.4f}, "
                      f"Prec={cls_metrics.get('precision', 0):.4f}, "
                      f"Rec={cls_metrics.get('recall', 0):.4f}")

    # ================================================================
    # STEP 8: Experiments on real data
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Run experiments on real data")
    print("=" * 70)

    # 8a: Correlation analysis
    try:
        from cbm_variable_stars.experiments.correlation import run_correlation_analysis
        corr_result = run_correlation_analysis(
            features_df=cv_features_scaled,
            labels=cv_labels,
            output_dir=str(results_dir / "correlation"),
        )
        print("  Correlation analysis: DONE")
    except Exception as e:
        print(f"  Correlation analysis: FAILED - {e}")

    # 8b: Noise injection intervention
    try:
        from cbm_variable_stars.experiments.intervention import run_noise_injection_experiment
        test_data = {"features": test_features_scaled, "labels": test_labels}
        noise_result = run_noise_injection_experiment(
            model=model, test_data=test_data,
            noise_stds=[0.1, 0.5, 1.0, 2.0],
        )
        print(f"  Noise injection: clean_acc={noise_result['clean_accuracy']:.4f}")
        for sigma_key, sigma_data in noise_result["per_noise_level"].items():
            print(f"    sigma={sigma_key}: noisy_acc={sigma_data['accuracy_noisy_all']:.4f}")
    except Exception as e:
        print(f"  Noise injection: FAILED - {e}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Real Gaia DR3 Data")
    print("=" * 70)

    print(f"\n  Total sources: {n_total_valid}")
    print(f"  CV subset:     {len(cv_features_imputed)}")
    print(f"  Test set:      {len(test_features_imputed)}")

    print(f"\n  {'Model':<20} {'Accuracy':>12} {'Macro F1':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for name, result in all_results.items():
        agg = result.get("aggregated", {})
        acc_m = agg.get("accuracy_mean", 0)
        acc_s = agg.get("accuracy_std", 0)
        f1_m = agg.get("macro_f1_mean", 0)
        f1_s = agg.get("macro_f1_std", 0)
        print(f"  {name:<20} {acc_m:.4f}±{acc_s:.4f} {f1_m:.4f}±{f1_s:.4f}")

    print(f"\n  Hold-out: Acc={eval_result['accuracy']:.4f}, F1={eval_result['macro_f1']:.4f}")

    # Save summary
    import json
    summary = {
        "n_total": n_total_valid,
        "n_cv": len(cv_features_imputed),
        "n_test": len(test_features_imputed),
        "class_distribution": {cls: int((cv_labels == i).sum() + (test_labels == i).sum()) for i, cls in enumerate(CLASS_NAMES)},
        "model_results": {
            name: {k: float(v) for k, v in result.get("aggregated", {}).items()
                   if isinstance(v, (int, float))}
            for name, result in all_results.items()
        },
        "holdout_accuracy": eval_result["accuracy"],
        "holdout_macro_f1": eval_result["macro_f1"],
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {results_dir / 'summary.json'}")
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
