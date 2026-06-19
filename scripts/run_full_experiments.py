#!/usr/bin/env python
"""
Complete Experiment Suite for CBM Variable Star Classification

Uses cached real Gaia DR3 data (--skip-download from previous pipeline run).
Runs all experiments:
  1. Ablation A1: Remove color (11 vs 12 concepts)
  2. Ablation A2: Minimal 4-concept set
  3. Ablation A4: Architecture comparison (HardCBM vs SoftCBM vs CEM vs HardCBM_Cal)
  4. Ablation A5a: Label predictor complexity (Linear vs MLP)
  5. Intervention: Sequential random, greedy, noise injection, case studies
  6. Learning curve: Performance vs training data size
  7. Correlation analysis: Concept correlations, ANOVA, mutual information
  8. Significance tests: Paired t-test between models

Usage:
    python scripts/run_full_experiments.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES, CONCEPT_NAMES_12, LABEL_TO_IDX, NUM_CONCEPTS, NUM_CLASSES,
    RANDOM_SEED, N_CLASSES, MINIMAL_CONCEPTS, CONCEPTS_NO_COLOR,
)
from cbm_variable_stars.shared.reproducibility import set_global_seed

# ======================================================================
# Configuration
# ======================================================================
DATA_PATH = PROJECT_ROOT / "data" / "real" / "gaia_all_features.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 256
N_FOLDS = 5

# For learning curve: reduced sample sizes and repeats to save time
LEARNING_CURVE_SIZES = [500, 1000, 2000, 4000, 8000, 12000]
LEARNING_CURVE_REPEATS = 3


class SimpleConfig:
    """Minimal config object for ablation/experiment functions."""
    class project:
        random_seed = RANDOM_SEED
    class training:
        batch_size = BATCH_SIZE
        learning_rate = 1e-3
        weight_decay = 1e-4
        max_epochs = MAX_EPOCHS
        patience = PATIENCE
        device = "cpu"


cfg = SimpleConfig()


def load_data():
    """Load cached real data, split, and standardize."""
    print(f"Loading cached data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df)} sources")

    features = df[CONCEPT_NAMES_12].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)

    # Impute and split
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from pipeline_utils import compute_imputation_stats, apply_imputation
    from cbm_variable_stars.data.splits import create_full_split

    split = create_full_split(labels, test_ratio=0.15)
    cv_idx = split["cv_indices"]
    test_idx = split["test_indices"]

    imp_stats = compute_imputation_stats(features[cv_idx], labels[cv_idx])
    cv_features = apply_imputation(
        features[cv_idx], labels[cv_idx], imp_stats, use_class_labels=True,
    )
    test_features = apply_imputation(
        features[test_idx], labels[test_idx], imp_stats, use_class_labels=False,
    )
    cv_labels = labels[cv_idx]
    test_labels = labels[test_idx]

    print(f"  CV pool:  {len(cv_idx)} samples")
    print(f"  Test set: {len(test_idx)} samples")
    for i, cls in enumerate(CLASS_NAMES):
        n_cv = (cv_labels == i).sum()
        n_test = (test_labels == i).sum()
        print(f"    {cls}: CV={n_cv}, Test={n_test}")

    return cv_features, cv_labels, test_features, test_labels


def train_holdout_model(cv_features, cv_labels, test_features, test_labels,
                        concept_gt=None, model_name="hard_cbm", device="cpu"):
    """Train a model on full CV data for intervention experiments."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from cbm_variable_stars.models import create_model
    from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
    from cbm_variable_stars.training.trainer import Trainer
    from cbm_variable_stars.losses.cbm_loss import CBMJointLoss, compute_class_weights

    # Split CV into train_sub and val_sub for early stopping (avoid test leak)
    train_idx, val_idx = train_test_split(
        np.arange(len(cv_labels)), test_size=0.15,
        stratify=cv_labels, random_state=RANDOM_SEED,
    )

    train_features = cv_features[train_idx]
    train_labels = cv_labels[train_idx]
    val_features = cv_features[val_idx]
    val_labels = cv_labels[val_idx]

    # Concept ground-truth split (if available)
    train_concept_gt = concept_gt[train_idx] if concept_gt is not None else None
    val_concept_gt = concept_gt[val_idx] if concept_gt is not None else None

    # Standardize: fit on train_sub, transform val_sub and test
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    model = create_model(model_name)

    train_dataset = VariableStarDataset(features=train_features, labels=train_labels,
                                         concept_gt=train_concept_gt)
    val_dataset = VariableStarDataset(features=val_features, labels=val_labels,
                                       concept_gt=val_concept_gt)
    test_dataset = VariableStarDataset(features=test_features_scaled, labels=test_labels)

    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, device=device)
    val_loader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, device=device)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, device=device)

    class_weights = compute_class_weights(torch.tensor(cv_labels, dtype=torch.long))
    loss_fn = CBMJointLoss(alpha=0.0, beta=1.0, class_weights=class_weights)

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=device,
        log_dir=str(RESULTS_DIR / "holdout_logs"),
        checkpoint_dir=str(RESULTS_DIR / "holdout_ckpts"),
    )
    trainer.fit(train_loader, val_loader, fold_id=0)  # val_loader, NOT test_loader
    return model, scaler


def run_experiment(name, func, *args, **kwargs):
    """Run a single experiment with error handling and timing."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  [{name}] COMPLETED in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{name}] FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return {"error": str(e), "elapsed_seconds": elapsed}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Full experiment suite for CBM variable stars")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device: 'cpu' or 'cuda' (default: cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device
    cfg.training.device = device

    set_global_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Load Data
    # ================================================================
    cv_features, cv_labels, test_features, test_labels = load_data()

    all_experiment_results = {}

    # ================================================================
    # EXPERIMENT 1: Ablation A1 — Remove C11 (color_bp_rp)
    # ================================================================
    def exp_ablation_A1():
        from cbm_variable_stars.experiments.ablation import run_ablation_A1
        return run_ablation_A1(
            features=cv_features, labels=cv_labels, cfg=cfg,
            output_dir=RESULTS_DIR / "ablation",
        )

    all_experiment_results["A1_remove_color"] = run_experiment(
        "Ablation A1: Remove color_bp_rp", exp_ablation_A1,
    )

    # ================================================================
    # EXPERIMENT 2: Ablation A2 — Minimal 4-concept set
    # ================================================================
    def exp_ablation_A2():
        from cbm_variable_stars.experiments.ablation import run_ablation_A2
        return run_ablation_A2(
            features=cv_features, labels=cv_labels, cfg=cfg,
            output_dir=RESULTS_DIR / "ablation",
        )

    all_experiment_results["A2_minimal_concepts"] = run_experiment(
        "Ablation A2: Minimal 4-concept set", exp_ablation_A2,
    )

    # ================================================================
    # EXPERIMENT 3: Ablation A4 — Architecture comparison
    # ================================================================
    def exp_ablation_A4():
        from cbm_variable_stars.experiments.ablation import run_ablation_A4
        return run_ablation_A4(
            features=cv_features, labels=cv_labels, cfg=cfg,
            output_dir=RESULTS_DIR / "ablation",
        )

    all_experiment_results["A4_architecture_comparison"] = run_experiment(
        "Ablation A4: HardCBM vs SoftCBM vs CEM", exp_ablation_A4,
    )

    # ================================================================
    # EXPERIMENT 4: Ablation A5a — Linear vs MLP label predictor
    # ================================================================
    def exp_ablation_A5a():
        a5a_models = ["hard_cbm_linear", "hard_cbm"]
        a5a_output = str(RESULTS_DIR / "ablation" / "A5a")

        results = {}
        if device.startswith("cuda"):
            from cbm_variable_stars.training.parallel_cv import run_parallel_cross_validation
            try:
                all_res = run_parallel_cross_validation(
                    features=cv_features, labels=cv_labels,
                    model_names=a5a_models, n_folds=N_FOLDS,
                    batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                    patience=PATIENCE, output_dir=a5a_output,
                    device=device,
                )
                results = {m: all_res[m]["aggregated"] for m in a5a_models if m in all_res}
            except Exception as e:
                print(f"  Parallel A5a FAILED, falling back to serial: {e}")
                traceback.print_exc()
                results = {}

        # Serial fallback (or CPU path)
        from cbm_variable_stars.training.cross_val import run_cross_validation
        for model_name in a5a_models:
            if model_name in results:
                continue
            result = run_cross_validation(
                features=cv_features, labels=cv_labels,
                model_name=model_name, n_folds=N_FOLDS,
                batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                patience=PATIENCE, output_dir=a5a_output,
                device=device,
            )
            results[model_name] = result["aggregated"]

        # Compute delta
        delta = {
            "accuracy": (results["hard_cbm"]["accuracy_mean"]
                         - results["hard_cbm_linear"]["accuracy_mean"]),
            "macro_f1": (results["hard_cbm"]["macro_f1_mean"]
                         - results["hard_cbm_linear"]["macro_f1_mean"]),
        }
        return {
            "hard_cbm_linear": results["hard_cbm_linear"],
            "hard_cbm_mlp": results["hard_cbm"],
            "mlp_over_linear_delta": delta,
        }

    all_experiment_results["A5a_predictor_complexity"] = run_experiment(
        "Ablation A5a: Linear vs MLP predictor", exp_ablation_A5a,
    )

    # ================================================================
    # EXPERIMENT 5: Train hold-out model for intervention experiments
    # ================================================================
    print(f"\n{'='*70}")
    print("  Training HardCBM on full CV data for intervention experiments...")
    print(f"{'='*70}")
    model, scaler = train_holdout_model(cv_features, cv_labels, test_features, test_labels, device=device)
    test_features_scaled = scaler.transform(test_features)
    test_data = {"features": test_features_scaled, "labels": test_labels}

    # === Evaluate holdout model on test set ===
    from cbm_variable_stars.training.trainer import evaluate_model
    from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader

    test_dataset = VariableStarDataset(features=test_features_scaled, labels=test_labels)
    test_loader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, device=device)
    holdout_test_metrics = evaluate_model(model, test_loader, device=device)

    # Store in results (include new metrics: MCC, Kappa, AUC, Specificity)
    all_experiment_results["holdout_test"] = {
        "accuracy": holdout_test_metrics["accuracy"],
        "macro_f1": holdout_test_metrics["macro_f1"],
        "weighted_f1": holdout_test_metrics["weighted_f1"],
        "mcc": holdout_test_metrics.get("mcc"),
        "cohen_kappa": holdout_test_metrics.get("cohen_kappa"),
        "auc_roc_macro": holdout_test_metrics.get("auc_roc_macro"),
        "auc_roc_weighted": holdout_test_metrics.get("auc_roc_weighted"),
        "per_class": holdout_test_metrics["per_class"],
        "per_class_specificity": holdout_test_metrics.get("per_class_specificity", {}),
        "per_class_sensitivity": holdout_test_metrics.get("per_class_sensitivity", {}),
        "confusion_matrix": holdout_test_metrics["confusion_matrix"],
    }

    print(f"\n  Hold-out Test Results:")
    print(f"    Accuracy:    {holdout_test_metrics['accuracy']:.4f}")
    print(f"    Macro F1:    {holdout_test_metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {holdout_test_metrics['weighted_f1']:.4f}")
    print(f"    MCC:         {holdout_test_metrics.get('mcc', 0):.4f}")
    print(f"    Kappa:       {holdout_test_metrics.get('cohen_kappa', 0):.4f}")
    auc_val = holdout_test_metrics.get('auc_roc_macro')
    if auc_val is not None:
        print(f"    AUC-ROC:     {auc_val:.4f}")
    for cls_name, cls_metrics in holdout_test_metrics['per_class'].items():
        print(f"    {cls_name}: F1={cls_metrics['f1']:.4f}, P={cls_metrics['precision']:.4f}, R={cls_metrics['recall']:.4f}")

    # === Bootstrap CI for holdout test metrics ===
    if "all_preds" in holdout_test_metrics and "all_labels" in holdout_test_metrics:
        from cbm_variable_stars.evaluation.significance import bootstrap_confidence_interval
        from sklearn.metrics import accuracy_score, f1_score

        holdout_preds = np.array(holdout_test_metrics["all_preds"])
        holdout_labels = np.array(holdout_test_metrics["all_labels"])

        # Accuracy CI
        acc_ci = bootstrap_confidence_interval(
            y_true=holdout_labels, y_pred=holdout_preds,
            metric_fn=lambda yt, yp: accuracy_score(yt, yp),
            n_bootstrap=2000, confidence_level=0.95,
        )

        # Macro F1 CI
        f1_ci = bootstrap_confidence_interval(
            y_true=holdout_labels, y_pred=holdout_preds,
            metric_fn=lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
            n_bootstrap=2000, confidence_level=0.95,
        )

        all_experiment_results["holdout_bootstrap_ci"] = {
            "accuracy": acc_ci,
            "macro_f1": f1_ci,
        }

        print(f"\n  Bootstrap 95% CI (BCa, 2000 resamples):")
        acc_bounds = acc_ci.get("ci_0.95", (0, 0))
        f1_bounds = f1_ci.get("ci_0.95", (0, 0))
        print(f"    Accuracy: {acc_ci['point_estimate']:.4f}  "
              f"[{acc_bounds[0]:.4f}, {acc_bounds[1]:.4f}]")
        print(f"    Macro F1: {f1_ci['point_estimate']:.4f}  "
              f"[{f1_bounds[0]:.4f}, {f1_bounds[1]:.4f}]")

    # ================================================================
    # EXPERIMENT 6: Sequential Random Intervention
    # ================================================================
    def exp_intervention_random():
        from cbm_variable_stars.experiments.intervention import intervene_sequential_random
        return intervene_sequential_random(
            model=model, test_data=test_data,
            n_concepts=NUM_CONCEPTS, n_trials=5,
        )

    all_experiment_results["intervention_random"] = run_experiment(
        "Intervention: Sequential Random", exp_intervention_random,
    )

    # ================================================================
    # EXPERIMENT 7: Sequential Greedy Intervention
    # ================================================================
    def exp_intervention_greedy():
        from cbm_variable_stars.experiments.intervention import intervene_sequential_greedy
        return intervene_sequential_greedy(
            model=model, test_data=test_data, n_concepts=NUM_CONCEPTS,
        )

    all_experiment_results["intervention_greedy"] = run_experiment(
        "Intervention: Sequential Greedy", exp_intervention_greedy,
    )

    # ================================================================
    # EXPERIMENT 8: Noise Injection + Recovery
    # ================================================================
    def exp_noise_injection():
        from cbm_variable_stars.experiments.intervention import run_noise_injection_experiment
        return run_noise_injection_experiment(
            model=model, test_data=test_data,
            noise_stds=[0.1, 0.25, 0.5, 1.0, 2.0],
        )

    all_experiment_results["noise_injection"] = run_experiment(
        "Intervention: Noise Injection + Recovery", exp_noise_injection,
    )

    # ================================================================
    # EXPERIMENT 9: Case Studies (misclassified examples)
    # ================================================================
    def exp_case_studies():
        from cbm_variable_stars.experiments.intervention import run_case_studies
        return run_case_studies(
            model=model, test_data=test_data, n_cases=20,
        )

    all_experiment_results["case_studies"] = run_experiment(
        "Intervention: Case Studies", exp_case_studies,
    )

    # ================================================================
    # EXPERIMENT 10: Correlation Analysis
    # ================================================================
    def exp_correlation():
        from cbm_variable_stars.experiments.correlation import run_correlation_analysis
        return run_correlation_analysis(
            features_df=cv_features, labels=cv_labels,
            output_dir=str(RESULTS_DIR / "correlation"),
        )

    all_experiment_results["correlation"] = run_experiment(
        "Correlation Analysis", exp_correlation,
    )

    # ================================================================
    # EXPERIMENT 11: Learning Curve
    # ================================================================
    def exp_learning_curve():
        from cbm_variable_stars.experiments.learning_curve import run_learning_curve
        return run_learning_curve(
            features=cv_features, labels=cv_labels,
            model_name="hard_cbm",
            sample_sizes=LEARNING_CURVE_SIZES,
            n_repeats=LEARNING_CURVE_REPEATS,
            cfg=cfg,
            output_dir=RESULTS_DIR / "learning_curve",
        )

    all_experiment_results["learning_curve"] = run_experiment(
        "Learning Curve: HardCBM", exp_learning_curve,
    )

    # ================================================================
    # EXPERIMENT 12: Significance Tests (Paired t-test on CV folds)
    # ================================================================
    def exp_significance():
        from cbm_variable_stars.evaluation.significance import paired_cv_ttest

        # Collect per-fold scores from ablation A4
        a4_result = all_experiment_results.get("A4_architecture_comparison", {})
        if "error" in a4_result:
            return {"skipped": True, "reason": "A4 failed"}

        # Get detailed fold results
        detailed = a4_result.get("detailed_results", {})
        pairs_to_test = [
            ("hard_cbm", "soft_cbm"),
            ("hard_cbm", "hard_cbm_cal"),
            ("hard_cbm", "cem"),
            ("soft_cbm", "cem"),
        ]

        significance_results = {}
        for name_a, name_b in pairs_to_test:
            res_a = detailed.get(name_a, {})
            res_b = detailed.get(name_b, {})
            folds_a = res_a.get("fold_results", [])
            folds_b = res_b.get("fold_results", [])

            if not folds_a or not folds_b:
                significance_results[f"{name_a}_vs_{name_b}"] = {
                    "skipped": True, "reason": "Missing fold results",
                }
                continue

            # Extract per-fold macro F1
            scores_a = []
            scores_b = []
            for fr in folds_a:
                m = fr.get("metrics", fr)
                val = m.get("val_macro_f1", m.get("macro_f1", None))
                if val is not None:
                    scores_a.append(float(val))
            for fr in folds_b:
                m = fr.get("metrics", fr)
                val = m.get("val_macro_f1", m.get("macro_f1", None))
                if val is not None:
                    scores_b.append(float(val))

            if len(scores_a) >= 3 and len(scores_b) >= 3:
                test_result = paired_cv_ttest(
                    scores_a=scores_a, scores_b=scores_b,
                    model_a_name=name_a, model_b_name=name_b,
                )
                significance_results[f"{name_a}_vs_{name_b}"] = test_result
            else:
                significance_results[f"{name_a}_vs_{name_b}"] = {
                    "skipped": True,
                    "reason": f"Too few folds: a={len(scores_a)}, b={len(scores_b)}",
                }

        # Collect all p-values for multiple comparison correction
        p_values_list = []
        for pair_key, result in significance_results.items():
            if isinstance(result, dict) and "p_value" in result:
                p_values_list.append((pair_key, result["p_value"]))

        # Apply Holm-Bonferroni correction to control family-wise error rate
        if p_values_list:
            from cbm_variable_stars.evaluation.significance import holm_bonferroni
            corrected = holm_bonferroni(p_values_list, alpha=0.05)
            significance_results["holm_bonferroni_correction"] = corrected
            print(f"\n  Holm-Bonferroni corrected results ({len(p_values_list)} comparisons):")
            for entry in corrected:
                print(f"    {entry['comparison']}: p={entry['p_value']:.4f}, "
                      f"adjusted_alpha={entry['adjusted_alpha']:.4f}, "
                      f"significant={entry['significant']}")

        return significance_results

    all_experiment_results["significance_tests"] = run_experiment(
        "Significance Tests: Paired t-test", exp_significance,
    )

    # ================================================================
    # SAVE ALL RESULTS
    # ================================================================
    print(f"\n{'='*70}")
    print("  SAVING ALL EXPERIMENT RESULTS")
    print(f"{'='*70}")

    # Save full results (convert numpy to native types)
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        return obj

    serializable_results = make_serializable(all_experiment_results)

    # Remove large nested objects (fold_results with predictions) to keep file manageable
    def prune_large_fields(obj, max_depth=6, depth=0):
        if depth > max_depth:
            return "..."
        if isinstance(obj, dict):
            pruned = {}
            for k, v in obj.items():
                if k in ("predictions", "true_labels", "val_indices") and isinstance(v, list) and len(v) > 100:
                    pruned[k] = f"[{len(v)} items]"
                elif k == "fold_results" and isinstance(v, list):
                    pruned[k] = [
                        {kk: vv for kk, vv in fr.items()
                         if kk not in ("predictions", "true_labels", "val_indices")}
                        if isinstance(fr, dict) else fr
                        for fr in v
                    ]
                else:
                    pruned[k] = prune_large_fields(v, max_depth, depth + 1)
            return pruned
        if isinstance(obj, list) and len(obj) > 200:
            return obj[:10] + ["..."]
        return obj

    pruned = prune_large_fields(serializable_results)

    results_path = RESULTS_DIR / "full_experiment_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(pruned, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Full results saved to: {results_path}")

    # ================================================================
    # PRINT COMPREHENSIVE REPORT
    # ================================================================
    print_comprehensive_report(all_experiment_results)

    print(f"\nAll experiments complete! Results in: {RESULTS_DIR}")


def print_comprehensive_report(results):
    """Print a detailed report of all experiment results."""
    print("\n" + "=" * 80)
    print("=" * 80)
    print("  COMPREHENSIVE CBM VARIABLE STAR CLASSIFICATION REPORT")
    print("  Data Source: Gaia DR3 (18,000 real variable stars)")
    print("=" * 80)
    print("=" * 80)

    # --- Section 1: Cross-validation results + Hold-out Test ---
    print("\n" + "-" * 80)
    print("  1. MODEL PERFORMANCE (5-Fold Cross-Validation + Hold-out Test)")
    print("-" * 80)

    # Try to load CV summary from summary.json (generated by another script)
    summary_path = PROJECT_ROOT / "results" / "real" / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        print(f"\n  Total: {summary['n_total']} sources | "
              f"CV: {summary['n_cv']} | Test: {summary['n_test']}")
        print(f"\n  {'Model':<20} {'Accuracy':>14} {'Macro F1':>14}")
        print(f"  {'─'*20} {'─'*14} {'─'*14}")

        model_results = summary.get("model_results", {})
        for name in ["soft_cbm", "hard_cbm_cal", "cem", "hard_cbm", "hard_cbm_linear", "mlp", "rf", "xgb"]:
            if name in model_results:
                r = model_results[name]
                acc = f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
                f1 = f"{r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.4f}"
                print(f"  {name:<20} {acc:>14} {f1:>14}")

    # Use our own holdout_test results (computed in this script)
    holdout = results.get("holdout_test", {})
    if holdout and "accuracy" in holdout:
        print(f"\n  Hold-out Test (computed in this run):")
        print(f"    Accuracy:    {holdout['accuracy']:.4f}")
        print(f"    Macro F1:    {holdout['macro_f1']:.4f}")
        print(f"    Weighted F1: {holdout['weighted_f1']:.4f}")
        per_class = holdout.get("per_class", {})
        if per_class:
            print(f"\n    {'Class':<18} {'F1':>8} {'Precision':>10} {'Recall':>8}")
            print(f"    {'─'*18} {'─'*8} {'─'*10} {'─'*8}")
            for cls_name, cls_m in per_class.items():
                print(f"    {cls_name:<18} {cls_m['f1']:>8.4f} {cls_m['precision']:>10.4f} {cls_m['recall']:>8.4f}")
    else:
        print(f"\n  Hold-out Test: not available (holdout evaluation did not run)")

    # --- Section 2: Ablation A1 ---
    print("\n" + "-" * 80)
    print("  2. ABLATION A1: Color Feature Contribution (color_bp_rp)")
    print("-" * 80)

    a1 = results.get("A1_remove_color", {})
    if "error" not in a1:
        no_color = a1.get("no_color_model", {}).get("results", {})
        full = a1.get("full_model", {}).get("results", {})
        delta = a1.get("performance_delta", {})
        print(f"\n  Full 12-concept:    Acc={full.get('accuracy_mean', 0):.4f}±"
              f"{full.get('accuracy_std', 0):.4f}, "
              f"F1={full.get('macro_f1_mean', 0):.4f}")
        print(f"  No-color (11-dim):  Acc={no_color.get('accuracy_mean', 0):.4f}±"
              f"{no_color.get('accuracy_std', 0):.4f}, "
              f"F1={no_color.get('macro_f1_mean', 0):.4f}")
        print(f"  Delta (no_color - full): Acc={delta.get('accuracy', 0):+.4f}, "
              f"F1={delta.get('macro_f1', 0):+.4f}")
        if delta.get("accuracy", 0) < -0.005:
            print("  → color_bp_rp contributes meaningfully to classification")
        elif delta.get("accuracy", 0) > 0.005:
            print("  → Removing color_bp_rp IMPROVES performance (possible noise)")
        else:
            print("  → color_bp_rp has marginal impact")
    else:
        print(f"  FAILED: {a1.get('error', 'unknown')}")

    # --- Section 3: Ablation A2 ---
    print("\n" + "-" * 80)
    print("  3. ABLATION A2: Minimal Concept Set")
    print("-" * 80)

    a2 = results.get("A2_minimal_concepts", {})
    if "error" not in a2:
        minimal = a2.get("minimal_model", {}).get("results", {})
        full = a2.get("full_model", {}).get("results", {})
        delta = a2.get("performance_delta", {})
        print(f"\n  Minimal concepts: {MINIMAL_CONCEPTS}")
        print(f"  Full 12-concept:    Acc={full.get('accuracy_mean', 0):.4f}, "
              f"F1={full.get('macro_f1_mean', 0):.4f}")
        print(f"  Minimal 4-concept:  Acc={minimal.get('accuracy_mean', 0):.4f}, "
              f"F1={minimal.get('macro_f1_mean', 0):.4f}")
        print(f"  Delta (min - full): Acc={delta.get('accuracy', 0):+.4f}, "
              f"F1={delta.get('macro_f1', 0):+.4f}")
        retained = (minimal.get("accuracy_mean", 0) / full.get("accuracy_mean", 1) * 100
                     if full.get("accuracy_mean", 0) > 0 else 0)
        print(f"  → 4 concepts retain {retained:.1f}% of full-model accuracy")
    else:
        print(f"  FAILED: {a2.get('error', 'unknown')}")

    # --- Section 4: Ablation A4 ---
    print("\n" + "-" * 80)
    print("  4. ABLATION A4: Architecture Comparison")
    print("-" * 80)

    a4 = results.get("A4_architecture_comparison", {})
    if "error" not in a4:
        comparison = a4.get("comparison_table", {})
        print(f"\n  {'Architecture':<25} {'Accuracy':>14} {'Macro F1':>14}")
        print(f"  {'─'*25} {'─'*14} {'─'*14}")
        for arch_name in ["hard_cbm", "hard_cbm_cal", "soft_cbm", "cem"]:
            if arch_name in comparison:
                c = comparison[arch_name]
                print(f"  {arch_name:<25} "
                      f"{c.get('accuracy_mean', 0):.4f}±{c.get('accuracy_std', 0):.4f} "
                      f"{c.get('macro_f1_mean', 0):.4f}±{c.get('macro_f1_std', 0):.4f}")
        # Find best
        best_arch = max(comparison, key=lambda k: comparison[k].get("accuracy_mean", 0))
        print(f"\n  → Best CBM architecture: {best_arch} "
              f"(Acc={comparison[best_arch].get('accuracy_mean', 0):.4f})")
    else:
        print(f"  FAILED: {a4.get('error', 'unknown')}")

    # --- Section 5: Ablation A5a ---
    print("\n" + "-" * 80)
    print("  5. ABLATION A5a: Label Predictor Complexity (Plan A)")
    print("-" * 80)

    a5a = results.get("A5a_predictor_complexity", {})
    if "error" not in a5a:
        linear_r = a5a.get("hard_cbm_linear", {})
        mlp_r = a5a.get("hard_cbm_mlp", {})
        delta = a5a.get("mlp_over_linear_delta", {})
        print(f"\n  Linear predictor: Acc={linear_r.get('accuracy_mean', 0):.4f}±"
              f"{linear_r.get('accuracy_std', 0):.4f}, "
              f"F1={linear_r.get('macro_f1_mean', 0):.4f}")
        print(f"  MLP predictor:    Acc={mlp_r.get('accuracy_mean', 0):.4f}±"
              f"{mlp_r.get('accuracy_std', 0):.4f}, "
              f"F1={mlp_r.get('macro_f1_mean', 0):.4f}")
        print(f"  Delta (MLP - Linear): Acc={delta.get('accuracy', 0):+.4f}, "
              f"F1={delta.get('macro_f1', 0):+.4f}")
    else:
        print(f"  FAILED: {a5a.get('error', 'unknown')}")

    # --- Section 6: Intervention ---
    print("\n" + "-" * 80)
    print("  6. CONCEPT INTERVENTION EXPERIMENTS")
    print("-" * 80)

    # 6a: Greedy intervention
    greedy = results.get("intervention_greedy", {})
    if "error" not in greedy:
        print(f"\n  6a. Sequential Greedy Intervention")
        print(f"      Baseline accuracy (no intervention): {greedy.get('baseline_accuracy', 0):.4f}")
        print(f"      Full intervention (all 12 concepts): {greedy.get('full_accuracy', 0):.4f}")
        order = greedy.get("concept_names_order", greedy.get("intervention_order", []))
        accs = greedy.get("accuracies", [])
        gains = greedy.get("marginal_gains", [])
        print(f"\n      Concept intervention order (most → least impactful):")
        for i, (concept, acc, gain) in enumerate(zip(order, accs[1:] if len(accs) > 1 else accs, gains)):
            print(f"        {i+1}. {concept:<18} → Acc={acc:.4f} (gain={gain:+.4f})")
    else:
        print(f"  Greedy intervention FAILED: {greedy.get('error', 'unknown')}")

    # 6b: Random intervention
    random_int = results.get("intervention_random", {})
    if "error" not in random_int:
        print(f"\n  6b. Sequential Random Intervention (avg over {random_int.get('n_trials', 5)} trials)")
        mean_accs = random_int.get("mean_accuracies", [])
        if mean_accs:
            print(f"      0 interventions: {mean_accs[0]:.4f}")
            for n_steps in [1, 3, 6, 9, 12]:
                if n_steps < len(mean_accs):
                    print(f"      {n_steps} interventions: {mean_accs[n_steps]:.4f}")

    # 6c: Noise injection
    noise = results.get("noise_injection", {})
    if "error" not in noise:
        print(f"\n  6c. Noise Injection Experiment")
        print(f"      Clean accuracy: {noise.get('clean_accuracy', 0):.4f}")
        per_noise = noise.get("per_noise_level", {})
        print(f"\n      {'σ':>8} {'Noisy Acc':>12} {'Recovered':>12} {'Drop':>10}")
        print(f"      {'─'*8} {'─'*12} {'─'*12} {'─'*10}")
        for sigma_key, sigma_data in per_noise.items():
            noisy = sigma_data.get("accuracy_noisy_all", 0)
            # Try to find best recovery
            best_recovery = sigma_data.get("best_recovery_accuracy", noisy)
            drop = noisy - noise.get("clean_accuracy", 0)
            print(f"      {sigma_key:>8} {noisy:>12.4f} {best_recovery:>12.4f} {drop:>+10.4f}")
    else:
        print(f"  Noise injection FAILED: {noise.get('error', 'unknown')}")

    # 6d: Case studies
    cases = results.get("case_studies", {})
    if "error" not in cases:
        print(f"\n  6d. Case Studies")
        print(f"      Total test samples: {cases.get('total_samples', 0)}")
        print(f"      Total misclassified: {cases.get('total_misclassified', 0)}")
        print(f"      Analyzed: {cases.get('n_cases_returned', 0)}")
        case_list = cases.get("cases", [])
        if case_list:
            n_fixed = sum(1 for c in case_list if c.get("corrected_by_intervention", False))
            print(f"      Fixed by intervention: {n_fixed}/{len(case_list)}")

    # --- Section 7: Correlation Analysis ---
    print("\n" + "-" * 80)
    print("  7. CONCEPT CORRELATION ANALYSIS")
    print("-" * 80)

    corr = results.get("correlation", {})
    if "error" not in corr:
        # ANOVA ranking
        assoc = corr.get("class_association", {})
        ranking = assoc.get("concept_ranking", [])
        if ranking:
            print(f"\n  Concept discriminative power (ANOVA F-statistic):")
            print(f"  {'Rank':>4} {'Concept':<18} {'F-stat':>10} {'p-value':>12}")
            print(f"  {'─'*4} {'─'*18} {'─'*10} {'─'*12}")
            for i, item in enumerate(ranking[:12]):
                if isinstance(item, dict):
                    print(f"  {i+1:>4} {item.get('concept', '?'):<18} "
                          f"{item.get('f_statistic', 0):>10.1f} "
                          f"{item.get('p_value', 1):>12.2e}")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    print(f"  {i+1:>4} {item[0]:<18} {item[1]:>10.1f}")

        # MI ranking
        mi = corr.get("mutual_information", {})
        mi_ranking = mi.get("mi_ranking", [])
        if mi_ranking:
            print(f"\n  Mutual Information ranking:")
            for i, item in enumerate(mi_ranking[:12]):
                if isinstance(item, dict):
                    print(f"    {i+1}. {item.get('concept', '?')}: "
                          f"MI={item.get('mutual_information', 0):.4f}")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    print(f"    {i+1}. {item[0]}: MI={item[1]:.4f}")

        # High correlations
        corr_data = corr.get("correlation", {})
        high_corr = corr_data.get("high_correlations", [])
        if high_corr:
            print(f"\n  Highly correlated concept pairs (|r| > 0.7):")
            for pair in high_corr[:10]:
                if isinstance(pair, dict):
                    print(f"    {pair.get('concept_a', '?')} <-> {pair.get('concept_b', '?')}: "
                          f"r={pair.get('correlation', 0):.3f}")
                elif isinstance(pair, (list, tuple)) and len(pair) >= 3:
                    print(f"    {pair[0]} ↔ {pair[1]}: r={pair[2]:.3f}")
    else:
        print(f"  FAILED: {corr.get('error', 'unknown')}")

    # --- Section 8: Learning Curve ---
    print("\n" + "-" * 80)
    print("  8. LEARNING CURVE")
    print("-" * 80)

    lc = results.get("learning_curve", {})
    if "error" not in lc:
        lc_results = lc.get("results", {})
        if lc_results:
            print(f"\n  HardCBM accuracy vs training set size:")
            print(f"  {'N_train':>8} {'Accuracy':>14} {'Macro F1':>14}")
            print(f"  {'─'*8} {'─'*14} {'─'*14}")
            for size_key in sorted(lc_results.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                r = lc_results[size_key]
                if isinstance(r, dict):
                    acc_m = r.get("accuracy_mean", r.get("accuracy", {}).get("mean", 0))
                    acc_s = r.get("accuracy_std", r.get("accuracy", {}).get("std", 0))
                    f1_m = r.get("macro_f1_mean", r.get("macro_f1", {}).get("mean", 0))
                    f1_s = r.get("macro_f1_std", r.get("macro_f1", {}).get("std", 0))
                    print(f"  {size_key:>8} {acc_m:.4f}±{acc_s:.4f} {f1_m:.4f}±{f1_s:.4f}")
    else:
        print(f"  FAILED: {lc.get('error', 'unknown')}")

    # --- Section 9: Significance Tests ---
    print("\n" + "-" * 80)
    print("  9. STATISTICAL SIGNIFICANCE TESTS")
    print("-" * 80)

    sig = results.get("significance_tests", {})
    if "error" not in sig:
        print(f"\n  Paired t-test on 5-fold CV Macro F1:")
        print(f"  {'Comparison':<30} {'t-stat':>8} {'p-value':>10} {'Cohen d':>8} {'Sig?':>6}")
        print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*8} {'─'*6}")
        for comp_name, test_result in sig.items():
            if isinstance(test_result, dict) and "t_statistic" in test_result:
                sig_str = "***" if test_result.get("p_value", 1) < 0.001 else \
                          "**" if test_result.get("p_value", 1) < 0.01 else \
                          "*" if test_result.get("p_value", 1) < 0.05 else "n.s."
                print(f"  {comp_name:<30} "
                      f"{test_result.get('t_statistic', 0):>8.3f} "
                      f"{test_result.get('p_value', 1):>10.4f} "
                      f"{test_result.get('cohens_d', 0):>8.3f} "
                      f"{sig_str:>6}")

        # Holm-Bonferroni corrected results
        hb_correction = sig.get("holm_bonferroni_correction", [])
        if hb_correction:
            print(f"\n  Holm-Bonferroni multiple comparison correction ({len(hb_correction)} tests):")
            print(f"  {'Comparison':<30} {'p-value':>10} {'adj. alpha':>12} {'Sig?':>6}")
            print(f"  {'─'*30} {'─'*10} {'─'*12} {'─'*6}")
            for entry in hb_correction:
                if isinstance(entry, dict):
                    sig_str = "Yes" if entry.get("significant", False) else "No"
                    print(f"  {entry.get('comparison', '?'):<30} "
                          f"{entry.get('p_value', 1):>10.4f} "
                          f"{entry.get('adjusted_alpha', 0):>12.4f} "
                          f"{sig_str:>6}")
    else:
        print(f"  FAILED: {sig.get('error', 'unknown')}")

    # --- Section 10: Bootstrap CI for holdout test ---
    holdout_ci = results.get("holdout_bootstrap_ci", {})
    if holdout_ci:
        print("\n" + "-" * 80)
        print("  10. HOLDOUT TEST BOOTSTRAP CONFIDENCE INTERVALS")
        print("-" * 80)
        for metric_name, ci_data in holdout_ci.items():
            if isinstance(ci_data, dict):
                ci_bounds = ci_data.get("ci_0.95", (0, 0))
                print(f"\n  {metric_name}:")
                print(f"    Point estimate: {ci_data.get('point_estimate', 0):.4f}")
                print(f"    95% CI [{ci_bounds[0]:.4f}, {ci_bounds[1]:.4f}]  "
                      f"(method: {ci_data.get('method', '?')})")

    print("\n" + "=" * 80)
    print("  END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
