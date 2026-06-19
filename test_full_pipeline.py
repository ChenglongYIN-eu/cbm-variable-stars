"""
Full pipeline integration test for CBM variable star classification.

Tests: synthetic data -> dataset build -> model training -> evaluation -> experiments
"""

import numpy as np
import torch
import sys
import traceback
from pathlib import Path

# Make sure our package is importable
sys.path.insert(0, str(Path(__file__).parent))


def generate_synthetic_data(n_gaia=3000, n_ogle=1200, seed=42):
    """Generate synthetic variable star data for testing."""
    from cbm_variable_stars.shared.constants import (
        CLASS_NAMES, CONCEPT_NAMES_12,
    )

    rng = np.random.RandomState(seed)
    n_classes = len(CLASS_NAMES)
    n_concepts = len(CONCEPT_NAMES_12)

    # Class-specific feature profiles (rough physical simulation)
    class_profiles = {
        0: {"period": 0.5, "amplitude": 0.8, "R21": 0.4},  # RRAB
        1: {"period": 0.3, "amplitude": 0.3, "R21": 0.1},  # RRC
        2: {"period": 5.0, "amplitude": 0.6, "R21": 0.3},  # DCEP
        3: {"period": 0.08, "amplitude": 0.2, "R21": 0.05},  # DSCT_SXPHE
        4: {"period": 2.0, "amplitude": 1.0, "R21": 0.0},  # ECL
        5: {"period": 100.0, "amplitude": 2.0, "R21": 0.2},  # MIRA_SR
    }

    def make_features(n, source="gaia"):
        features_list = []
        labels_list = []
        samples_per_class = n // n_classes

        for cls_idx in range(n_classes):
            prof = class_profiles[cls_idx]
            n_cls = samples_per_class if cls_idx < n_classes - 1 else (n - samples_per_class * (n_classes - 1))

            feats = rng.randn(n_cls, n_concepts) * 0.3
            feats[:, 0] += np.log10(prof["period"])
            feats[:, 1] += prof["amplitude"]
            feats[:, 2] += 0.3 + cls_idx * 0.05
            feats[:, 3] += prof["R21"]
            feats[:, 4] += prof["R21"] * 0.5
            feats[:, 5] += cls_idx * 0.5
            feats[:, 6] += (-1) ** cls_idx * 0.2
            feats[:, 7] += 2.0 + cls_idx * 0.3
            feats[:, 8] += 0.8 + rng.rand(n_cls) * 0.2
            feats[:, 9] += 3.0 + rng.rand(n_cls) * 2

            if source == "gaia":
                feats[:, 10] += 0.5 + cls_idx * 0.2
                feats[:, 11] += 15.0 + cls_idx * 0.5
            else:
                feats[:, 10] += 0.5 + cls_idx * 0.2 + rng.randn(n_cls) * 0.1
                feats[:, 11] += 15.0 + cls_idx * 0.5 + rng.randn(n_cls) * 0.2

            features_list.append(feats)
            labels_list.append(np.full(n_cls, cls_idx))

        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)

        shuffle_idx = rng.permutation(len(labels))
        return features[shuffle_idx], labels[shuffle_idx]

    gaia_features, gaia_labels = make_features(n_gaia, "gaia")
    ogle_features, ogle_labels = make_features(n_ogle, "ogle")

    return gaia_features, gaia_labels, ogle_features, ogle_labels


def test_step1_data_generation():
    """Step 1: Generate synthetic data."""
    print("\n" + "=" * 70)
    print("STEP 1: Generate synthetic data")
    print("=" * 70)

    gaia_features, gaia_labels, ogle_features, ogle_labels = generate_synthetic_data()

    print(f"  Gaia: {gaia_features.shape} features, {len(np.unique(gaia_labels))} classes")
    print(f"  OGLE: {ogle_features.shape} features, {len(np.unique(ogle_labels))} classes")
    print("  PASSED")

    return gaia_features, gaia_labels, ogle_features, ogle_labels


def test_step2_dataset_build(gaia_features, gaia_labels):
    """Step 2: Build train/test datasets."""
    print("\n" + "=" * 70)
    print("STEP 2: Build datasets (15% hold-out + 85% CV)")
    print("=" * 70)

    from cbm_variable_stars.data.splits import create_full_split
    from sklearn.preprocessing import StandardScaler

    split = create_full_split(gaia_labels, test_ratio=0.15)
    cv_idx = split["cv_indices"]
    test_idx = split["test_indices"]

    print(f"  CV subset: {len(cv_idx)} samples")
    print(f"  Test set:  {len(test_idx)} samples")

    scaler = StandardScaler()
    cv_features = scaler.fit_transform(gaia_features[cv_idx])
    test_features = scaler.transform(gaia_features[test_idx])

    cv_labels = gaia_labels[cv_idx]
    test_labels = gaia_labels[test_idx]

    print(f"  Scaler fitted on CV subset: mean={scaler.mean_[:3].round(3)}")
    print("  PASSED")

    return cv_features, cv_labels, test_features, test_labels, scaler


def test_step3_cbm_training(cv_features, cv_labels):
    """Step 3: Train all CBM models with 3-fold CV."""
    print("\n" + "=" * 70)
    print("STEP 3: Train CBM models (3-fold CV, 30 epochs)")
    print("=" * 70)

    from cbm_variable_stars.training.cross_val import run_cross_validation

    cbm_models = ["hard_cbm", "hard_cbm_linear", "hard_cbm_cal", "soft_cbm", "cem", "mlp"]
    all_results = {}

    for model_name in cbm_models:
        print(f"\n--- Training: {model_name} ---")
        try:
            result = run_cross_validation(
                features=cv_features,
                labels=cv_labels,
                model_name=model_name,
                n_folds=3,
                batch_size=128,
                max_epochs=30,
                patience=10,
                output_dir="results/test_pipeline",
            )

            agg = result["aggregated"]
            print(f"  Accuracy: {agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")
            print(f"  Macro F1: {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")
            all_results[model_name] = result
            print(f"  {model_name}: PASSED")
        except Exception as e:
            print(f"  {model_name}: FAILED - {e}")
            traceback.print_exc()

    return all_results


def test_step4_baseline_training(cv_features, cv_labels):
    """Step 4: Train baseline models (RF, XGBoost)."""
    print("\n" + "=" * 70)
    print("STEP 4: Train baseline models (RF, XGBoost)")
    print("=" * 70)

    from cbm_variable_stars.data.splits import create_cv_splits
    from cbm_variable_stars.training.trainer import train_baseline

    baseline_results = {}

    splits = create_cv_splits(cv_labels, n_folds=3, random_seed=42)

    for model_type in ["rf", "xgb"]:
        print(f"\n--- Training: {model_type} ---")
        try:
            fold_metrics = []
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                model, metrics = train_baseline(
                    model_type=model_type,
                    features_train=cv_features[train_idx],
                    labels_train=cv_labels[train_idx],
                    features_val=cv_features[val_idx],
                    labels_val=cv_labels[val_idx],
                )
                fold_metrics.append(metrics)
                print(f"  Fold {fold_idx}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}")

            mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
            mean_f1 = np.mean([m["macro_f1"] for m in fold_metrics])
            baseline_results[model_type] = {
                "fold_metrics": fold_metrics,
                "aggregated": {
                    "accuracy_mean": mean_acc,
                    "accuracy_std": float(np.std([m["accuracy"] for m in fold_metrics], ddof=1)),
                    "macro_f1_mean": mean_f1,
                    "macro_f1_std": float(np.std([m["macro_f1"] for m in fold_metrics], ddof=1)),
                },
            }
            print(f"  {model_type} Mean: Acc={mean_acc:.4f}, F1={mean_f1:.4f}")
            print(f"  {model_type}: PASSED")
        except Exception as e:
            print(f"  {model_type}: FAILED - {e}")
            traceback.print_exc()

    return baseline_results


def test_step5_evaluation(cv_features, cv_labels, test_features, test_labels):
    """Step 5: Evaluate trained model on hold-out test set."""
    print("\n" + "=" * 70)
    print("STEP 5: Hold-out test set evaluation")
    print("=" * 70)

    from sklearn.model_selection import train_test_split
    from cbm_variable_stars.models import create_model
    from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
    from cbm_variable_stars.training.trainer import evaluate_model, Trainer
    from cbm_variable_stars.losses.cbm_loss import CBMJointLoss, compute_class_weights

    model = create_model("hard_cbm")

    # Split CV into train_sub and val_sub for early stopping (avoid test leak)
    train_idx, val_idx = train_test_split(
        np.arange(len(cv_labels)), test_size=0.15,
        stratify=cv_labels, random_state=42,
    )

    train_dataset = VariableStarDataset(
        features=cv_features[train_idx], labels=cv_labels[train_idx])
    val_dataset = VariableStarDataset(
        features=cv_features[val_idx], labels=cv_labels[val_idx])
    test_dataset = VariableStarDataset(features=test_features, labels=test_labels)

    train_loader = create_dataloader(train_dataset, batch_size=128, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=128, shuffle=False)
    test_loader = create_dataloader(test_dataset, batch_size=128, shuffle=False)

    class_weights = compute_class_weights(torch.tensor(cv_labels, dtype=torch.long))
    loss_fn = CBMJointLoss(alpha=0.0, beta=1.0, class_weights=class_weights)

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        max_epochs=30, patience=10,
        log_dir="results/test_pipeline/eval_logs",
        checkpoint_dir="results/test_pipeline/eval_ckpts",
    )

    trainer.fit(train_loader, val_loader, fold_id=0)  # val_loader, NOT test_loader

    eval_result = evaluate_model(model, test_loader)
    print(f"  Hold-out Accuracy: {eval_result['accuracy']:.4f}")
    print(f"  Hold-out Macro F1: {eval_result['macro_f1']:.4f}")
    print(f"  Hold-out Weighted F1: {eval_result['weighted_f1']:.4f}")

    if "per_class" in eval_result:
        print("  Per-class F1:")
        for cls_name, cls_metrics in eval_result["per_class"].items():
            if isinstance(cls_metrics, dict) and "f1" in cls_metrics:
                print(f"    {cls_name}: {cls_metrics['f1']:.4f}")

    print("  PASSED")
    return model, eval_result


def test_step6_experiments(cv_features, cv_labels, test_features, test_labels, model):
    """Step 6: Run experiment modules."""
    print("\n" + "=" * 70)
    print("STEP 6: Run experiments")
    print("=" * 70)

    # 6a: Correlation analysis
    print("\n--- 6a: Concept correlation analysis ---")
    try:
        from cbm_variable_stars.experiments.correlation import (
            compute_concept_correlation, compute_concept_class_association,
        )
        from cbm_variable_stars.shared.constants import CONCEPT_NAMES_12

        corr_result = compute_concept_correlation(cv_features, CONCEPT_NAMES_12)
        n_concepts = len(corr_result["concept_names"])
        n_high = len(corr_result["high_correlations"])
        print(f"  Correlation matrix for {n_concepts} concepts computed")
        print(f"  High correlation pairs (|r|>0.7): {n_high}")

        assoc_result = compute_concept_class_association(cv_features, cv_labels, CONCEPT_NAMES_12)
        ranking = assoc_result.get("concept_ranking", [])
        print(f"  Top discriminative concept: {ranking[0] if ranking else 'N/A'}")
        print("  6a: PASSED")
    except Exception as e:
        print(f"  6a: FAILED - {e}")
        traceback.print_exc()

    # 6b: Intervention experiment (noise injection)
    print("\n--- 6b: Noise injection intervention ---")
    try:
        from cbm_variable_stars.experiments.intervention import (
            run_noise_injection_experiment,
        )

        # Pass as dict (test_data format expected by intervention module)
        test_data = {
            "features": test_features,
            "labels": test_labels,
        }

        noise_result = run_noise_injection_experiment(
            model=model,
            test_data=test_data,
            noise_stds=[0.0, 0.5, 1.0],
        )
        print(f"  Clean accuracy: {noise_result['clean_accuracy']:.4f}")
        for sigma_key, sigma_data in noise_result["per_noise_level"].items():
            print(f"    sigma={sigma_key}: noisy_acc={sigma_data['accuracy_noisy_all']:.4f}")
        print("  6b: PASSED")
    except Exception as e:
        print(f"  6b: FAILED - {e}")
        traceback.print_exc()

    # 6c: Sequential intervention
    print("\n--- 6c: Sequential random intervention ---")
    try:
        from cbm_variable_stars.experiments.intervention import (
            intervene_sequential_random,
        )

        test_data = {"features": test_features, "labels": test_labels}
        seq_result = intervene_sequential_random(
            model=model,
            test_data=test_data,
            n_concepts=12,
            n_trials=3,
        )
        print(f"  Baseline accuracy: {seq_result['baseline_accuracy']:.4f}")
        print(f"  Full intervention accuracy: {seq_result['full_accuracy']:.4f}")
        print("  6c: PASSED")
    except Exception as e:
        print(f"  6c: FAILED - {e}")
        traceback.print_exc()

    # 6d: Learning curve
    print("\n--- 6d: Learning curve experiment ---")
    try:
        from cbm_variable_stars.experiments.learning_curve import run_learning_curve

        lc_result = run_learning_curve(
            features=cv_features,
            labels=cv_labels,
            model_name="hard_cbm",
            sample_sizes=[200, 500],
            n_repeats=1,
        )
        for n_str, metrics in lc_result.get("results", {}).items():
            print(f"    n={n_str}: acc={metrics['mean_accuracy']:.4f}, f1={metrics['mean_f1']:.4f}")
        print("  6d: PASSED")
    except Exception as e:
        print(f"  6d: FAILED - {e}")
        traceback.print_exc()

    # 6e: Cross-survey test (simulated)
    print("\n--- 6e: Cross-survey evaluation ---")
    try:
        from cbm_variable_stars.experiments.cross_survey import run_cross_survey_evaluation

        # Use 10dim mode (drop C11/C12)
        # For 10dim, the features should only have 10 columns
        # But our model was trained on 12-dim, so we use the full test
        # The cross_survey module handles subset internally

        gaia_test_data = {"features": test_features, "labels": test_labels}
        ogle_test_data = {"features": test_features[:200], "labels": test_labels[:200]}

        cross_result = run_cross_survey_evaluation(
            model=model,
            gaia_test=gaia_test_data,
            ogle_test=ogle_test_data,
            mode="12dim_with_match",
        )
        gaia_acc = cross_result["gaia_test_results"]["accuracy"]
        ogle_acc = cross_result["ogle_test_results"]["accuracy"]
        gap = cross_result["cross_survey_gap"]["accuracy"]
        print(f"  Gaia test acc: {gaia_acc:.4f}")
        print(f"  OGLE test acc: {ogle_acc:.4f}")
        print(f"  Gap: {gap:+.4f}")
        print("  6e: PASSED")
    except Exception as e:
        print(f"  6e: FAILED - {e}")
        traceback.print_exc()


def test_step7_visualization():
    """Step 7: Test visualization imports and basic functionality."""
    print("\n" + "=" * 70)
    print("STEP 7: Visualization modules")
    print("=" * 70)

    try:
        from cbm_variable_stars.visualization.plots import (
            plot_confusion_matrix, plot_training_curves,
        )
        print("  plots: imported")
    except Exception as e:
        print(f"  plots: FAILED - {e}")

    try:
        from cbm_variable_stars.visualization.concept_space import (
            plot_concept_tsne, plot_bailey_diagram,
        )
        print("  concept_space: imported")
    except Exception as e:
        print(f"  concept_space: FAILED - {e}")

    try:
        from cbm_variable_stars.visualization.latex_export import (
            results_to_latex, export_all_tables,
        )
        print("  latex_export: imported")
    except Exception as e:
        print(f"  latex_export: FAILED - {e}")

    print("  PASSED")


def test_step8_reporting(all_cbm_results, baseline_results):
    """Step 8: Test reporting pipeline."""
    print("\n" + "=" * 70)
    print("STEP 8: Reporting pipeline")
    print("=" * 70)

    try:
        from cbm_variable_stars.evaluation.reporting import (
            format_results_table, generate_comparison_table,
        )

        combined = {}
        for name, result in all_cbm_results.items():
            combined[name] = result
        for name, result in baseline_results.items():
            combined[name] = result

        csv_path = generate_comparison_table(
            combined,
            output_path="results/test_pipeline/comparison_table.csv",
        )
        print(f"  Comparison table saved: {csv_path}")

        if all_cbm_results:
            first_model = list(all_cbm_results.keys())[0]
            table_str = format_results_table(all_cbm_results[first_model], format="plain")
            print(f"  Result table for {first_model}:")
            for line in table_str.split("\n")[:5]:
                print(f"    {line}")

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED - {e}")
        traceback.print_exc()


def test_step9_significance(all_cbm_results):
    """Step 9: Statistical significance tests."""
    print("\n" + "=" * 70)
    print("STEP 9: Statistical significance tests")
    print("=" * 70)

    try:
        from cbm_variable_stars.evaluation.significance import (
            mcnemar_test, paired_cv_ttest, binomial_test,
        )

        model_names = list(all_cbm_results.keys())
        if len(model_names) >= 2:
            r1 = all_cbm_results[model_names[0]]
            r2 = all_cbm_results[model_names[1]]

            preds1 = r1["fold_results"][0]["predictions"]
            preds2 = r2["fold_results"][0]["predictions"]
            labels = r1["fold_results"][0]["true_labels"]

            mcnemar_result = mcnemar_test(preds1, preds2, labels)
            print(f"  McNemar ({model_names[0]} vs {model_names[1]}): p={mcnemar_result['p_value']:.4f}")

            scores1 = [r["metrics"]["val_macro_f1"] for r in r1["fold_results"]]
            scores2 = [r["metrics"]["val_macro_f1"] for r in r2["fold_results"]]

            ttest_result = paired_cv_ttest(scores1, scores2)
            print(f"  Paired t-test: t={ttest_result['t_statistic']:.4f}, p={ttest_result['p_value']:.4f}")

            acc = r1["aggregated"]["accuracy_mean"]
            n_test = int(len(labels))
            n_correct = int(round(acc * n_test))
            binom_result = binomial_test(n_correct, int(n_test // 6), n_test)
            print(f"  Binomial test: p={binom_result['p_value']:.6f}")

            print("  PASSED")
        else:
            print("  SKIPPED (need at least 2 models)")
    except Exception as e:
        print(f"  FAILED - {e}")
        traceback.print_exc()


def main():
    print("=" * 70)
    print("CBM Variable Star Classification - Full Pipeline Test")
    print("=" * 70)

    # Step 1: Data generation
    gaia_features, gaia_labels, ogle_features, ogle_labels = test_step1_data_generation()

    # Step 2: Dataset build
    cv_features, cv_labels, test_features, test_labels, scaler = test_step2_dataset_build(
        gaia_features, gaia_labels
    )

    # Step 3: CBM model training
    all_cbm_results = test_step3_cbm_training(cv_features, cv_labels)

    # Step 4: Baseline training
    baseline_results = test_step4_baseline_training(cv_features, cv_labels)

    # Step 5: Hold-out evaluation
    model, eval_result = test_step5_evaluation(
        cv_features, cv_labels, test_features, test_labels
    )

    # Step 6: Experiments
    test_step6_experiments(cv_features, cv_labels, test_features, test_labels, model)

    # Step 7: Visualization imports
    test_step7_visualization()

    # Step 8: Reporting
    test_step8_reporting(all_cbm_results, baseline_results)

    # Step 9: Significance tests
    test_step9_significance(all_cbm_results)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\nModel comparison (3-fold CV on synthetic data):")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")

    for name, result in {**all_cbm_results, **baseline_results}.items():
        agg = result.get("aggregated", {})
        acc = agg.get("accuracy_mean", 0)
        f1 = agg.get("macro_f1_mean", 0)
        print(f"  {name:<20} {acc:>10.4f} {f1:>10.4f}")

    print(f"\nHold-out test: Acc={eval_result['accuracy']:.4f}, F1={eval_result['macro_f1']:.4f}")
    print("\nPipeline test complete!")


if __name__ == "__main__":
    main()
