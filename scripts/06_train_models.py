"""
Script 06: Train all models with 5-fold cross-validation.

This is the main training script for the CBM Variable Star Classification project.
It trains all models defined in the MODEL_REGISTRY using 5-fold stratified CV,
evaluates on the in-domain hold-out test set, and saves all results.

Usage:
    python scripts/06_train_models.py --data_path path/to/features_gaia.parquet
    python scripts/06_train_models.py --data_path data/features_gaia.parquet \\
        --ogle_data_path data/features_ogle.parquet \\
        --models hard_cbm hard_cbm_linear hard_cbm_cal mlp rf xgb \\
        --output_dir results/ \\
        --seed 42

Output directory structure:
    results/
    ├── hard_cbm/
    │   ├── cv_results.json
    │   ├── logs/training_log_fold{i}.csv
    │   └── checkpoints/best_model_fold{i}.pt
    ├── hard_cbm_linear/
    ├── hard_cbm_cal/
    ├── soft_cbm/
    ├── cem/
    ├── mlp/
    ├── rf/
    ├── xgb/
    ├── comparison_table.csv
    ├── comparison_table.tex
    ├── domain_comparison.json
    └── significance_tests.json
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure the package is importable when run as a script
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES, CONCEPT_NAMES_12, CLASS_NAMES, RANDOM_SEED, NUM_CONCEPTS, NUM_CLASSES,
)
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.shared.config import load_config
from cbm_variable_stars.data.splits import create_full_split
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
from cbm_variable_stars.training.cross_val import run_cross_validation
from cbm_variable_stars.training.trainer import evaluate_model, evaluate_baseline
from cbm_variable_stars.baselines.random_forest import train_random_forest
from cbm_variable_stars.baselines.xgboost_model import train_xgboost
from cbm_variable_stars.evaluation.reporting import (
    generate_comparison_table,
    save_domain_comparison,
    save_results,
)
from cbm_variable_stars.evaluation.significance import paired_cv_ttest, mcnemar_test


# Neural-network model names (use run_cross_validation)
NN_MODELS = {"hard_cbm", "hard_cbm_linear", "hard_cbm_cal", "soft_cbm", "cem", "mlp"}

# Baseline model names (use train_random_forest / train_xgboost)
BASELINE_MODELS = {"rf", "xgb"}

# Default model configs for ablation
MODEL_CONFIGS = {
    "hard_cbm":        {"hidden_dims": [64, 32], "dropout_rate": 0.3},
    "hard_cbm_linear": {},
    "hard_cbm_cal":    {"predictor_hidden_dims": [64, 32], "dropout_rate": 0.3},
    "soft_cbm":        {"concept_embed_dim": 4, "hidden_dims": [64, 32]},
    "cem":             {"concept_embed_dim": 8, "hidden_dims": [64, 32]},
    "mlp":             {"hidden_dims": [128, 64], "dropout_rate": 0.3},
}


def load_features(data_path: str) -> tuple:
    """
    Load standardized features from parquet file.

    Args:
        data_path: Path to parquet file with columns matching CONCEPT_NAMES + 'label'

    Returns:
        Tuple of (features ndarray, labels ndarray)
    """
    df = pd.read_parquet(data_path)

    # Verify required columns
    missing = [c for c in CONCEPT_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing concept columns in {data_path}: {missing}\n"
            f"Expected: {CONCEPT_NAMES}"
        )

    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {data_path}")

    features = df[CONCEPT_NAMES].values.astype(np.float32)
    labels_raw = df["label"].values

    # Encode string labels to integers if needed
    if labels_raw.dtype.kind in ("U", "S", "O"):
        from cbm_variable_stars.shared.constants import LABEL_MAP
        labels = np.array([LABEL_MAP[str(l)] for l in labels_raw], dtype=np.int64)
    else:
        labels = labels_raw.astype(np.int64)

    print(f"  Loaded {len(features)} samples, {features.shape[1]} features")
    print(f"  Class distribution:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        n = int(np.sum(labels == class_idx))
        print(f"    {class_name}: {n} ({100 * n / len(labels):.1f}%)")

    return features, labels


def train_nn_models(
    models: list,
    cv_features: np.ndarray,
    cv_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Train neural network models with 5-fold CV."""
    all_results = {}

    for model_name in models:
        if model_name not in NN_MODELS:
            continue

        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")

        model_kwargs = MODEL_CONFIGS.get(model_name, {})

        result = run_cross_validation(
            features=cv_features,
            labels=cv_labels,
            model_name=model_name,
            model_kwargs=model_kwargs,
            n_folds=5,
            batch_size=256,
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=args.max_epochs,
            patience=args.patience,
            random_seed=args.seed,
            output_dir=args.output_dir,
            device=args.device,
        )

        all_results[model_name] = result

        # Print summary
        agg = result["aggregated"]
        print(
            f"\n{model_name} CV Summary: "
            f"Acc={agg['accuracy_mean']:.4f}+/-{agg['accuracy_std']:.4f} | "
            f"F1={agg['macro_f1_mean']:.4f}+/-{agg['macro_f1_std']:.4f}"
        )

    return all_results


def train_baseline_models(
    models: list,
    cv_features: np.ndarray,
    cv_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """Train baseline models (RF, XGBoost) with 5-fold CV."""
    all_results = {}

    if "rf" in models:
        print(f"\n{'='*70}")
        print("Training: Random Forest")
        print(f"{'='*70}")
        rf_result = train_random_forest(
            features=cv_features,
            labels=cv_labels,
            output_dir=f"{args.output_dir}/rf",
            random_seed=args.seed,
        )
        all_results["rf"] = rf_result

    if "xgb" in models:
        print(f"\n{'='*70}")
        print("Training: XGBoost")
        print(f"{'='*70}")
        try:
            xgb_result = train_xgboost(
                features=cv_features,
                labels=cv_labels,
                output_dir=f"{args.output_dir}/xgb",
                random_seed=args.seed,
                compute_shap=True,
            )
            all_results["xgb"] = xgb_result
        except ImportError:
            print("  Warning: xgboost not installed; skipping XGBoost training.")

    return all_results


def evaluate_on_test_set(
    all_nn_results: dict,
    all_baseline_results: dict,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    """
    Evaluate trained models on in-domain hold-out test set.

    For NN models: loads best checkpoint from last fold and evaluates.
    For baseline models: evaluates using fold results (approximate).
    """
    print(f"\n{'='*70}")
    print("Evaluating on in-domain hold-out test set")
    print(f"{'='*70}")

    test_metrics = {}

    # Evaluate NN models (use fold with best validation macro F1)
    for model_name, result in all_nn_results.items():
        import torch
        from cbm_variable_stars.models import create_model

        model_kwargs = MODEL_CONFIGS.get(model_name, {})
        model_dir = Path(args.output_dir) / model_name / "checkpoints"
        n_folds = 5

        # Find the best fold based on validation macro F1
        best_fold = 0
        best_f1 = 0.0
        for fold_id in range(n_folds):
            ckpt_path = model_dir / f"best_model_fold{fold_id}.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
                fold_f1 = ckpt.get("val_macro_f1", 0.0)
                if fold_f1 > best_f1:
                    best_f1 = fold_f1
                    best_fold = fold_id

        print(f"  {model_name}: Using best fold {best_fold} (val_macro_f1={best_f1:.4f}) for test evaluation")
        ckpt_path = model_dir / f"best_model_fold{best_fold}.pt"

        if ckpt_path.exists():
            model = create_model(model_name, **model_kwargs)
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=args.device)
            model.load_state_dict(ckpt["model_state_dict"])

            test_dataset = VariableStarDataset(
                features=test_features,
                labels=test_labels,
            )
            test_loader = create_dataloader(test_dataset, batch_size=512, shuffle=False)

            metrics = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=args.device,
                class_names=CLASS_NAMES,
            )
            test_metrics[model_name] = metrics

            print(
                f"  {model_name}: "
                f"Acc={metrics['accuracy']:.4f} | "
                f"Macro F1={metrics['macro_f1']:.4f}"
            )
        else:
            print(f"  {model_name}: no checkpoint found in {model_dir}")

    return test_metrics


def run_significance_tests(
    all_nn_results: dict,
    args: argparse.Namespace,
) -> dict:
    """Run paired t-tests comparing CBM vs baselines across CV folds."""
    print(f"\n{'='*70}")
    print("Statistical significance tests")
    print(f"{'='*70}")

    sig_results = {}

    # Compare hard_cbm vs mlp
    if "hard_cbm" in all_nn_results and "mlp" in all_nn_results:
        cbm_f1s = [
            r["metrics"]["val_macro_f1"]
            for r in all_nn_results["hard_cbm"]["fold_results"]
        ]
        mlp_f1s = [
            r["metrics"]["val_macro_f1"]
            for r in all_nn_results["mlp"]["fold_results"]
        ]

        ttest = paired_cv_ttest(cbm_f1s, mlp_f1s, "HardCBM", "MLP")
        sig_results["hard_cbm_vs_mlp"] = ttest

        print(
            f"  HardCBM vs MLP: "
            f"mean_diff={ttest['mean_diff']:.4f} | "
            f"p={ttest['p_value']:.4f} | "
            f"significant={ttest['significant_at_005']}"
        )

    # Compare hard_cbm vs hard_cbm_linear
    if "hard_cbm" in all_nn_results and "hard_cbm_linear" in all_nn_results:
        cbm_f1s = [
            r["metrics"]["val_macro_f1"]
            for r in all_nn_results["hard_cbm"]["fold_results"]
        ]
        linear_f1s = [
            r["metrics"]["val_macro_f1"]
            for r in all_nn_results["hard_cbm_linear"]["fold_results"]
        ]

        ttest = paired_cv_ttest(cbm_f1s, linear_f1s, "HardCBM", "HardCBM_Linear")
        sig_results["hard_cbm_vs_linear"] = ttest

        print(
            f"  HardCBM vs HardCBM_Linear: "
            f"mean_diff={ttest['mean_diff']:.4f} | "
            f"p={ttest['p_value']:.4f} | "
            f"significant={ttest['significant_at_005']}"
        )

    # Save significance tests
    sig_path = Path(args.output_dir) / "significance_tests.json"
    with open(sig_path, "w") as f:
        json.dump(sig_results, f, indent=2, default=str)
    print(f"\n  Significance tests saved to: {sig_path}")

    return sig_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all CBM and baseline models for variable star classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to standardized Gaia features parquet file.",
    )
    parser.add_argument(
        "--ogle_data_path",
        type=str,
        default=None,
        help="Path to OGLE out-of-domain test features parquet file (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Root directory for saving all results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["hard_cbm", "hard_cbm_linear", "hard_cbm_cal", "mlp", "rf"],
        help=(
            "Models to train. Choose from: "
            "hard_cbm, hard_cbm_linear, hard_cbm_cal, soft_cbm, cem, mlp, rf, xgb"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Global random seed.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Maximum training epochs per fold (for NN models).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs).",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip in-domain test set evaluation after training.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config file path.",
    )

    args = parser.parse_args()

    # Load optional config file (command-line arguments take priority)
    cfg = load_config(args.config)
    training_cfg = cfg.get("training", {}) if hasattr(cfg, "get") else getattr(cfg, "training", {})
    # Use config values as defaults only when the CLI argument was not explicitly set
    if args.patience == 15 and training_cfg:
        args.patience = int(training_cfg.get("patience", 15)) if hasattr(training_cfg, "get") else int(getattr(training_cfg, "patience", 15))

    # ===== Setup =====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCBM Variable Star Classification -- Training Script")
    print(f"  Data:       {args.data_path}")
    print(f"  Models:     {args.models}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Device:     {args.device}")
    print(f"  Seed:       {args.seed}")
    print(f"  Max epochs: {args.max_epochs}")

    # Set random seeds (includes CUDA and cudnn deterministic settings)
    set_global_seed(args.seed)

    # ===== Load data =====
    print(f"\n{'='*70}")
    print("Loading data")
    print(f"{'='*70}")

    # Prefer pre-built datasets from step 05 (05_build_dataset.py)
    processed_dir = Path(args.data_path).parent / "processed" if Path(args.data_path).is_file() else Path(args.data_path) / "processed"
    cv_pool_path = processed_dir / "cv_pool.parquet"
    test_path = processed_dir / "test_in_domain.parquet"

    if cv_pool_path.exists() and test_path.exists():
        print("  Loading pre-built datasets from step 05...")
        cv_pool = pd.read_parquet(cv_pool_path)
        test_data = pd.read_parquet(test_path)
        cv_features = cv_pool[CONCEPT_NAMES_12].values.astype(np.float32)
        cv_labels = cv_pool["label"].values.astype(np.int64)
        test_features = test_data[CONCEPT_NAMES_12].values.astype(np.float32)
        test_labels = test_data["label"].values.astype(np.int64)
    else:
        print("  Pre-built datasets not found, falling back to raw data loading...")
        features, labels = load_features(args.data_path)

        # ===== Domain split: 85% CV / 15% hold-out test =====
        split = create_full_split(labels, random_seed=args.seed)
        cv_features = features[split["cv_indices"]]
        cv_labels = labels[split["cv_indices"]]
        test_features = features[split["test_indices"]]
        test_labels = labels[split["test_indices"]]

    print(f"\n  CV subset:         {len(cv_features)} samples")
    print(f"  Hold-out test set: {len(test_features)} samples")

    # ===== Train NN models =====
    nn_models_to_train = [m for m in args.models if m in NN_MODELS]
    baseline_models_to_train = [m for m in args.models if m in BASELINE_MODELS]

    all_nn_results = {}
    all_baseline_results = {}

    if nn_models_to_train:
        all_nn_results = train_nn_models(
            nn_models_to_train, cv_features, cv_labels, args
        )

    # ===== Train baseline models =====
    if baseline_models_to_train:
        all_baseline_results = train_baseline_models(
            baseline_models_to_train, cv_features, cv_labels, args
        )

    # ===== Evaluate on hold-out test set =====
    if not args.skip_eval and all_nn_results:
        test_metrics = evaluate_on_test_set(
            all_nn_results, all_baseline_results,
            test_features, test_labels, args,
        )

        # Save test set results
        test_results_path = output_dir / "test_set_results.json"
        with open(test_results_path, "w") as f:
            json.dump(test_metrics, f, indent=2, default=str)
        print(f"\n  Test set results saved to: {test_results_path}")

    # ===== Evaluate on OGLE out-of-domain test set =====
    if args.ogle_data_path is not None:
        print(f"\n{'='*70}")
        print("Evaluating on OGLE out-of-domain test set")
        print(f"{'='*70}")
        try:
            ogle_features, ogle_labels = load_features(args.ogle_data_path)
            print(f"  OGLE samples: {len(ogle_features)}")

            ogle_metrics = {}
            import torch
            from cbm_variable_stars.models import create_model

            for model_name, result in all_nn_results.items():
                model_kwargs = MODEL_CONFIGS.get(model_name, {})
                ogle_model_dir = Path(args.output_dir) / model_name / "checkpoints"
                n_folds = 5

                # Find the best fold based on validation macro F1
                ogle_best_fold = 0
                ogle_best_f1 = 0.0
                for fold_id in range(n_folds):
                    fold_ckpt = ogle_model_dir / f"best_model_fold{fold_id}.pt"
                    if fold_ckpt.exists():
                        fold_data = torch.load(fold_ckpt, weights_only=False, map_location="cpu")
                        fold_f1 = fold_data.get("val_macro_f1", 0.0)
                        if fold_f1 > ogle_best_f1:
                            ogle_best_f1 = fold_f1
                            ogle_best_fold = fold_id

                ckpt_path = ogle_model_dir / f"best_model_fold{ogle_best_fold}.pt"
                if ckpt_path.exists():
                    model = create_model(model_name, **model_kwargs)
                    ckpt = torch.load(ckpt_path, weights_only=False, map_location=args.device)
                    model.load_state_dict(ckpt["model_state_dict"])

                    ogle_dataset = VariableStarDataset(
                        features=ogle_features, labels=ogle_labels
                    )
                    ogle_loader = create_dataloader(ogle_dataset, batch_size=512, shuffle=False)

                    metrics = evaluate_model(model, ogle_loader, args.device, CLASS_NAMES)
                    ogle_metrics[model_name] = metrics

                    print(
                        f"  {model_name} (OGLE, fold {ogle_best_fold}): "
                        f"Acc={metrics['accuracy']:.4f} | "
                        f"Macro F1={metrics['macro_f1']:.4f}"
                    )

            # Save OGLE results
            ogle_results_path = output_dir / "ogle_test_results.json"
            with open(ogle_results_path, "w") as f:
                json.dump(ogle_metrics, f, indent=2, default=str)

        except Exception as e:
            print(f"  Warning: Could not evaluate on OGLE data: {e}")

    # ===== Generate comparison table =====
    print(f"\n{'='*70}")
    print("Generating comparison table")
    print(f"{'='*70}")

    all_results_combined = {**all_nn_results, **all_baseline_results}

    if all_results_combined:
        table_path = generate_comparison_table(
            all_results_combined,
            output_path=str(output_dir / "comparison_table.csv"),
        )
        print(f"  Comparison table saved to: {table_path}")

    # ===== Statistical significance tests =====
    if len(all_nn_results) >= 2:
        run_significance_tests(all_nn_results, args)

    # ===== Save combined results =====
    combined_path = output_dir / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(
            {
                "nn_models": {
                    k: {
                        "aggregated": v["aggregated"],
                        "n_folds": len(v["fold_results"]),
                    }
                    for k, v in all_nn_results.items()
                },
                "baseline_models": {
                    k: {"aggregated": v["aggregated"]}
                    for k, v in all_baseline_results.items()
                },
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n{'='*70}")
    print("All experiments completed.")
    print(f"Results saved to: {args.output_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
