#!/usr/bin/env python3
"""
End-to-end CBM experiments for variable star classification.

Runs the full experimental pipeline:
1. 5-fold cross-validation of EndToEndHardCBM
2. Concept prediction quality analysis (per-concept R², Pearson r)
3. Intervention experiments (replace predicted concepts with ground truth)
4. Comparison with hand-crafted feature CBM (HardCBM)
5. Ablation: concept loss weight alpha

Usage:
    python scripts/run_e2e_experiments.py [--device cuda] [--alpha 1.0]
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from cbm_variable_stars.shared.constants import (
    CONCEPT_NAMES_12, CLASS_NAMES, NUM_CONCEPTS, RANDOM_SEED,
)
from cbm_variable_stars.models import create_model
from cbm_variable_stars.data.lightcurve_dataset import LightCurveDataset
from cbm_variable_stars.data.dataset import VariableStarDataset, create_dataloader
from cbm_variable_stars.data.splits import create_cv_splits
from cbm_variable_stars.losses.cbm_loss import CBMJointLoss, compute_class_weights
from cbm_variable_stars.training.trainer import Trainer
from cbm_variable_stars.training.cross_val import run_e2e_cross_validation


def load_raw_features(data_dir: str = "data/interim") -> tuple:
    """Load raw features from parquet."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(data_dir, "gaia_features_raw.parquet"))
    features = df[CONCEPT_NAMES_12].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)
    return features, labels


def run_cv_experiment(args):
    """Run 5-fold CV for EndToEndHardCBM."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: End-to-End CBM 5-Fold Cross-Validation")
    print("=" * 70)

    features, labels = load_raw_features()
    print(f"Data: {features.shape[0]} samples, {features.shape[1]} features, "
          f"{len(np.unique(labels))} classes")

    results = run_e2e_cross_validation(
        raw_features=features,
        labels=labels,
        model_kwargs={
            "n_bins": 100,
            "hidden_dims": [64, 32],
        },
        n_folds=5,
        batch_size=256,
        learning_rate=args.lr,
        weight_decay=1e-4,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        concept_loss_alpha=args.alpha,
        noise_level=args.noise_level,
        output_dir=args.output_dir,
    )
    return results


def run_concept_quality_analysis(args):
    """Analyze per-concept prediction quality (R², Pearson r)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Concept Prediction Quality Analysis")
    print("=" * 70)

    features, labels = load_raw_features()
    splits = create_cv_splits(labels, 5, RANDOM_SEED)

    all_concept_preds = []
    all_concept_gts = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_train_raw = features[train_idx]
        fold_val_raw = features[val_idx]
        fold_train_labels = labels[train_idx]
        fold_val_labels = labels[val_idx]

        # Per-fold standardization
        scaler = StandardScaler()
        train_gt = scaler.fit_transform(fold_train_raw)
        val_gt = scaler.transform(fold_val_raw)

        # Create datasets
        train_dataset = LightCurveDataset(
            fold_train_raw, fold_train_labels, train_gt,
            noise_level=args.noise_level, augment=True,
        )
        val_dataset = LightCurveDataset(
            fold_val_raw, fold_val_labels, val_gt,
            noise_level=args.noise_level, augment=False,
        )
        train_loader = create_dataloader(train_dataset, batch_size=256, shuffle=True, device=args.device)
        val_loader = create_dataloader(val_dataset, batch_size=256, shuffle=False, device=args.device)

        # Train model
        model = create_model("e2e_hard_cbm", n_bins=100, hidden_dims=[64, 32])
        class_weights = compute_class_weights(torch.tensor(fold_train_labels, dtype=torch.long))
        loss_fn = CBMJointLoss(alpha=args.alpha, beta=1.0, class_weights=class_weights, use_concept_loss=True)
        trainer = Trainer(
            model=model, loss_fn=loss_fn, learning_rate=args.lr,
            weight_decay=1e-4, max_epochs=args.max_epochs, patience=args.patience,
            device=args.device,
            log_dir=str(Path(args.output_dir) / "e2e_hard_cbm" / "logs"),
            checkpoint_dir=str(Path(args.output_dir) / "e2e_hard_cbm" / "checkpoints"),
        )
        trainer.fit(train_loader, val_loader, fold_id=fold_idx)

        # Collect concept predictions
        model.eval()
        fold_preds, fold_gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(args.device)
                gt = batch["concept_gt"]
                output = model(x)
                fold_preds.append(output["concepts"].cpu().numpy())
                fold_gts.append(gt.numpy())

        all_concept_preds.append(np.concatenate(fold_preds, axis=0))
        all_concept_gts.append(np.concatenate(fold_gts, axis=0))

    # Aggregate across folds
    concept_preds = np.concatenate(all_concept_preds, axis=0)
    concept_gts = np.concatenate(all_concept_gts, axis=0)

    print(f"\nPer-concept prediction quality (N={len(concept_preds)}):")
    print(f"{'Concept':20s} {'R2':>8s} {'Pearson r':>10s} {'MSE':>8s}")
    print("-" * 50)

    concept_results = {}
    for i, name in enumerate(CONCEPT_NAMES_12):
        pred = concept_preds[:, i]
        gt = concept_gts[:, i]
        ss_res = np.sum((pred - gt) ** 2)
        ss_tot = np.sum((gt - gt.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        r, p_val = pearsonr(pred, gt)
        mse = np.mean((pred - gt) ** 2)

        concept_results[name] = {"r2": float(r2), "pearson_r": float(r), "mse": float(mse)}
        print(f"{name:20s} {r2:8.4f} {r:10.4f} {mse:8.4f}")

    # Save results
    output_path = Path(args.output_dir) / "e2e_hard_cbm" / "concept_quality.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(concept_results, f, indent=2)

    return concept_results


def run_intervention_experiment(args):
    """
    Intervention experiment using sequential training:
    Phase 1: Train encoder + label predictor jointly (concept + cls loss)
    Phase 2: Freeze encoder, retrain label predictor on GT concepts
    Phase 3: Evaluate intervention (replace predicted with GT concepts)

    This follows the standard CBM protocol (Koh et al. 2020): the label
    predictor must be trained on GT concepts for intervention to work,
    otherwise distribution shift between predicted/GT concepts breaks it.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Concept Intervention Analysis (Sequential Training)")
    print("=" * 70)

    features, labels = load_raw_features()
    splits = create_cv_splits(labels, 5, RANDOM_SEED)
    from sklearn.metrics import accuracy_score, f1_score

    # Use first fold for intervention analysis
    train_idx, val_idx = splits[0]
    fold_train_raw = features[train_idx]
    fold_val_raw = features[val_idx]
    fold_train_labels = labels[train_idx]
    fold_val_labels = labels[val_idx]

    scaler = StandardScaler()
    train_gt = scaler.fit_transform(fold_train_raw)
    val_gt = scaler.transform(fold_val_raw)

    train_dataset = LightCurveDataset(
        fold_train_raw, fold_train_labels, train_gt,
        noise_level=args.noise_level, augment=True,
    )
    val_dataset = LightCurveDataset(
        fold_val_raw, fold_val_labels, val_gt,
        noise_level=args.noise_level, augment=False,
    )
    train_loader = create_dataloader(train_dataset, batch_size=256, shuffle=True, device=args.device)
    val_loader = create_dataloader(val_dataset, batch_size=256, shuffle=False, device=args.device)

    # === Phase 1: Joint training of encoder + predictor ===
    print("\n--- Phase 1: Joint training ---")
    model = create_model("e2e_hard_cbm", n_bins=100, hidden_dims=[64, 32])
    class_weights = compute_class_weights(torch.tensor(fold_train_labels, dtype=torch.long))
    loss_fn = CBMJointLoss(alpha=args.alpha, beta=1.0, class_weights=class_weights, use_concept_loss=True)
    trainer = Trainer(
        model=model, loss_fn=loss_fn, learning_rate=args.lr,
        weight_decay=1e-4, max_epochs=args.max_epochs, patience=args.patience,
        device=args.device,
        log_dir=str(Path(args.output_dir) / "e2e_hard_cbm" / "logs"),
        checkpoint_dir=str(Path(args.output_dir) / "e2e_hard_cbm" / "checkpoints"),
    )
    trainer.fit(train_loader, val_loader, fold_id=99)

    # === Phase 2: Retrain label predictor on GT concepts ===
    print("\n--- Phase 2: Retraining label predictor on GT concepts ---")

    # Freeze encoder
    for param in model.concept_encoder.parameters():
        param.requires_grad = False

    # Re-initialize label predictor
    model._init_predictor_weights()

    # Create GT-concept datasets (features = GT concepts, not light curves)
    from cbm_variable_stars.data.dataset import VariableStarDataset
    gt_train_ds = VariableStarDataset(
        features=train_gt.astype(np.float32),
        labels=fold_train_labels,
    )
    gt_val_ds = VariableStarDataset(
        features=val_gt.astype(np.float32),
        labels=fold_val_labels,
    )
    gt_train_loader = create_dataloader(gt_train_ds, batch_size=256, shuffle=True, device=args.device)
    gt_val_loader = create_dataloader(gt_val_ds, batch_size=256, shuffle=False, device=args.device)

    # Train label predictor as a simple HardCBM (concepts → labels)
    from cbm_variable_stars.models.cbm_hard import HardCBM
    predictor_model = HardCBM(num_concepts=12, num_classes=6, hidden_dims=[64, 32])
    predictor_loss = CBMJointLoss(alpha=0.0, beta=1.0, class_weights=class_weights, use_concept_loss=False)
    predictor_trainer = Trainer(
        model=predictor_model, loss_fn=predictor_loss, learning_rate=args.lr,
        weight_decay=1e-4, max_epochs=args.max_epochs, patience=args.patience,
        device=args.device,
        log_dir=str(Path(args.output_dir) / "e2e_hard_cbm" / "logs"),
        checkpoint_dir=str(Path(args.output_dir) / "e2e_hard_cbm" / "checkpoints"),
    )
    predictor_trainer.fit(gt_train_loader, gt_val_loader, fold_id=98)

    # Copy trained label predictor weights back to E2E model
    model.label_predictor.load_state_dict(predictor_model.label_predictor.state_dict())
    model.to(args.device)
    model.eval()

    # === Phase 3: Intervention evaluation ===
    print("\n--- Phase 3: Intervention evaluation ---")

    def evaluate_with_intervention(n_concepts_to_intervene):
        """Evaluate with top-N concept interventions."""
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(args.device)
                gt = batch["concept_gt"].to(args.device)
                true_labels = batch["label"]

                if n_concepts_to_intervene > 0:
                    override = torch.full_like(gt, float("nan"))
                    override[:, :n_concepts_to_intervene] = gt[:, :n_concepts_to_intervene]
                    output = model(x, concept_override=override)
                else:
                    output = model(x)

                preds = output["logits"].argmax(1).cpu()
                all_preds.extend(preds.tolist())
                all_labels_list.extend(true_labels.tolist())

        acc = accuracy_score(all_labels_list, all_preds)
        f1 = f1_score(all_labels_list, all_preds, average="macro", zero_division=0)
        return acc, f1

    intervention_results = {}
    n_concepts_list = [0, 1, 2, 3, 4, 6, 8, 12]

    print(f"\n{'N concepts':>12s} {'Accuracy':>10s} {'Macro F1':>10s}")
    print("-" * 35)

    for n in n_concepts_list:
        acc, f1 = evaluate_with_intervention(n)
        intervention_results[str(n)] = {"accuracy": float(acc), "macro_f1": float(f1)}
        print(f"{n:12d} {acc:10.4f} {f1:10.4f}")

    # Recovery rate
    baseline_acc = intervention_results["0"]["accuracy"]
    full_intervention_acc = intervention_results["12"]["accuracy"]
    if full_intervention_acc > baseline_acc:
        recovery = (full_intervention_acc - baseline_acc) / max(1.0 - baseline_acc, 1e-10)
    else:
        recovery = 0.0

    intervention_results["recovery_rate"] = float(recovery)
    print(f"\nBaseline accuracy (encoder concepts):   {baseline_acc:.4f}")
    print(f"Full intervention (all GT concepts):    {full_intervention_acc:.4f}")
    print(f"Recovery rate:                          {recovery:.4f}")

    output_path = Path(args.output_dir) / "e2e_hard_cbm" / "intervention_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(intervention_results, f, indent=2)

    return intervention_results


def run_comparison_experiment(args):
    """Compare EndToEndHardCBM vs hand-crafted HardCBM."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: End-to-End vs Hand-Crafted Feature CBM")
    print("=" * 70)

    from cbm_variable_stars.training.cross_val import run_cross_validation

    features, labels = load_raw_features()

    # Run hand-crafted CBM for comparison
    print("\n--- Hand-crafted HardCBM (Plan A) ---")
    handcrafted_results = run_cross_validation(
        features=features,
        labels=labels,
        model_name="hard_cbm",
        batch_size=256,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Load or run E2E results
    e2e_results_path = Path(args.output_dir) / "e2e_hard_cbm" / "cv_results.json"
    if e2e_results_path.exists():
        with open(e2e_results_path) as f:
            e2e_saved = json.load(f)
        e2e_agg = e2e_saved["aggregated"]
    else:
        e2e_full = run_cv_experiment(args)
        e2e_agg = e2e_full["aggregated"]

    hc_agg = handcrafted_results["aggregated"]

    print(f"\n{'Model':25s} {'Accuracy':>15s} {'Macro F1':>15s}")
    print("-" * 60)
    print(f"{'HardCBM (x=c)':25s} "
          f"{hc_agg['accuracy_mean']:.4f}±{hc_agg['accuracy_std']:.4f}  "
          f"{hc_agg['macro_f1_mean']:.4f}±{hc_agg['macro_f1_std']:.4f}")
    print(f"{'EndToEndHardCBM':25s} "
          f"{e2e_agg['accuracy_mean']:.4f}±{e2e_agg['accuracy_std']:.4f}  "
          f"{e2e_agg['macro_f1_mean']:.4f}±{e2e_agg['macro_f1_std']:.4f}")

    comparison = {
        "hard_cbm": hc_agg,
        "e2e_hard_cbm": e2e_agg,
    }
    output_path = Path(args.output_dir) / "e2e_hard_cbm" / "comparison.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def main():
    parser = argparse.ArgumentParser(description="End-to-end CBM experiments")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--alpha", type=float, default=1.0, help="Concept loss weight")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--noise-level", type=float, default=0.03, help="Light curve noise level")
    parser.add_argument("--output-dir", type=str, default="results/e2e", help="Output directory")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "cv", "concept_quality", "intervention", "comparison"],
                        help="Which experiment to run")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Concept loss alpha: {args.alpha}")
    print(f"Noise level: {args.noise_level}")

    os.chdir(project_root)

    if args.experiment in ("all", "cv"):
        run_cv_experiment(args)

    if args.experiment in ("all", "concept_quality"):
        run_concept_quality_analysis(args)

    if args.experiment in ("all", "intervention"):
        run_intervention_experiment(args)

    if args.experiment in ("all", "comparison"):
        run_comparison_experiment(args)

    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
