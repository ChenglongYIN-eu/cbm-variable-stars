"""
CBM Variable Star Classification -- Concept Intervention Experiments

S5 fix: Noise injection intervention experiments to meaningfully evaluate
concept importance in Plan A (input = concepts), where naive intervention
(c_pred = c_true) is trivially zero.

Intervention mechanics use concept_override with NaN masking:
    override = torch.full_like(features, float('nan'))
    override[:, concept_idx] = true_concept_values[:, concept_idx]
    output = model(features, concept_override=override)

Experiments:
    - Sequential random intervention: randomly select concepts to intervene
    - Sequential greedy intervention: greedily pick the best concept each step
    - Noise injection: inject Gaussian noise into concept values and measure
      recovery via targeted intervention
    - Case studies: inspect misclassified examples and trace intervention effects
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_NAMES_12,
    RANDOM_SEED,
)
from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.shared.reproducibility import set_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(
    model_name: str,
    n_concepts: int,
    concept_names: List[str],
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "n_concepts": n_concepts,
        "concept_names": concept_names,
    }


def _accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


def _get_tensors(
    test_data: Dict[str, Any],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features and labels tensors from a test_data dict or tuple.

    test_data may be:
        - dict with keys "features" and "labels"
        - tuple/list (features_tensor, labels_tensor)
    """
    if isinstance(test_data, dict):
        features = test_data["features"]
        labels = test_data["labels"]
    else:
        features, labels = test_data

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(np.asarray(features), dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(np.asarray(labels), dtype=torch.long)

    return features.to(device), labels.to(device)


# ---------------------------------------------------------------------------
# Sequential random intervention
# ---------------------------------------------------------------------------

def intervene_sequential_random(
    model: nn.Module,
    test_data: Any,
    n_concepts: int,
    n_trials: int = 5,
    device: str = "cpu",
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Randomly select concepts to intervene (replace with true values) and
    measure accuracy at each intervention step.

    The experiment is repeated `n_trials` times with different random orderings
    to estimate the mean and variance of accuracy across orderings.

    Parameters
    ----------
    model : nn.Module
        Trained CBM model with concept_override support.
    test_data : dict or (features, labels) tuple
        Test dataset. For noise injection: features should already contain noise.
        True concept values (ground truth) must be available as test_data["true_features"]
        or, if test_data is a dict without "true_features", as test_data["features"].
    n_concepts : int
        Total number of concepts (e.g. 12).
    n_trials : int
        Number of random orderings to average over.
    device : str
        Torch device.
    random_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        {
          "n_concepts": int,
          "n_trials": int,
          "mean_accuracies": list of float (length n_concepts+1),
          "std_accuracies": list of float (length n_concepts+1),
          "baseline_accuracy": float,  -- 0 concepts intervened
          "full_accuracy": float,       -- all concepts intervened
          "per_trial_accuracies": list of list of float,
        }
    """
    set_global_seed(random_seed)
    model = model.to(device)
    model.eval()

    features, labels = _get_tensors(test_data, device)

    # If noisy features are provided separately, true features are the clean ones
    if isinstance(test_data, dict) and "true_features" in test_data:
        true_features = test_data["true_features"]
        if not isinstance(true_features, torch.Tensor):
            true_features = torch.tensor(np.asarray(true_features), dtype=torch.float32)
        true_features = true_features.to(device)
    else:
        # No noise scenario: true = input (intervention trivially restores full accuracy)
        true_features = features

    rng = random.Random(random_seed)

    per_trial_accuracies: List[List[float]] = []

    with torch.no_grad():
        # Baseline: no intervention
        out_base = model(features)
        acc_base = _accuracy(out_base["logits"].argmax(dim=1), labels)

        for trial in range(n_trials):
            # Random concept ordering
            order = list(range(n_concepts))
            rng.shuffle(order)

            trial_accs = [acc_base]  # step 0: no intervention
            override = torch.full_like(features, float("nan"))

            for concept_idx in order:
                override[:, concept_idx] = true_features[:, concept_idx]
                out = model(features, concept_override=override)
                acc = _accuracy(out["logits"].argmax(dim=1), labels)
                trial_accs.append(acc)

            per_trial_accuracies.append(trial_accs)

    mean_accs = np.mean(per_trial_accuracies, axis=0).tolist()
    std_accs = np.std(per_trial_accuracies, axis=0).tolist()

    result = {
        "n_concepts": n_concepts,
        "n_trials": n_trials,
        "mean_accuracies": mean_accs,
        "std_accuracies": std_accs,
        "baseline_accuracy": acc_base,
        "full_accuracy": mean_accs[-1],
        "per_trial_accuracies": per_trial_accuracies,
        "n_steps": list(range(n_concepts + 1)),
    }

    logger.info(
        f"Sequential random intervention: "
        f"baseline={acc_base:.4f} -> "
        f"all-intervened={mean_accs[-1]:.4f} "
        f"(avg over {n_trials} trials)"
    )
    return result


# ---------------------------------------------------------------------------
# Sequential greedy intervention
# ---------------------------------------------------------------------------

def intervene_sequential_greedy(
    model: nn.Module,
    test_data: Any,
    n_concepts: int,
    device: str = "cpu",
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Greedily select the best concept to intervene at each step.

    At each step, all remaining concepts are tried and the one that yields
    the highest accuracy is selected for permanent intervention.

    Parameters
    ----------
    model : nn.Module
        Trained CBM model.
    test_data : dict or (features, labels) tuple
        Test dataset. If dict, may contain "true_features" for noise scenarios.
    n_concepts : int
        Total number of concepts.
    device : str
    random_seed : int

    Returns
    -------
    dict
        {
          "intervention_order": list of int,      -- concept indices
          "concept_names_order": list of str,      -- corresponding names
          "accuracies": list of float,             -- accuracy after each intervention
          "marginal_gains": list of float,         -- per-step accuracy gain
          "baseline_accuracy": float,
          "full_accuracy": float,
        }
    """
    set_global_seed(random_seed)
    model = model.to(device)
    model.eval()

    features, labels = _get_tensors(test_data, device)

    if isinstance(test_data, dict) and "true_features" in test_data:
        true_features = test_data["true_features"]
        if not isinstance(true_features, torch.Tensor):
            true_features = torch.tensor(np.asarray(true_features), dtype=torch.float32)
        true_features = true_features.to(device)
    else:
        true_features = features

    with torch.no_grad():
        out_base = model(features)
        acc_base = _accuracy(out_base["logits"].argmax(dim=1), labels)

    remaining = set(range(n_concepts))
    intervention_order: List[int] = []
    accuracies: List[float] = [acc_base]
    current_override = torch.full_like(features, float("nan"))

    logger.info(f"Greedy intervention: baseline acc={acc_base:.4f}")

    for step in range(n_concepts):
        best_idx = -1
        best_acc = -1.0

        with torch.no_grad():
            for idx in remaining:
                trial_override = current_override.clone()
                trial_override[:, idx] = true_features[:, idx]
                out = model(features, concept_override=trial_override)
                acc = _accuracy(out["logits"].argmax(dim=1), labels)
                if acc > best_acc:
                    best_acc = acc
                    best_idx = idx

        remaining.remove(best_idx)
        intervention_order.append(best_idx)
        current_override[:, best_idx] = true_features[:, best_idx]
        accuracies.append(best_acc)

        concept_name = (
            CONCEPT_NAMES_12[best_idx]
            if best_idx < len(CONCEPT_NAMES_12)
            else f"concept_{best_idx}"
        )
        logger.info(
            f"  Step {step+1}/{n_concepts}: intervene {concept_name} "
            f"-> acc={best_acc:.4f}"
        )

    marginal_gains = [
        accuracies[i + 1] - accuracies[i]
        for i in range(len(accuracies) - 1)
    ]

    concept_names_order = [
        CONCEPT_NAMES_12[i] if i < len(CONCEPT_NAMES_12) else f"concept_{i}"
        for i in intervention_order
    ]

    result = {
        "intervention_order": intervention_order,
        "concept_names_order": concept_names_order,
        "accuracies": accuracies,
        "marginal_gains": marginal_gains,
        "n_steps": list(range(n_concepts + 1)),
        "baseline_accuracy": acc_base,
        "full_accuracy": accuracies[-1],
    }

    logger.info(
        f"Greedy intervention complete: "
        f"baseline={acc_base:.4f} -> full={accuracies[-1]:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Noise injection experiment
# ---------------------------------------------------------------------------

def run_noise_injection_experiment(
    model: nn.Module,
    test_data: Any,
    noise_stds: Optional[List[float]] = None,
    device: str = "cpu",
    random_seed: int = RANDOM_SEED,
    n_concepts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Inject Gaussian noise into concept values and measure classification
    degradation, then measure recovery via targeted intervention.

    For each noise level sigma in noise_stds:
        1. Inject N(0, sigma) noise into ALL concepts.
        2. Record accuracy (degraded performance).
        3. For each concept individually: inject noise, intervene with
           true value, record recovery.

    Parameters
    ----------
    model : nn.Module
        Trained CBM model.
    test_data : dict or (features, labels) tuple
        Clean test data (no pre-existing noise).
    noise_stds : list of float
        Gaussian noise standard deviations to test.
        Default: [0.1, 0.5, 1.0, 2.0].
    device : str
    random_seed : int
    n_concepts : int or None
        If None, inferred from features shape.

    Returns
    -------
    dict
        {
          "noise_stds": [0.1, 0.5, 1.0, 2.0],
          "clean_accuracy": float,
          "per_noise_level": {
              "0.1": {
                  "accuracy_noisy_all": float,
                  "per_concept_recovery": {concept_name: recovery_rate},
              },
              ...
          }
        }
    """
    if noise_stds is None:
        noise_stds = [0.1, 0.5, 1.0, 2.0]

    set_global_seed(random_seed)
    model = model.to(device)
    model.eval()

    features, labels = _get_tensors(test_data, device)

    if n_concepts is None:
        n_concepts = features.shape[1]

    rng = torch.Generator(device=device)
    rng.manual_seed(random_seed)

    with torch.no_grad():
        out_clean = model(features)
        clean_acc = _accuracy(out_clean["logits"].argmax(dim=1), labels)

    logger.info(f"Noise injection experiment: clean accuracy = {clean_acc:.4f}")

    per_noise_level: Dict[str, Any] = {}

    for sigma in noise_stds:
        logger.info(f"  Testing noise sigma={sigma}...")

        # Reset RNG per sigma level for independence between noise levels
        rng = torch.Generator(device=device)
        rng.manual_seed(random_seed + int(sigma * 1000))

        with torch.no_grad():
            # Inject noise into all concepts
            noise_all = torch.randn_like(features, generator=rng) * sigma
            features_noisy_all = features + noise_all
            out_noisy = model(features_noisy_all)
            acc_noisy = _accuracy(out_noisy["logits"].argmax(dim=1), labels)

        per_concept_recovery: Dict[str, Any] = {}

        for concept_idx in range(n_concepts):
            concept_name = (
                CONCEPT_NAMES_12[concept_idx]
                if concept_idx < len(CONCEPT_NAMES_12)
                else f"concept_{concept_idx}"
            )

            with torch.no_grad():
                # Inject noise into single concept only
                noise_single = torch.zeros_like(features)
                noise_single[:, concept_idx] = (
                    torch.randn(features.shape[0], generator=rng) * sigma
                )
                features_noisy_single = features + noise_single
                out_noisy_single = model(features_noisy_single)
                acc_noisy_single = _accuracy(
                    out_noisy_single["logits"].argmax(dim=1), labels
                )

                # Intervene: restore the noisy concept to its clean value
                override = torch.full_like(features_noisy_single, float("nan"))
                override[:, concept_idx] = features[:, concept_idx]
                out_intervened = model(features_noisy_single, concept_override=override)
                acc_intervened = _accuracy(
                    out_intervened["logits"].argmax(dim=1), labels
                )

            perf_drop = clean_acc - acc_noisy_single
            if perf_drop > 1e-6:
                recovery_rate = min(
                    (acc_intervened - acc_noisy_single) / perf_drop, 1.0
                )
            else:
                recovery_rate = 1.0

            per_concept_recovery[concept_name] = {
                "accuracy_clean": clean_acc,
                "accuracy_noisy_single": float(acc_noisy_single),
                "accuracy_intervened": float(acc_intervened),
                "performance_drop": float(perf_drop),
                "recovery_rate": float(recovery_rate),
            }

        per_noise_level[str(sigma)] = {
            "sigma": sigma,
            "accuracy_noisy_all": float(acc_noisy),
            "accuracy_drop_all": float(clean_acc - acc_noisy),
            "per_concept_recovery": per_concept_recovery,
        }

        logger.info(
            f"    sigma={sigma}: noisy_acc={acc_noisy:.4f} "
            f"(drop={clean_acc - acc_noisy:.4f})"
        )

    result = {
        "noise_stds": noise_stds,
        "clean_accuracy": float(clean_acc),
        "n_concepts": n_concepts,
        "per_noise_level": per_noise_level,
    }

    return result


# ---------------------------------------------------------------------------
# Case studies
# ---------------------------------------------------------------------------

def run_case_studies(
    model: nn.Module,
    test_data: Any,
    n_cases: int = 10,
    device: str = "cpu",
    random_seed: int = RANDOM_SEED,
    n_concepts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Select interesting misclassified cases and show concept values and
    intervention effects.

    For each misclassified example:
        1. Record predicted class, true class, concept values, probabilities.
        2. Try intervening each concept individually and record accuracy change.
        3. Identify which concept intervention most helps correct the prediction.

    Parameters
    ----------
    model : nn.Module
        Trained CBM model.
    test_data : dict or (features, labels) tuple
    n_cases : int
        Number of cases to analyze (selects first n_cases misclassified).
    device : str
    random_seed : int
    n_concepts : int or None

    Returns
    -------
    dict
        {"cases": [case_dict, ...]} with n_cases entries.
    """
    set_global_seed(random_seed)
    model = model.to(device)
    model.eval()

    features, labels = _get_tensors(test_data, device)

    if n_concepts is None:
        n_concepts = features.shape[1]

    concept_names = (
        CONCEPT_NAMES_12[:n_concepts]
        if n_concepts <= len(CONCEPT_NAMES_12)
        else CONCEPT_NAMES_12 + [f"concept_{i}" for i in range(12, n_concepts)]
    )

    with torch.no_grad():
        out = model(features)
        preds = out["logits"].argmax(dim=1)
        probs = out["probabilities"]
        concepts = out["concepts"]

    # Compute per-class concept means for meaningful intervention targets
    num_classes = int(labels.max().item()) + 1
    class_means: Dict[int, torch.Tensor] = {}
    for cls_idx in range(num_classes):
        mask = (labels == cls_idx)
        if mask.sum() > 0:
            class_means[cls_idx] = features[mask].mean(dim=0)

    # Find misclassified indices
    misclassified = (preds != labels).nonzero(as_tuple=True)[0].tolist()

    if len(misclassified) == 0:
        logger.warning("No misclassified examples found in test data.")
        return {"cases": [], "total_misclassified": 0}

    selected_indices = misclassified[: min(n_cases, len(misclassified))]
    cases: List[Dict[str, Any]] = []

    for sample_idx in selected_indices:
        true_label = int(labels[sample_idx].item())
        pred_label = int(preds[sample_idx].item())
        sample_features = features[sample_idx].unsqueeze(0)  # (1, n_concepts)

        true_class_name = (
            CLASS_NAMES[true_label] if true_label < len(CLASS_NAMES) else str(true_label)
        )
        pred_class_name = (
            CLASS_NAMES[pred_label] if pred_label < len(CLASS_NAMES) else str(pred_label)
        )

        # Concept values at this sample
        concept_values = {
            concept_names[i]: float(concepts[sample_idx, i].item())
            for i in range(n_concepts)
        }

        # Original class probabilities
        class_probs = {
            CLASS_NAMES[k] if k < len(CLASS_NAMES) else str(k): float(
                probs[sample_idx, k].item()
            )
            for k in range(probs.shape[1])
        }

        # Intervention analysis: for each concept, try fixing it
        intervention_analysis: List[Dict[str, Any]] = []

        # Use true class mean as intervention target (more informative than global 0.0)
        true_class_mean = class_means.get(true_label)

        with torch.no_grad():
            for concept_idx in range(n_concepts):
                # Intervention: replace concept value with the true class mean
                # for that concept. This simulates "what if this sample had a
                # typical value for its true class?"
                override_class_mean = torch.full_like(sample_features, float("nan"))
                if true_class_mean is not None:
                    intervened_value = float(true_class_mean[concept_idx].item())
                else:
                    intervened_value = 0.0  # fallback to standardized mean
                override_class_mean[:, concept_idx] = intervened_value

                out_intervened = model(sample_features, concept_override=override_class_mean)
                pred_intervened = int(
                    out_intervened["logits"].argmax(dim=1).item()
                )
                prob_true_class_intervened = float(
                    out_intervened["probabilities"][0, true_label].item()
                )
                prob_true_class_original = float(probs[sample_idx, true_label].item())
                prob_gain = prob_true_class_intervened - prob_true_class_original
                corrects = pred_intervened == true_label

                intervention_analysis.append({
                    "concept_name": concept_names[concept_idx],
                    "concept_idx": concept_idx,
                    "original_value": concept_values[concept_names[concept_idx]],
                    "intervened_to": intervened_value,
                    "pred_after_intervention": pred_intervened,
                    "pred_class_after": (
                        CLASS_NAMES[pred_intervened]
                        if pred_intervened < len(CLASS_NAMES)
                        else str(pred_intervened)
                    ),
                    "corrects_prediction": corrects,
                    "prob_true_class_gain": float(prob_gain),
                })

        # Sort by probability gain (most helpful intervention first)
        intervention_analysis.sort(key=lambda x: x["prob_true_class_gain"], reverse=True)

        cases.append({
            "sample_idx": sample_idx,
            "true_label": true_label,
            "true_class": true_class_name,
            "predicted_label": pred_label,
            "predicted_class": pred_class_name,
            "concept_values": concept_values,
            "class_probabilities": class_probs,
            "intervention_analysis": intervention_analysis,
            "top_corrective_concept": (
                intervention_analysis[0]["concept_name"]
                if intervention_analysis
                else None
            ),
        })

    result = {
        "n_cases_requested": n_cases,
        "n_cases_returned": len(cases),
        "total_misclassified": len(misclassified),
        "total_samples": int(features.shape[0]),
        "overall_accuracy": float(
            _accuracy(preds, labels)
        ),
        "cases": cases,
    }

    logger.info(
        f"Case studies: analyzed {len(cases)} misclassified examples "
        f"(total misclassified: {len(misclassified)}/{int(features.shape[0])})"
    )
    return result


# ---------------------------------------------------------------------------
# run_all_interventions
# ---------------------------------------------------------------------------

def run_all_interventions(
    model: nn.Module,
    test_data: Any,
    cfg: Any,
    output_dir: str | Path = "results/intervention",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run all intervention experiments and save results to output_dir.

    Parameters
    ----------
    model : nn.Module
        Trained CBM model (must support concept_override).
    test_data : dict or (features, labels) tuple
        Test dataset. Should be the domain-internal hold-out set.
    cfg : DictConfig
        Project configuration.
    output_dir : str or Path
        Directory for saving results.
    device : str
        Torch device.

    Returns
    -------
    dict
        {"sequential_random": ..., "sequential_greedy": ...,
         "noise_injection": ..., "case_studies": ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = getattr(getattr(cfg, "project", cfg), "random_seed", RANDOM_SEED)
    set_global_seed(seed)

    features, labels = _get_tensors(test_data, device)
    n_concepts = features.shape[1]

    logger.info("=" * 70)
    logger.info("Running ALL intervention experiments")
    logger.info(f"  n_samples={features.shape[0]}, n_concepts={n_concepts}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70)

    all_results: Dict[str, Any] = {}
    t_total = time.time()

    # 1. Sequential random
    logger.info("\n[1/4] Sequential random intervention...")
    n_trials = getattr(getattr(cfg, "intervention", None) or {}, "n_trials", 5)
    rand_result = intervene_sequential_random(
        model=model,
        test_data=test_data,
        n_concepts=n_concepts,
        n_trials=n_trials,
        device=device,
        random_seed=seed,
    )
    rand_result["_metadata"] = _make_metadata(
        model_name=model.__class__.__name__,
        n_concepts=n_concepts,
        concept_names=CONCEPT_NAMES_12[:n_concepts],
        random_seed=seed,
    )
    all_results["sequential_random"] = rand_result

    out_path = output_dir / "sequential_random.json"
    with open(out_path, "w") as f:
        json.dump(rand_result, f, indent=2, default=str)
    logger.info(f"  Saved to {out_path}")

    # 2. Sequential greedy
    logger.info("\n[2/4] Sequential greedy intervention...")
    greedy_result = intervene_sequential_greedy(
        model=model,
        test_data=test_data,
        n_concepts=n_concepts,
        device=device,
        random_seed=seed,
    )
    greedy_result["_metadata"] = _make_metadata(
        model_name=model.__class__.__name__,
        n_concepts=n_concepts,
        concept_names=CONCEPT_NAMES_12[:n_concepts],
        random_seed=seed,
    )
    all_results["sequential_greedy"] = greedy_result

    out_path = output_dir / "sequential_greedy.json"
    with open(out_path, "w") as f:
        json.dump(greedy_result, f, indent=2, default=str)
    logger.info(f"  Saved to {out_path}")

    # 3. Noise injection
    logger.info("\n[3/4] Noise injection experiment...")
    noise_stds_cfg = getattr(
        getattr(cfg, "intervention", None) or {}, "noise_stds", None
    )
    noise_stds = noise_stds_cfg if noise_stds_cfg else [0.1, 0.5, 1.0, 2.0]

    noise_result = run_noise_injection_experiment(
        model=model,
        test_data=test_data,
        noise_stds=list(noise_stds),
        device=device,
        random_seed=seed,
        n_concepts=n_concepts,
    )
    noise_result["_metadata"] = _make_metadata(
        model_name=model.__class__.__name__,
        n_concepts=n_concepts,
        concept_names=CONCEPT_NAMES_12[:n_concepts],
        random_seed=seed,
    )
    all_results["noise_injection"] = noise_result

    out_path = output_dir / "noise_injection.json"
    with open(out_path, "w") as f:
        json.dump(noise_result, f, indent=2, default=str)
    logger.info(f"  Saved to {out_path}")

    # 4. Case studies
    logger.info("\n[4/4] Case studies (misclassified examples)...")
    n_cases = getattr(getattr(cfg, "intervention", None) or {}, "n_cases", 10)
    case_result = run_case_studies(
        model=model,
        test_data=test_data,
        n_cases=n_cases,
        device=device,
        random_seed=seed,
        n_concepts=n_concepts,
    )
    case_result["_metadata"] = _make_metadata(
        model_name=model.__class__.__name__,
        n_concepts=n_concepts,
        concept_names=CONCEPT_NAMES_12[:n_concepts],
        random_seed=seed,
    )
    all_results["case_studies"] = case_result

    out_path = output_dir / "case_studies.json"
    with open(out_path, "w") as f:
        json.dump(case_result, f, indent=2, default=str)
    logger.info(f"  Saved to {out_path}")

    elapsed = time.time() - t_total
    logger.info(
        f"\nAll intervention experiments complete in {elapsed:.1f}s. "
        f"Results in {output_dir}"
    )

    return all_results
