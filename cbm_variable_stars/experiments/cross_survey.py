"""
CBM Variable Star Classification -- Cross-Survey Validation Experiment

S6 fix: Handles the 10dim vs 12dim modes for OGLE cross-survey evaluation.

OGLE has no native Gaia BP-RP photometry (C11) and uses a different magnitude
band for C12.  Two evaluation modes are supported:

    - "10dim": Use only the 10 band-independent concepts (drop C11, C12).
               Provides a fair, photometry-free cross-survey comparison.

    - "12dim_with_match": Use all 12 concepts, restricted to sources that
               have a Gaia cross-match so C11/C12 are available.

The Gaia-trained model is evaluated directly on OGLE data without any
domain adaptation, measuring the degree of concept-space generalization.
"""

from __future__ import annotations

import json
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
    CONCEPTS_CROSS_SURVEY_10,
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


def _extract_tensor_data(
    dataset: Any,
    concept_names: List[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract feature and label tensors from various data representations.

    Accepts:
        - dict with "features" (np.ndarray or Tensor) and "labels"
        - tuple (features, labels)
        - pandas DataFrame (columns = concept_names + "label")
    """
    try:
        import pandas as pd
        is_df = isinstance(dataset, pd.DataFrame)
    except ImportError:
        is_df = False

    if is_df:
        import pandas as pd
        df = dataset
        available = [c for c in concept_names if c in df.columns]
        if len(available) < len(concept_names):
            missing = [c for c in concept_names if c not in df.columns]
            logger.warning(f"Missing concept columns in DataFrame: {missing}")
        features = df[available].values.astype(np.float32)
        labels = df["label"].values.astype(np.int64)
    elif isinstance(dataset, dict):
        features = dataset["features"]
        labels = dataset["labels"]
    else:
        features, labels = dataset

    if not isinstance(features, torch.Tensor):
        features = torch.tensor(np.asarray(features, dtype=np.float32))
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(np.asarray(labels, dtype=np.int64))

    return features.to(device), labels.to(device)


def _evaluate_model(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Evaluate a CBM model on features/labels and return full metrics.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    model.eval()
    with torch.no_grad():
        out = model(features)
        preds = out["logits"].argmax(dim=1).cpu().numpy()
        probs = out["probabilities"].cpu().numpy()
        concepts = out["concepts"].cpu().numpy()

    labels_np = labels.cpu().numpy()

    all_labels = list(range(len(class_names)))
    acc = float(accuracy_score(labels_np, preds))
    macro_f1 = float(f1_score(labels_np, preds, labels=all_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels_np, preds, labels=all_labels, average="weighted", zero_division=0))
    macro_prec = float(
        precision_score(labels_np, preds, labels=all_labels, average="macro", zero_division=0)
    )
    macro_rec = float(
        recall_score(labels_np, preds, labels=all_labels, average="macro", zero_division=0)
    )

    per_class_f1 = f1_score(labels_np, preds, labels=all_labels, average=None, zero_division=0)
    per_class_prec = precision_score(labels_np, preds, labels=all_labels, average=None, zero_division=0)
    per_class_rec = recall_score(labels_np, preds, labels=all_labels, average=None, zero_division=0)
    cm = confusion_matrix(labels_np, preds, labels=all_labels)

    per_class = {}
    for i, cn in enumerate(class_names):
        if i < len(per_class_f1):
            per_class[cn] = {
                "f1": float(per_class_f1[i]),
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "support": int(np.sum(labels_np == i)),
            }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(labels_np)),
        "predictions": preds.tolist(),
        "true_labels": labels_np.tolist(),
        "concept_means": concepts.mean(axis=0).tolist(),
    }


def _subset_features_by_names(
    features: torch.Tensor,
    all_names: List[str],
    target_names: List[str],
) -> torch.Tensor:
    """Extract column subset of features tensor by concept name."""
    indices = [all_names.index(n) for n in target_names]
    return features[:, indices]


# ---------------------------------------------------------------------------
# run_cross_survey_evaluation
# ---------------------------------------------------------------------------

def run_cross_survey_evaluation(
    model: nn.Module,
    gaia_test: Any,
    ogle_test: Any,
    mode: str = "10dim",
    device: str = "cpu",
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Evaluate a Gaia-trained model on OGLE cross-survey data.

    Parameters
    ----------
    model : nn.Module
        CBM model trained on Gaia data (12-concept or 10-concept input).
    gaia_test : any
        Gaia domain-internal test data (dict, tuple, or DataFrame).
        Used for within-domain baseline comparison.
    ogle_test : any
        OGLE cross-survey test data.
        - For mode="10dim": must contain only 10-concept columns.
        - For mode="12dim_with_match": must contain all 12 concepts;
          only sources with Gaia match are included.
    mode : str
        "10dim" or "12dim_with_match".
    device : str
    random_seed : int

    Returns
    -------
    dict
        {
          "_metadata": {...},
          "mode": "10dim" | "12dim_with_match",
          "gaia_test_results": {...},
          "ogle_test_results": {...},
          "cross_survey_gap": {...},  -- metric differences
        }

    Notes
    -----
    S6 fix:
        mode="10dim": Drop C11 (color_bp_rp) and C12 (mean_mag).
            OGLE uses I-band magnitudes (incomparable to Gaia G)
            and has no native BP-RP photometry.
            The 10-dim model was trained on CONCEPTS_CROSS_SURVEY_10.

        mode="12dim_with_match": Use all 12 concepts.
            Only OGLE sources with a Gaia cross-match are used
            (they have real Gaia BP-RP and are brighter/more reliable).
    """
    set_global_seed(random_seed)
    model = model.to(device)
    model.eval()

    if mode not in ("10dim", "12dim_with_match"):
        raise ValueError(
            f"Unknown mode='{mode}'. Must be '10dim' or '12dim_with_match'."
        )

    # Determine active concepts based on mode
    if mode == "10dim":
        active_concepts = CONCEPTS_CROSS_SURVEY_10
        n_concepts = len(CONCEPTS_CROSS_SURVEY_10)
        mode_description = (
            "10 band-independent concepts only (drop C11 color_bp_rp, C12 mean_mag). "
            "Fair cross-survey comparison without photometric band mismatch."
        )
    else:  # 12dim_with_match
        active_concepts = CONCEPT_NAMES_12
        n_concepts = 12
        mode_description = (
            "All 12 concepts. Only OGLE sources with Gaia cross-match included. "
            "C11/C12 from matched Gaia photometry."
        )

    logger.info(f"Cross-survey evaluation: mode={mode}")
    logger.info(f"  Active concepts ({n_concepts}): {active_concepts}")

    # Extract Gaia test data
    gaia_features, gaia_labels = _extract_tensor_data(
        gaia_test, active_concepts, device
    )

    # Extract OGLE test data
    ogle_features, ogle_labels = _extract_tensor_data(
        ogle_test, active_concepts, device
    )

    # Fix #05: In 10dim mode, if the model expects 12-dim input, pad missing
    # concept columns with zeros (the standardized mean) so features match
    # the model's expected input dimension.
    if mode == "10dim":
        # Find which CONCEPT_NAMES_12 columns are missing from CONCEPTS_CROSS_SURVEY_10
        missing_concepts = [c for c in CONCEPT_NAMES_12 if c not in CONCEPTS_CROSS_SURVEY_10]
        if missing_concepts:
            missing_idx = [CONCEPT_NAMES_12.index(c) for c in missing_concepts]
            logger.info(
                f"  10dim: padding {len(missing_concepts)} missing concepts "
                f"{missing_concepts} with zeros (standardized mean)"
            )

            def _pad_to_12dim(feats: torch.Tensor) -> torch.Tensor:
                """Insert zero columns at the positions of missing concepts."""
                n_samples = feats.shape[0]
                full = torch.zeros(n_samples, len(CONCEPT_NAMES_12), device=feats.device)
                # Map 10dim columns to their positions in the 12dim layout
                present_idx = [CONCEPT_NAMES_12.index(c) for c in CONCEPTS_CROSS_SURVEY_10]
                for src_col, dst_col in enumerate(present_idx):
                    full[:, dst_col] = feats[:, src_col]
                return full

            gaia_features = _pad_to_12dim(gaia_features)
            ogle_features = _pad_to_12dim(ogle_features)

    logger.info(
        f"  Gaia test: {gaia_features.shape[0]} samples | "
        f"OGLE test: {ogle_features.shape[0]} samples"
    )

    # Evaluate on Gaia (within-domain baseline)
    logger.info("  Evaluating on Gaia test set (within-domain)...")
    gaia_results = _evaluate_model(model, gaia_features, gaia_labels)

    # Evaluate on OGLE (cross-survey)
    logger.info("  Evaluating on OGLE test set (cross-survey)...")
    ogle_results = _evaluate_model(model, ogle_features, ogle_labels)

    # Cross-survey performance gap
    cross_survey_gap = {
        "accuracy": ogle_results["accuracy"] - gaia_results["accuracy"],
        "macro_f1": ogle_results["macro_f1"] - gaia_results["macro_f1"],
        "weighted_f1": ogle_results["weighted_f1"] - gaia_results["weighted_f1"],
    }

    logger.info(
        f"  Gaia acc={gaia_results['accuracy']:.4f} | "
        f"OGLE acc={ogle_results['accuracy']:.4f} | "
        f"gap={cross_survey_gap['accuracy']:+.4f}"
    )

    result = {
        "_metadata": _make_metadata(
            model_name=model.__class__.__name__,
            n_concepts=n_concepts,
            concept_names=active_concepts,
            random_seed=random_seed,
        ),
        "mode": mode,
        "mode_description": mode_description,
        "active_concepts": active_concepts,
        "n_gaia_test_samples": int(gaia_features.shape[0]),
        "n_ogle_test_samples": int(ogle_features.shape[0]),
        "gaia_test_results": gaia_results,
        "ogle_test_results": ogle_results,
        "cross_survey_gap": cross_survey_gap,
    }

    return result


# ---------------------------------------------------------------------------
# run_cross_survey_experiment
# ---------------------------------------------------------------------------

def run_cross_survey_experiment(
    cfg: Any,
    output_dir: str | Path = "results/cross_survey",
    model: Optional[nn.Module] = None,
    gaia_test: Optional[Any] = None,
    ogle_test: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Full cross-survey experiment running both 10dim and 12dim_with_match modes.

    If model/data are not provided, attempts to load them from the paths
    specified in cfg.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration. Used for:
            - cfg.paths.processed: where to find processed data parquets
            - cfg.project.random_seed
    output_dir : str or Path
        Directory for saving results JSON files.
    model : nn.Module or None
        Pre-loaded CBM model. If None, loaded from cfg.paths.
    gaia_test : any or None
        Gaia hold-out test data. If None, loaded from processed dir.
    ogle_test : any or None
        OGLE test data. If None, loaded from processed dir.
    device : str

    Returns
    -------
    dict
        {"10dim": result_dict, "12dim_with_match": result_dict}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = getattr(getattr(cfg, "project", cfg), "random_seed", RANDOM_SEED)
    set_global_seed(seed)

    # Load data if not provided
    if gaia_test is None or ogle_test is None:
        try:
            import pandas as pd
            processed_dir = Path(
                getattr(getattr(cfg, "paths", cfg), "processed", "data/processed")
            )

            if gaia_test is None:
                gaia_path = processed_dir / "test_in_domain.parquet"
                if gaia_path.exists():
                    gaia_test = pd.read_parquet(gaia_path)
                    logger.info(f"Loaded Gaia test data: {len(gaia_test)} rows")
                else:
                    logger.warning(f"Gaia test data not found at {gaia_path}")

            if ogle_test is None:
                ogle_path = processed_dir / "test_cross_survey.parquet"
                if ogle_path.exists():
                    ogle_test = pd.read_parquet(ogle_path)
                    logger.info(f"Loaded OGLE test data: {len(ogle_test)} rows")
                else:
                    logger.warning(f"OGLE test data not found at {ogle_path}")
        except Exception as e:
            logger.warning(f"Could not load data from filesystem: {e}")

    if gaia_test is None or ogle_test is None:
        raise RuntimeError(
            "gaia_test and ogle_test must be provided either directly or "
            "via cfg.paths.processed."
        )

    # Load model if not provided
    if model is None:
        try:
            from cbm_variable_stars.models import create_model
            model_paths_dir = Path(
                getattr(getattr(cfg, "paths", cfg), "results", "results")
            )
            # Try to load the default hard_cbm model
            from cbm_variable_stars.shared.constants import N_CLASSES
            model = create_model("hard_cbm", num_concepts=12, num_classes=N_CLASSES)
            # Look for a saved checkpoint
            ckpt_candidates = list(model_paths_dir.glob("**/best_model_fold*.pt"))
            if ckpt_candidates:
                ckpt_path = sorted(ckpt_candidates)[-1]
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(state.get("model_state_dict", state))
                logger.info(f"Loaded model checkpoint: {ckpt_path}")
            else:
                logger.warning("No checkpoint found; using randomly initialized model.")
        except Exception as e:
            raise RuntimeError(f"Could not load model: {e}")

    logger.info("=" * 70)
    logger.info("Cross-Survey Validation Experiment (S6 fix)")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70)

    t_total = time.time()
    all_results: Dict[str, Any] = {}

    # --- Mode 1: 10dim ---
    logger.info("\n[1/2] Mode: 10dim (10 band-independent concepts)")
    try:
        result_10dim = run_cross_survey_evaluation(
            model=model,
            gaia_test=gaia_test,
            ogle_test=ogle_test,
            mode="10dim",
            device=device,
            random_seed=seed,
        )
        all_results["10dim"] = result_10dim

        out_path = output_dir / "gaia_vs_ogle_10dim.json"
        with open(out_path, "w") as f:
            json.dump(result_10dim, f, indent=2, default=str)
        logger.info(f"  10dim results saved to {out_path}")
    except Exception as e:
        logger.error(f"10dim cross-survey evaluation failed: {e}")
        all_results["10dim"] = {"error": str(e)}

    # --- Mode 2: 12dim_with_match ---
    logger.info("\n[2/2] Mode: 12dim_with_match (all 12 concepts, Gaia-matched sources)")
    try:
        # For 12dim mode, filter OGLE data to sources with Gaia match
        import pandas as pd
        if isinstance(ogle_test, pd.DataFrame) and "has_gaia_match" in ogle_test.columns:
            ogle_matched = ogle_test[ogle_test["has_gaia_match"]].copy()
            n_before = len(ogle_test)
            n_after = len(ogle_matched)
            logger.info(
                f"  12dim: filtered OGLE to {n_after}/{n_before} "
                f"Gaia-matched sources ({100*n_after/n_before:.1f}%)"
            )
        else:
            ogle_matched = ogle_test
            logger.warning(
                "  12dim: 'has_gaia_match' column not found; "
                "using all OGLE sources (may include missing C11/C12)."
            )

        result_12dim = run_cross_survey_evaluation(
            model=model,
            gaia_test=gaia_test,
            ogle_test=ogle_matched,
            mode="12dim_with_match",
            device=device,
            random_seed=seed,
        )
        all_results["12dim_with_match"] = result_12dim

        out_path = output_dir / "gaia_vs_ogle_12dim.json"
        with open(out_path, "w") as f:
            json.dump(result_12dim, f, indent=2, default=str)
        logger.info(f"  12dim results saved to {out_path}")
    except Exception as e:
        logger.error(f"12dim cross-survey evaluation failed: {e}")
        all_results["12dim_with_match"] = {"error": str(e)}

    elapsed = time.time() - t_total

    # Summary
    summary_rows = []
    for mode_key, res in all_results.items():
        if "error" in res:
            continue
        ogle_res = res.get("ogle_test_results", {})
        gaia_res = res.get("gaia_test_results", {})
        gap = res.get("cross_survey_gap", {})
        summary_rows.append({
            "mode": mode_key,
            "gaia_accuracy": gaia_res.get("accuracy", float("nan")),
            "ogle_accuracy": ogle_res.get("accuracy", float("nan")),
            "gaia_macro_f1": gaia_res.get("macro_f1", float("nan")),
            "ogle_macro_f1": ogle_res.get("macro_f1", float("nan")),
            "accuracy_gap": gap.get("accuracy", float("nan")),
            "macro_f1_gap": gap.get("macro_f1", float("nan")),
        })

    summary = {
        "_metadata": _make_metadata(
            model_name=model.__class__.__name__,
            n_concepts=12,
            concept_names=CONCEPT_NAMES_12,
            random_seed=seed,
        ),
        "elapsed_seconds": elapsed,
        "summary": summary_rows,
        "modes_run": list(all_results.keys()),
    }

    summary_path = output_dir / "cross_survey_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nCross-survey summary saved to {summary_path}")

    return all_results
