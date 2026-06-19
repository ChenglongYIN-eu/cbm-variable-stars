"""
CBM Variable Star Classification -- Ablation Experiments A1-A5

Ablation experiments to assess the contribution of individual concept groups,
architectural choices, and training modes to overall model performance.

Experiments:
    A1: Remove C11 (color_bp_rp) -- band-dependent concept removal
    A2: Minimal 4-concept set (period, amplitude, R21, phi21)
    A3: Extended 20-concept set
    A4: Hard CBM vs Soft CBM vs CEM architectural comparison
    A5: Training mode comparison (joint/sequential/independent) for Plan B;
        for Plan A: label predictor complexity (linear vs MLP) -- M1 fix
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_GROUPS,
    CONCEPT_NAMES_12,
    CONCEPT_NAMES_20,
    CONCEPTS_NO_COLOR,
    MINIMAL_CONCEPTS,
    N_CLASSES,
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
    """Build the mandatory _metadata block for every result JSON."""
    return {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "n_concepts": n_concepts,
        "concept_names": concept_names,
    }


def _subset_features(
    features: np.ndarray,
    all_concept_names: List[str],
    target_concept_names: List[str],
) -> np.ndarray:
    """
    Extract a column-subset of the feature matrix.

    Parameters
    ----------
    features : np.ndarray
        Full feature matrix, shape (N, len(all_concept_names)).
    all_concept_names : list of str
        Column names for `features` (must align with CONCEPT_NAMES_12 order).
    target_concept_names : list of str
        Subset of concept names to keep.

    Returns
    -------
    np.ndarray
        Reduced feature matrix, shape (N, len(target_concept_names)).
    """
    indices = [all_concept_names.index(c) for c in target_concept_names]
    return features[:, indices]


def _run_cv_for_ablation(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    n_concepts: int,
    concept_names: List[str],
    cfg: Any,
    output_dir: Optional[Path] = None,
    training_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Shared wrapper: run 5-fold CV via training.cross_val.run_cross_validation.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix already subsetted to the desired concept columns.
    labels : np.ndarray
        Integer class labels.
    model_name : str
        Model identifier passed to create_model().
    n_concepts : int
        Number of concepts (determines model input dimension).
    concept_names : list of str
        Names of the active concepts.
    cfg : OmegaConf DictConfig
        Project configuration.
    output_dir : Path or None
        If provided, CV results are saved here.
    training_mode : str or None
        Training mode for Plan B models ("joint", "sequential", "independent").
        If None, the default training mode is used.

    Returns
    -------
    dict
        {
          "_metadata": {...},
          "results": {aggregated CV metrics},
          "fold_results": [...],
        }
    """
    from cbm_variable_stars.training.cross_val import run_cross_validation

    seed = getattr(getattr(cfg, "project", cfg), "random_seed", RANDOM_SEED)
    set_global_seed(seed)

    model_kwargs = {"num_concepts": n_concepts, "num_classes": N_CLASSES}

    # Pull training hyper-parameters from cfg if available, otherwise use defaults
    hp = getattr(cfg, "training", None) or {}
    batch_size = getattr(hp, "batch_size", 256)
    lr = getattr(hp, "learning_rate", 1e-3)
    wd = getattr(hp, "weight_decay", 1e-4)
    max_epochs = getattr(hp, "max_epochs", 200)
    patience = getattr(hp, "patience", 15)

    out_dir_str = str(output_dir) if output_dir else "results/ablation"

    cv_kwargs: Dict[str, Any] = dict(
        features=features,
        labels=labels,
        model_name=model_name,
        model_kwargs=model_kwargs,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=wd,
        max_epochs=max_epochs,
        patience=patience,
        random_seed=seed,
        output_dir=out_dir_str,
    )
    if training_mode is not None:
        cv_kwargs["training_mode"] = training_mode

    cv_result = run_cross_validation(**cv_kwargs)

    metadata = _make_metadata(
        model_name=model_name,
        n_concepts=n_concepts,
        concept_names=concept_names,
        random_seed=seed,
    )

    return {
        "_metadata": metadata,
        "results": cv_result.get("aggregated", {}),
        "fold_results": [
            {k: v for k, v in r.items() if k not in ("predictions",)}
            for r in cv_result.get("fold_results", [])
        ],
    }


# ---------------------------------------------------------------------------
# A1: Remove C11 (color_bp_rp)
# ---------------------------------------------------------------------------

def run_ablation_A1(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: Any,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ablation A1: Remove C11 (color_bp_rp) -- compare with full 12-concept model.

    Uses CONCEPTS_NO_COLOR (11 concepts: all 12 minus color_bp_rp).

    Parameters
    ----------
    features : np.ndarray
        Full 12-concept feature matrix, shape (N, 12), column order = CONCEPT_NAMES_12.
    labels : np.ndarray
        Integer class labels, shape (N,).
    cfg : DictConfig
        Project configuration.
    output_dir : Path or None
        Directory to save results JSON.

    Returns
    -------
    dict
        Ablation result with _metadata and CV metrics for both
        the no-color model and the full 12-concept baseline.
    """
    logger.info("=" * 60)
    logger.info("Ablation A1: Remove C11 (color_bp_rp)")
    logger.info(f"  Full concepts (12): {CONCEPT_NAMES_12}")
    logger.info(f"  No-color concepts (11): {CONCEPTS_NO_COLOR}")
    logger.info("=" * 60)

    t0 = time.time()

    # --- No-color model (11 concepts) ---
    features_no_color = _subset_features(features, CONCEPT_NAMES_12, CONCEPTS_NO_COLOR)
    result_no_color = _run_cv_for_ablation(
        features=features_no_color,
        labels=labels,
        model_name="hard_cbm",
        n_concepts=len(CONCEPTS_NO_COLOR),
        concept_names=CONCEPTS_NO_COLOR,
        cfg=cfg,
        output_dir=output_dir / "A1_no_color" if output_dir else None,
    )

    # --- Full 12-concept baseline ---
    result_full = _run_cv_for_ablation(
        features=features,
        labels=labels,
        model_name="hard_cbm",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A1_full" if output_dir else None,
    )

    elapsed = time.time() - t0

    result = {
        "_metadata": _make_metadata(
            model_name="ablation_A1_remove_color",
            n_concepts=11,
            concept_names=CONCEPTS_NO_COLOR,
        ),
        "ablation_type": "A1_remove_color",
        "description": "Remove C11 (color_bp_rp). Compares 11-concept vs 12-concept model.",
        "removed_concepts": ["color_bp_rp"],
        "no_color_model": result_no_color,
        "full_model": result_full,
        "performance_delta": {
            "accuracy": (
                result_no_color["results"].get("accuracy_mean", float("nan"))
                - result_full["results"].get("accuracy_mean", float("nan"))
            ),
            "macro_f1": (
                result_no_color["results"].get("macro_f1_mean", float("nan"))
                - result_full["results"].get("macro_f1_mean", float("nan"))
            ),
        },
        "elapsed_seconds": elapsed,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "A1_remove_color.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"A1 results saved to {out_path}")

    logger.info(
        f"A1 complete. Accuracy delta (no_color - full): "
        f"{result['performance_delta']['accuracy']:+.4f}  "
        f"F1 delta: {result['performance_delta']['macro_f1']:+.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# A2: Minimal 4-concept set
# ---------------------------------------------------------------------------

def run_ablation_A2(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: Any,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ablation A2: Minimal 4-concept set (period, amplitude, R21, phi21).

    Uses MINIMAL_CONCEPTS to probe the information floor -- how well can the
    model classify with only the most physically fundamental features?

    Parameters
    ----------
    features : np.ndarray
        Full 12-concept feature matrix, shape (N, 12).
    labels : np.ndarray
        Integer class labels.
    cfg : DictConfig
    output_dir : Path or None

    Returns
    -------
    dict
        Ablation result comparing MINIMAL_CONCEPTS vs full 12-concept model.
    """
    logger.info("=" * 60)
    logger.info("Ablation A2: Minimal 4-concept set")
    logger.info(f"  Minimal concepts: {MINIMAL_CONCEPTS}")
    logger.info("=" * 60)

    t0 = time.time()

    features_minimal = _subset_features(features, CONCEPT_NAMES_12, MINIMAL_CONCEPTS)
    result_minimal = _run_cv_for_ablation(
        features=features_minimal,
        labels=labels,
        model_name="hard_cbm",
        n_concepts=len(MINIMAL_CONCEPTS),
        concept_names=MINIMAL_CONCEPTS,
        cfg=cfg,
        output_dir=output_dir / "A2_minimal" if output_dir else None,
    )

    result_full = _run_cv_for_ablation(
        features=features,
        labels=labels,
        model_name="hard_cbm",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A2_full" if output_dir else None,
    )

    elapsed = time.time() - t0

    result = {
        "_metadata": _make_metadata(
            model_name="ablation_A2_minimal_concepts",
            n_concepts=4,
            concept_names=MINIMAL_CONCEPTS,
        ),
        "ablation_type": "A2_minimal_concepts",
        "description": (
            "Minimal 4-concept model (period, amplitude, R21, phi21). "
            "Establishes the performance floor for physically interpretable features."
        ),
        "active_concepts": MINIMAL_CONCEPTS,
        "dropped_concepts": [c for c in CONCEPT_NAMES_12 if c not in MINIMAL_CONCEPTS],
        "minimal_model": result_minimal,
        "full_model": result_full,
        "performance_delta": {
            "accuracy": (
                result_minimal["results"].get("accuracy_mean", float("nan"))
                - result_full["results"].get("accuracy_mean", float("nan"))
            ),
            "macro_f1": (
                result_minimal["results"].get("macro_f1_mean", float("nan"))
                - result_full["results"].get("macro_f1_mean", float("nan"))
            ),
        },
        "elapsed_seconds": elapsed,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "A2_minimal_concepts.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"A2 results saved to {out_path}")

    return result


# ---------------------------------------------------------------------------
# A3: Extended 20-concept set
# ---------------------------------------------------------------------------

def run_ablation_A3(
    features_20: np.ndarray,
    labels: np.ndarray,
    cfg: Any,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ablation A3: Extended 20-concept model vs 12-concept baseline.

    Parameters
    ----------
    features_20 : np.ndarray
        20-concept feature matrix, shape (N, 20), column order = CONCEPT_NAMES_20.
    labels : np.ndarray
        Integer class labels.
    cfg : DictConfig
    output_dir : Path or None

    Returns
    -------
    dict
        Ablation result comparing 20-concept vs 12-concept model.
    """
    logger.info("=" * 60)
    logger.info("Ablation A3: Extended 20-concept set")
    logger.info(f"  Extended concepts (20): {CONCEPT_NAMES_20}")
    logger.info("=" * 60)

    t0 = time.time()

    result_20 = _run_cv_for_ablation(
        features=features_20,
        labels=labels,
        model_name="hard_cbm",
        n_concepts=20,
        concept_names=CONCEPT_NAMES_20,
        cfg=cfg,
        output_dir=output_dir / "A3_extended_20" if output_dir else None,
    )

    # 12-concept baseline: extract the 12 standard concepts by name
    features_12 = _subset_features(features_20, CONCEPT_NAMES_20, CONCEPT_NAMES_12)
    result_12 = _run_cv_for_ablation(
        features=features_12,
        labels=labels,
        model_name="hard_cbm",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A3_base_12" if output_dir else None,
    )

    elapsed = time.time() - t0

    extra_concepts = CONCEPT_NAMES_20[12:]

    result = {
        "_metadata": _make_metadata(
            model_name="ablation_A3_extended_20_concepts",
            n_concepts=20,
            concept_names=CONCEPT_NAMES_20,
        ),
        "ablation_type": "A3_extended_20_concepts",
        "description": (
            "Extended 20-concept model. Additional concepts: "
            f"{extra_concepts}"
        ),
        "extra_concepts": extra_concepts,
        "extended_20_model": result_20,
        "base_12_model": result_12,
        "performance_delta": {
            "accuracy": (
                result_20["results"].get("accuracy_mean", float("nan"))
                - result_12["results"].get("accuracy_mean", float("nan"))
            ),
            "macro_f1": (
                result_20["results"].get("macro_f1_mean", float("nan"))
                - result_12["results"].get("macro_f1_mean", float("nan"))
            ),
        },
        "elapsed_seconds": elapsed,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "A3_extended_20_concepts.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"A3 results saved to {out_path}")

    return result


# ---------------------------------------------------------------------------
# A4: Hard CBM vs Soft CBM vs CEM
# ---------------------------------------------------------------------------

def run_ablation_A4(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: Any,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ablation A4: Architectural comparison -- Hard CBM vs Soft CBM vs CEM.

    All three models are trained on the same data (full 12-concept set).
    This isolates the effect of the concept bottleneck design choice.

    Parameters
    ----------
    features : np.ndarray
        Full 12-concept feature matrix, shape (N, 12).
    labels : np.ndarray
        Integer class labels.
    cfg : DictConfig
    output_dir : Path or None

    Returns
    -------
    dict
        Results for all three architectures.
    """
    logger.info("=" * 60)
    logger.info("Ablation A4: Hard CBM vs Soft CBM vs CEM")
    logger.info("=" * 60)

    t0 = time.time()
    arch_results: Dict[str, Any] = {}

    # --- Hard CBM (Plan A) ---
    logger.info("A4: Training HardCBM (Plan A)...")
    arch_results["hard_cbm"] = _run_cv_for_ablation(
        features=features,
        labels=labels,
        model_name="hard_cbm",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A4_hard_cbm" if output_dir else None,
    )

    # --- Hard CBM Calibrated (Plan B) ---
    logger.info("A4: Training HardCBM_Calibrated (Plan B)...")
    arch_results["hard_cbm_cal"] = _run_cv_for_ablation(
        features=features,
        labels=labels,
        model_name="hard_cbm_cal",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A4_hard_cbm_cal" if output_dir else None,
    )

    # --- Soft CBM ---
    logger.info("A4: Training SoftCBM...")
    arch_results["soft_cbm"] = _run_cv_for_ablation(
        features=features,
        labels=labels,
        model_name="soft_cbm",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A4_soft_cbm" if output_dir else None,
    )

    # --- CEM (Concept Embedding Model) ---
    logger.info("A4: Training CEM...")
    arch_results["cem"] = _run_cv_for_ablation(
        features=features,
        labels=labels,
        model_name="cem",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A4_cem" if output_dir else None,
    )

    elapsed = time.time() - t0

    # Build comparison table
    comparison = {}
    for arch_name, res in arch_results.items():
        agg = res.get("results", {})
        comparison[arch_name] = {
            "accuracy_mean": agg.get("accuracy_mean", float("nan")),
            "accuracy_std": agg.get("accuracy_std", float("nan")),
            "macro_f1_mean": agg.get("macro_f1_mean", float("nan")),
            "macro_f1_std": agg.get("macro_f1_std", float("nan")),
        }

    result = {
        "_metadata": _make_metadata(
            model_name="ablation_A4_hard_vs_soft",
            n_concepts=12,
            concept_names=CONCEPT_NAMES_12,
        ),
        "ablation_type": "A4_hard_vs_soft_cbm",
        "description": (
            "Compare HardCBM (Plan A), HardCBM_Calibrated (Plan B), "
            "SoftCBM, and CEM on identical data."
        ),
        "comparison_table": comparison,
        "detailed_results": arch_results,
        "elapsed_seconds": elapsed,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "A4_hard_vs_soft.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"A4 results saved to {out_path}")

    return result


# ---------------------------------------------------------------------------
# A5: Training modes / label predictor complexity (M1 fix)
# ---------------------------------------------------------------------------

def run_ablation_A5(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: Any,
    output_dir: Optional[Path] = None,
    concept_gt: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Ablation A5:
        - For Plan B: Compare training modes (joint, sequential, independent).
        - For Plan A (A5a, M1 fix): Compare label predictor complexity
          (linear vs MLP).

    The M1 fix acknowledges that A5 (training modes) is only meaningful for
    Plan B where a concept predictor exists.  For Plan A the analogous
    ablation is label predictor complexity.

    Parameters
    ----------
    features : np.ndarray
        Full 12-concept feature matrix, shape (N, 12).
    labels : np.ndarray
        Integer class labels.
    cfg : DictConfig
    output_dir : Path or None
    concept_gt : np.ndarray or None
        Concept ground-truth annotations, shape (N, n_concepts).
        Required for meaningful A5b (sequential/independent training modes).

    Returns
    -------
    dict
        Results for both A5a (Plan A predictor complexity) and
        A5b (Plan B training modes).
    """
    logger.info("=" * 60)
    logger.info("Ablation A5: Training modes (Plan B) + Predictor complexity (Plan A)")
    logger.info("=" * 60)

    t0 = time.time()
    a5a_results: Dict[str, Any] = {}
    a5b_results: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # A5a (Plan A): Label predictor complexity -- linear vs MLP
    # -----------------------------------------------------------------------
    logger.info("A5a (Plan A): HardCBM_Linear vs HardCBM_MLP...")

    # Linear label predictor
    a5a_results["hard_cbm_linear"] = _run_cv_for_ablation(
        features=features,
        labels=labels,
        model_name="hard_cbm_linear",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A5a_linear" if output_dir else None,
    )

    # MLP label predictor (default HardCBM)
    a5a_results["hard_cbm_mlp"] = _run_cv_for_ablation(
        features=features,
        labels=labels,
        model_name="hard_cbm",
        n_concepts=12,
        concept_names=CONCEPT_NAMES_12,
        cfg=cfg,
        output_dir=output_dir / "A5a_mlp" if output_dir else None,
    )

    # -----------------------------------------------------------------------
    # A5b (Plan B): Training mode -- joint / sequential / independent
    # -----------------------------------------------------------------------
    logger.info("A5b (Plan B): Joint vs Sequential vs Independent training modes...")

    if concept_gt is None:
        logger.warning(
            "A5b (training mode ablation) requires concept_gt for meaningful results. "
            "Without concept_gt, sequential and independent modes degenerate to joint training."
        )

    for mode in ("joint", "sequential", "independent"):
        logger.info(f"  A5b: mode={mode}")
        a5b_results[mode] = _run_cv_for_ablation(
            features=features,
            labels=labels,
            model_name="hard_cbm_cal",
            n_concepts=12,
            concept_names=CONCEPT_NAMES_12,
            cfg=cfg,
            output_dir=output_dir / f"A5b_{mode}" if output_dir else None,
            training_mode=mode,
        )

    elapsed = time.time() - t0

    # Build comparison tables
    a5a_comparison = {
        name: {
            "accuracy_mean": res["results"].get("accuracy_mean", float("nan")),
            "macro_f1_mean": res["results"].get("macro_f1_mean", float("nan")),
        }
        for name, res in a5a_results.items()
    }

    a5b_comparison = {
        mode: {
            "accuracy_mean": res["results"].get("accuracy_mean", float("nan")),
            "macro_f1_mean": res["results"].get("macro_f1_mean", float("nan")),
        }
        for mode, res in a5b_results.items()
    }

    result = {
        "_metadata": _make_metadata(
            model_name="ablation_A5_training_modes",
            n_concepts=12,
            concept_names=CONCEPT_NAMES_12,
        ),
        "ablation_type": "A5_training_modes",
        "description": (
            "A5a (Plan A / M1 fix): HardCBM_Linear vs HardCBM_MLP. "
            "A5b (Plan B): joint vs sequential vs independent training."
        ),
        "A5a_plan_a_predictor_complexity": {
            "description": "M1 fix: linear vs MLP label predictor for Plan A.",
            "comparison": a5a_comparison,
            "detailed": a5a_results,
        },
        "A5b_plan_b_training_modes": {
            "description": "Training mode ablation -- only applies to Plan B (hard_cbm_cal).",
            "comparison": a5b_comparison,
            "detailed": a5b_results,
        },
        "elapsed_seconds": elapsed,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "A5_training_modes.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"A5 results saved to {out_path}")

    return result


# ---------------------------------------------------------------------------
# run_all_ablations
# ---------------------------------------------------------------------------

def run_all_ablations(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: Any,
    output_dir: str | Path = "results/ablation",
    features_20: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run all ablation experiments (A1-A5) and persist results.

    Parameters
    ----------
    features : np.ndarray
        Full 12-concept feature matrix, shape (N, 12).
    labels : np.ndarray
        Integer class labels.
    cfg : DictConfig
        Project configuration.
    output_dir : str or Path
        Root directory for saving all ablation results.
    features_20 : np.ndarray or None
        20-concept feature matrix for A3. If None, A3 is skipped.

    Returns
    -------
    dict
        {ablation_name: result_dict} for A1-A5.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Running ALL ablation experiments A1-A5")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)

    t_total = time.time()
    all_results: Dict[str, Any] = {}

    # A1: Remove color
    logger.info("\n[Ablation A1/5] Remove C11 (color_bp_rp)")
    all_results["A1"] = run_ablation_A1(features, labels, cfg, output_dir)

    # A2: Minimal concepts
    logger.info("\n[Ablation A2/5] Minimal 4-concept set")
    all_results["A2"] = run_ablation_A2(features, labels, cfg, output_dir)

    # A3: Extended 20-concept set (optional)
    if features_20 is not None:
        logger.info("\n[Ablation A3/5] Extended 20-concept set")
        all_results["A3"] = run_ablation_A3(features_20, labels, cfg, output_dir)
    else:
        logger.warning("A3 skipped: features_20 not provided.")
        all_results["A3"] = {"skipped": True, "reason": "features_20 not provided"}

    # A4: Architecture comparison
    logger.info("\n[Ablation A4/5] Hard vs Soft CBM")
    all_results["A4"] = run_ablation_A4(features, labels, cfg, output_dir)

    # A5: Training modes / predictor complexity
    logger.info("\n[Ablation A5/5] Training modes / predictor complexity")
    all_results["A5"] = run_ablation_A5(features, labels, cfg, output_dir)

    elapsed = time.time() - t_total

    # Summary
    summary: Dict[str, Any] = {
        "_metadata": _make_metadata(
            model_name="all_ablations",
            n_concepts=12,
            concept_names=CONCEPT_NAMES_12,
        ),
        "total_elapsed_seconds": elapsed,
        "summary": {},
    }

    # A1, A2 have performance_delta; A4, A5 have different structure
    for key in ("A1", "A2"):
        res = all_results.get(key, {})
        pd = res.get("performance_delta", {})
        summary["summary"][key] = {
            "accuracy_delta": pd.get("accuracy", float("nan")),
            "macro_f1_delta": pd.get("macro_f1", float("nan")),
        }

    # A4: architectural comparison -- extract comparison_table
    a4_res = all_results.get("A4", {})
    a4_comparison = a4_res.get("comparison_table", {})
    if a4_comparison:
        summary["summary"]["A4"] = {
            "type": "architectural_comparison",
            "models": {
                arch: {
                    "accuracy_mean": vals.get("accuracy_mean", float("nan")),
                    "macro_f1_mean": vals.get("macro_f1_mean", float("nan")),
                }
                for arch, vals in a4_comparison.items()
            },
        }
    else:
        summary["summary"]["A4"] = {"skipped_or_empty": True}

    # A5: training modes + predictor complexity
    a5_res = all_results.get("A5", {})
    a5a_comp = a5_res.get("A5a_plan_a_predictor_complexity", {}).get("comparison", {})
    a5b_comp = a5_res.get("A5b_plan_b_training_modes", {}).get("comparison", {})
    if a5a_comp or a5b_comp:
        summary["summary"]["A5"] = {
            "type": "training_modes_and_predictor_complexity",
            "A5a_comparison": a5a_comp,
            "A5b_comparison": a5b_comp,
        }
    else:
        summary["summary"]["A5"] = {"skipped_or_empty": True}

    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Ablation summary saved to {summary_path}")

    logger.info(
        f"\nAll ablations complete in {elapsed:.1f}s. "
        f"Results in {output_dir}"
    )
    return all_results
