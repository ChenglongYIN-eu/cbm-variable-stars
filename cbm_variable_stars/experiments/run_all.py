"""
CBM Variable Star Classification -- Main Experiment Orchestrator

Entry point that loads processed data, trains all models, and runs the
complete experiment suite:
    1. Model training (5-fold CV for all models)
    2. Ablation experiments A1-A5
    3. Concept intervention experiments
    4. Cross-survey validation (10dim + 12dim)
    5. Learning curve experiment
    6. Concept correlation analysis (I7)
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
    CONCEPT_NAMES_12,
    LEARNING_CURVE_SAMPLE_SIZES,
    N_CLASSES,
    RANDOM_SEED,
)
from cbm_variable_stars.shared.logger import logger, setup_logger
from cbm_variable_stars.shared.reproducibility import set_global_seed


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_features_labels(
    parquet_path: Path,
    concept_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load feature matrix and labels from a processed parquet file.

    Parameters
    ----------
    parquet_path : Path
        Path to a parquet file with concept columns + "label" column.
    concept_names : list of str or None
        Concept columns to extract.  Defaults to CONCEPT_NAMES_12.

    Returns
    -------
    (features, labels) as numpy arrays.
    """
    import pandas as pd

    if concept_names is None:
        concept_names = CONCEPT_NAMES_12

    df = pd.read_parquet(parquet_path)

    missing = [c for c in concept_names if c not in df.columns]
    if missing:
        raise ValueError(
            f"Parquet file {parquet_path} missing concept columns: {missing}"
        )

    features = df[concept_names].fillna(df[concept_names].median()).values.astype(
        np.float32
    )
    labels = df["label"].values.astype(np.int64)

    return features, labels


def _make_test_data_dict(
    parquet_path: Path,
    concept_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a dict {"features": np.ndarray, "labels": np.ndarray}."""
    features, labels = _load_features_labels(parquet_path, concept_names)
    return {"features": features, "labels": labels}


def _make_metadata_global(random_seed: int = RANDOM_SEED) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "concept_names": CONCEPT_NAMES_12,
        "class_names": CLASS_NAMES,
    }


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def _train_all_models(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: Any,
    results_dir: Path,
) -> Dict[str, Any]:
    """
    Train all CBM models and baselines with 5-fold CV.

    Models trained:
        - hard_cbm         (Plan A, MLP predictor)
        - hard_cbm_linear  (Plan A, linear predictor -- M1 fix)
        - hard_cbm_cal     (Plan B, calibrated)
        - soft_cbm
        - rf_baseline
        - xgb_baseline

    Returns
    -------
    dict: {model_name: cv_result_dict}
    """
    from cbm_variable_stars.training.cross_val import run_cross_validation
    from cbm_variable_stars.baselines.random_forest import train_random_forest
    from cbm_variable_stars.baselines.xgboost_model import train_xgboost

    seed = getattr(getattr(cfg, "project", cfg), "random_seed", RANDOM_SEED)
    hp = getattr(cfg, "training", None) or {}

    model_results: Dict[str, Any] = {}

    cbm_models = [
        "hard_cbm",
        "hard_cbm_linear",
        "hard_cbm_cal",
        "soft_cbm",
    ]

    for model_name in cbm_models:
        logger.info(f"\nTraining {model_name} (5-fold CV)...")
        try:
            result = run_cross_validation(
                features=features,
                labels=labels,
                model_name=model_name,
                model_kwargs={
                    "num_concepts": features.shape[1],
                    "num_classes": N_CLASSES,
                },
                batch_size=getattr(hp, "batch_size", 256),
                learning_rate=getattr(hp, "learning_rate", 1e-3),
                weight_decay=getattr(hp, "weight_decay", 1e-4),
                max_epochs=getattr(hp, "max_epochs", 200),
                patience=getattr(hp, "patience", 15),
                random_seed=seed,
                output_dir=str(results_dir),
            )
            model_results[model_name] = result
            agg = result.get("aggregated", {})
            logger.info(
                f"  {model_name}: "
                f"acc={agg.get('accuracy_mean', 'N/A'):.4f} "
                f"f1={agg.get('macro_f1_mean', 'N/A'):.4f}"
            )
        except Exception as e:
            logger.error(f"  {model_name} training failed: {e}")
            model_results[model_name] = {"error": str(e)}

    # RF baseline
    logger.info("\nTraining Random Forest baseline...")
    try:
        rf_result = train_random_forest(
            features=features,
            labels=labels,
            output_dir=str(results_dir / "rf_baseline"),
        )
        model_results["rf_baseline"] = rf_result
        agg = rf_result.get("aggregated", {})
        logger.info(
            f"  RF: acc={agg.get('accuracy_mean', 'N/A'):.4f} "
            f"f1={agg.get('macro_f1_mean', 'N/A'):.4f}"
        )
    except Exception as e:
        logger.error(f"  RF training failed: {e}")
        model_results["rf_baseline"] = {"error": str(e)}

    # XGBoost baseline
    logger.info("\nTraining XGBoost baseline...")
    try:
        xgb_result = train_xgboost(
            features=features,
            labels=labels,
            output_dir=str(results_dir / "xgb_baseline"),
        )
        model_results["xgb_baseline"] = xgb_result
        agg = xgb_result.get("aggregated", {})
        logger.info(
            f"  XGB: acc={agg.get('accuracy_mean', 'N/A'):.4f} "
            f"f1={agg.get('macro_f1_mean', 'N/A'):.4f}"
        )
    except Exception as e:
        logger.error(f"  XGB training failed: {e}")
        model_results["xgb_baseline"] = {"error": str(e)}

    return model_results


def _load_best_model(
    model_name: str,
    results_dir: Path,
    n_concepts: int = 12,
    device: str = "cpu",
) -> Any:
    """
    Load the best saved model checkpoint for a given model_name.
    Returns None if checkpoint not found.
    """
    import torch
    from cbm_variable_stars.models import create_model

    try:
        model = create_model(model_name, num_concepts=n_concepts, num_classes=N_CLASSES)
        ckpt_dir = results_dir / model_name / "checkpoints"
        ckpt_candidates = sorted(ckpt_dir.glob("best_model_fold*.pt"))
        if ckpt_candidates:
            ckpt = torch.load(
                ckpt_candidates[-1], map_location=device, weights_only=False
            )
            model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            model = model.to(device)
            model.eval()
            logger.info(f"Loaded {model_name} checkpoint: {ckpt_candidates[-1]}")
            return model
        else:
            logger.warning(f"No checkpoint found for {model_name} in {ckpt_dir}")
            return None
    except Exception as e:
        logger.error(f"Could not load {model_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# run_all_experiments
# ---------------------------------------------------------------------------

def run_all_experiments(
    cfg: Any,
    output_dir: Optional[str | Path] = None,
    device: str = "cpu",
    skip_training: bool = False,
    skip_ablation: bool = False,
    skip_intervention: bool = False,
    skip_cross_survey: bool = False,
    skip_learning_curve: bool = False,
    skip_correlation: bool = False,
) -> Dict[str, Any]:
    """
    Load data, train models, and run all experiments.

    Parameters
    ----------
    cfg : DictConfig
        Project configuration (loaded from configs/default.yaml).
    output_dir : str or Path or None
        Root results directory.  Defaults to cfg.paths.results or "results/".
    device : str
        Torch device ("cpu" or "cuda").
    skip_* : bool
        Skip individual experiment stages (useful for re-runs).

    Returns
    -------
    dict
        {
          "metadata": {...},
          "model_results": {...},
          "ablation": {...},
          "intervention": {...},
          "cross_survey": {...},
          "learning_curve": {...},
          "correlation": {...},
        }
    """
    # Setup
    seed = getattr(getattr(cfg, "project", cfg), "random_seed", RANDOM_SEED)
    set_global_seed(seed)

    if output_dir is None:
        output_dir = Path(
            getattr(getattr(cfg, "paths", cfg), "results", "results")
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    all_results: Dict[str, Any] = {
        "metadata": _make_metadata_global(seed),
    }

    logger.info("=" * 70)
    logger.info("CBM Variable Star Classification -- Full Experiment Suite")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 70)

    # ----- Load data -----
    processed_dir = Path(
        getattr(getattr(cfg, "paths", cfg), "processed", "data/processed")
    )

    cv_pool_path = processed_dir / "cv_pool.parquet"
    test_in_domain_path = processed_dir / "test_in_domain.parquet"
    test_cross_survey_path = processed_dir / "test_cross_survey.parquet"

    if not cv_pool_path.exists():
        raise FileNotFoundError(
            f"CV pool not found at {cv_pool_path}. "
            "Please run the data pipeline first (scripts/05_build_dataset.py)."
        )

    logger.info(f"\nLoading training data from {cv_pool_path}...")
    features, labels = _load_features_labels(cv_pool_path)
    logger.info(f"  CV pool: {features.shape[0]} samples, {features.shape[1]} concepts")

    test_data_id = None
    if test_in_domain_path.exists():
        test_data_id = _make_test_data_dict(test_in_domain_path)
        logger.info(
            f"  In-domain test: {test_data_id['features'].shape[0]} samples"
        )
    else:
        logger.warning(f"In-domain test set not found at {test_in_domain_path}")

    # ----- 1. Model Training -----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Model Training (5-fold CV)")
    logger.info("=" * 60)

    if not skip_training:
        t_stage = time.time()
        model_results = _train_all_models(features, labels, cfg, output_dir)
        all_results["model_results"] = model_results
        logger.info(f"Model training complete in {time.time() - t_stage:.1f}s")

        # Save model results summary
        with open(output_dir / "model_results.json", "w") as f:
            json.dump(model_results, f, indent=2, default=str)
    else:
        logger.info("Skipping model training (skip_training=True)")
        all_results["model_results"] = {}

    # ----- Pairwise significance testing with Holm-Bonferroni correction -----
    if not skip_training and len(all_results.get("model_results", {})) >= 2:
        from cbm_variable_stars.evaluation.significance import paired_cv_ttest, holm_bonferroni

        model_fold_scores: Dict[str, List[float]] = {}
        for mname, mres in all_results["model_results"].items():
            if isinstance(mres, dict) and "error" not in mres:
                fold_results = mres.get("fold_results", [])
                scores = []
                for fr in fold_results:
                    metrics = fr.get("metrics", fr)
                    val = metrics.get("val_macro_f1", metrics.get("macro_f1"))
                    if val is not None:
                        scores.append(float(val))
                if scores:
                    model_fold_scores[mname] = scores

        model_names_list = sorted(model_fold_scores.keys())
        pairwise_results: List[Dict[str, Any]] = []
        pairwise_pvals: List[tuple] = []

        for i in range(len(model_names_list)):
            for j in range(i + 1, len(model_names_list)):
                name_a = model_names_list[i]
                name_b = model_names_list[j]
                scores_a = model_fold_scores[name_a]
                scores_b = model_fold_scores[name_b]
                if len(scores_a) == len(scores_b) and len(scores_a) > 1:
                    result = paired_cv_ttest(
                        scores_a, scores_b,
                        model_a_name=name_a,
                        model_b_name=name_b,
                    )
                    comparison_name = f"{name_a}_vs_{name_b}"
                    pairwise_results.append(result)
                    pairwise_pvals.append((comparison_name, result["p_value"]))

        if pairwise_pvals:
            adjusted = holm_bonferroni(pairwise_pvals)
            all_results["significance"] = {
                "pairwise_tests": pairwise_results,
                "holm_bonferroni_correction": adjusted,
            }
            sig_path = output_dir / "significance_tests.json"
            with open(sig_path, "w") as f:
                json.dump(all_results["significance"], f, indent=2, default=str)
            logger.info(f"Significance tests saved to {sig_path}")

    # Load best model for experiments
    primary_model = _load_best_model("hard_cbm", output_dir, n_concepts=12, device=device)

    # ----- 2. Ablation Experiments -----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Ablation Experiments A1-A5")
    logger.info("=" * 60)

    if not skip_ablation:
        from cbm_variable_stars.experiments.ablation import run_all_ablations

        t_stage = time.time()
        ablation_dir = output_dir / "ablation"

        # Load 20-concept features if available
        features_20 = None
        cv_pool_20_path = processed_dir / "cv_pool_20concepts.parquet"
        if cv_pool_20_path.exists():
            try:
                from cbm_variable_stars.shared.constants import CONCEPT_NAMES_20
                features_20, _ = _load_features_labels(cv_pool_20_path, CONCEPT_NAMES_20)
                logger.info(f"  Loaded 20-concept features: {features_20.shape}")
            except Exception as e:
                logger.warning(f"  Could not load 20-concept features: {e}")

        ablation_results = run_all_ablations(
            features=features,
            labels=labels,
            cfg=cfg,
            output_dir=ablation_dir,
            features_20=features_20,
        )
        all_results["ablation"] = ablation_results
        logger.info(f"Ablation experiments complete in {time.time() - t_stage:.1f}s")
    else:
        logger.info("Skipping ablation experiments (skip_ablation=True)")
        all_results["ablation"] = {}

    # ----- 3. Intervention Experiments -----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Concept Intervention Experiments")
    logger.info("=" * 60)

    if not skip_intervention and test_data_id is not None and primary_model is not None:
        from cbm_variable_stars.experiments.intervention import run_all_interventions

        t_stage = time.time()
        intervention_dir = output_dir / "intervention"

        intervention_results = run_all_interventions(
            model=primary_model,
            test_data=test_data_id,
            cfg=cfg,
            output_dir=intervention_dir,
            device=device,
        )
        all_results["intervention"] = intervention_results
        logger.info(f"Intervention experiments complete in {time.time() - t_stage:.1f}s")
    else:
        if skip_intervention:
            logger.info("Skipping intervention experiments (skip_intervention=True)")
        elif test_data_id is None:
            logger.warning("Skipping intervention: no in-domain test data")
        elif primary_model is None:
            logger.warning("Skipping intervention: primary model not loaded")
        all_results["intervention"] = {}

    # ----- 4. Cross-Survey Validation -----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: Cross-Survey Validation")
    logger.info("=" * 60)

    if not skip_cross_survey:
        from cbm_variable_stars.experiments.cross_survey import run_cross_survey_experiment

        t_stage = time.time()
        cross_survey_dir = output_dir / "cross_survey"

        import pandas as pd
        ogle_test = None
        gaia_test = None

        if test_cross_survey_path.exists():
            ogle_test = pd.read_parquet(test_cross_survey_path)
            logger.info(f"  OGLE test: {len(ogle_test)} samples")
        else:
            logger.warning(
                f"OGLE cross-survey data not found at {test_cross_survey_path}"
            )

        if test_in_domain_path.exists():
            gaia_test = pd.read_parquet(test_in_domain_path)

        if ogle_test is not None and gaia_test is not None and primary_model is not None:
            cross_results = run_cross_survey_experiment(
                cfg=cfg,
                output_dir=cross_survey_dir,
                model=primary_model,
                gaia_test=gaia_test,
                ogle_test=ogle_test,
                device=device,
            )
            all_results["cross_survey"] = cross_results
            logger.info(
                f"Cross-survey experiments complete in {time.time() - t_stage:.1f}s"
            )
        else:
            logger.warning(
                "Skipping cross-survey: missing OGLE data or model"
            )
            all_results["cross_survey"] = {}
    else:
        logger.info("Skipping cross-survey (skip_cross_survey=True)")
        all_results["cross_survey"] = {}

    # ----- 5. Learning Curve -----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5: Learning Curve Experiment")
    logger.info("=" * 60)

    if not skip_learning_curve:
        from cbm_variable_stars.experiments.learning_curve import run_learning_curve

        t_stage = time.time()
        lc_dir = output_dir / "learning_curve"

        lc_result = run_learning_curve(
            features=features,
            labels=labels,
            model_name="hard_cbm",
            sample_sizes=LEARNING_CURVE_SAMPLE_SIZES,
            n_repeats=3,
            cfg=cfg,
            output_dir=lc_dir,
        )
        all_results["learning_curve"] = lc_result
        logger.info(f"Learning curve complete in {time.time() - t_stage:.1f}s")
    else:
        logger.info("Skipping learning curve (skip_learning_curve=True)")
        all_results["learning_curve"] = {}

    # ----- 6. Correlation Analysis -----
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 6: Concept Correlation Analysis (I7)")
    logger.info("=" * 60)

    if not skip_correlation:
        from cbm_variable_stars.experiments.correlation import run_correlation_analysis

        t_stage = time.time()
        corr_dir = output_dir / "correlation"

        # Use the full CV pool as features_df (DataFrame for convenience)
        import pandas as pd
        cv_pool_df = pd.read_parquet(cv_pool_path)

        corr_result = run_correlation_analysis(
            features_df=cv_pool_df,
            labels=labels,
            output_dir=corr_dir,
        )
        all_results["correlation"] = corr_result
        logger.info(
            f"Correlation analysis complete in {time.time() - t_stage:.1f}s"
        )
    else:
        logger.info("Skipping correlation analysis (skip_correlation=True)")
        all_results["correlation"] = {}

    # ----- Summary -----
    elapsed = time.time() - t_start
    all_results["total_elapsed_seconds"] = elapsed

    summary_path = output_dir / "experiment_summary.json"
    # Write a compact summary (exclude large list/array fields)
    compact = {
        "metadata": all_results["metadata"],
        "total_elapsed_seconds": elapsed,
        "stages_completed": [
            stage
            for stage in [
                "model_results",
                "ablation",
                "intervention",
                "cross_survey",
                "learning_curve",
                "correlation",
            ]
            if stage in all_results and all_results[stage]
        ],
    }

    with open(summary_path, "w") as f:
        json.dump(compact, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info(f"ALL EXPERIMENTS COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Results directory: {output_dir}")
    logger.info("=" * 70)

    return all_results
