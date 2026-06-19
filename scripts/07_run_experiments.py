"""
CBM Variable Star Classification -- Script 07: Run All Experiments

Runs the complete experiment suite:
    1. Train all models (hard_cbm, hard_cbm_linear, hard_cbm_cal, soft_cbm,
       rf_baseline, xgb_baseline) using 5-fold CV.
    2. Ablation experiments A1-A5.
    3. Concept intervention experiments (sequential random, greedy, noise injection,
       case studies).
    4. Cross-survey validation (OGLE, 10dim + 12dim_with_match modes).
    5. Learning curve experiment.
    6. Concept correlation analysis (I7).

Prerequisites:
    - scripts/01-06 completed (data downloaded, features extracted, dataset built)
    - cbm_variable_stars package installed (pip install -e .)
    - data/processed/ contains: cv_pool.parquet, test_in_domain.parquet

Usage:
    python scripts/07_run_experiments.py
    python scripts/07_run_experiments.py --config configs/default.yaml
    python scripts/07_run_experiments.py --device cuda
    python scripts/07_run_experiments.py --skip-training --skip-ablation
    python scripts/07_run_experiments.py --help
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is in sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.config import load_config
from cbm_variable_stars.shared.logger import logger, setup_logger
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.experiments.run_all import run_all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all CBM variable star classification experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Path to the project configuration YAML file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for saving results. "
            "Defaults to cfg.paths.results (usually 'results/')."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device for neural network training.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (assume checkpoints already exist).",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip ablation experiments A1-A5.",
    )
    parser.add_argument(
        "--skip-intervention",
        action="store_true",
        help="Skip concept intervention experiments.",
    )
    parser.add_argument(
        "--skip-cross-survey",
        action="store_true",
        help="Skip cross-survey validation experiment.",
    )
    parser.add_argument(
        "--skip-learning-curve",
        action="store_true",
        help="Skip learning curve experiment.",
    )
    parser.add_argument(
        "--skip-correlation",
        action="store_true",
        help="Skip concept correlation analysis.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    setup_logger(log_level=args.log_level)

    logger.info("=" * 70)
    logger.info("CBM Variable Star Classification")
    logger.info("Script 07: Run All Experiments")
    logger.info(f"  Config:     {args.config}")
    logger.info(f"  Device:     {args.device}")
    logger.info(f"  Results dir:{args.results_dir or '(from config)'}")
    logger.info("=" * 70)

    # Load config
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    cfg = load_config(str(args.config))

    # Set random seed
    seed = getattr(getattr(cfg, "project", cfg), "random_seed", 42)
    set_global_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Check device availability
    device = args.device
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
    elif device == "mps":
        import torch
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = "cpu"

    logger.info(f"Using device: {device}")

    # Determine results directory
    output_dir = args.results_dir
    if output_dir is None:
        output_dir = Path(
            getattr(getattr(cfg, "paths", cfg), "results", "results")
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir.resolve()}")

    # Check data prerequisites
    processed_dir = Path(
        getattr(getattr(cfg, "paths", cfg), "processed", "data/processed")
    )
    cv_pool_path = processed_dir / "cv_pool.parquet"
    if not cv_pool_path.exists():
        logger.error(
            f"CV pool not found at {cv_pool_path}.\n"
            "Please run the data pipeline first:\n"
            "  python scripts/05_build_dataset.py"
        )
        sys.exit(1)

    logger.info(f"CV pool found: {cv_pool_path}")

    # Run all experiments
    try:
        results = run_all_experiments(
            cfg=cfg,
            output_dir=output_dir,
            device=device,
            skip_training=args.skip_training,
            skip_ablation=args.skip_ablation,
            skip_intervention=args.skip_intervention,
            skip_cross_survey=args.skip_cross_survey,
            skip_learning_curve=args.skip_learning_curve,
            skip_correlation=args.skip_correlation,
        )

        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
        logger.info(f"  Total time: {results.get('total_elapsed_seconds', 0):.1f}s")
        logger.info(f"  Results:    {output_dir.resolve()}")
        logger.info("=" * 70)
        logger.info("\nNext step: python scripts/08_generate_figures.py")

    except KeyboardInterrupt:
        logger.warning("\nExperiment suite interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Experiment suite failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
