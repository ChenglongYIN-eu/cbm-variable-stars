"""
CBM Variable Star Classification -- Script 08: Generate Figures and Tables

Generates all publication-quality figures (PDF) and LaTeX tables from
the experiment results produced by script 07_run_experiments.py.

Output structure:
    paper/
    ├── figures/
    │   ├── fig01_class_distribution.pdf
    │   ├── fig02_confusion_matrix.pdf
    │   ├── fig03_training_curves.pdf
    │   ├── fig04_feature_importance.pdf
    │   ├── fig05_ablation_comparison.pdf
    │   ├── fig06_intervention_curve.pdf
    │   ├── fig07_learning_curve.pdf
    │   ├── fig08_concept_tsne.pdf
    │   ├── fig09_concept_umap.pdf
    │   ├── fig10_bailey_diagram.pdf
    │   └── fig11_concept_distributions.pdf
    └── tables/
        ├── tab_model_comparison.tex
        ├── tab_confusion_matrix.tex
        ├── tab_ablation.tex
        └── tab_cross_survey_*.tex

Prerequisites:
    - Script 07 completed (results/ directory populated)
    - cbm_variable_stars package installed (pip install -e .)

Usage:
    python scripts/08_generate_figures.py
    python scripts/08_generate_figures.py --results-dir results/ --output-dir paper/
    python scripts/08_generate_figures.py --figures-only
    python scripts/08_generate_figures.py --tables-only
    python scripts/08_generate_figures.py --help
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is in sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm_variable_stars.shared.constants import RANDOM_SEED
from cbm_variable_stars.shared.logger import logger, setup_logger
from cbm_variable_stars.shared.reproducibility import set_global_seed
from cbm_variable_stars.paper.generate_all import (
    generate_all_figures,
    generate_all_tables,
    main as _main_entry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all paper figures and LaTeX tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Root results directory (from script 07).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper",
        help="Output directory for figures and tables.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Processed data directory. Defaults to results-dir/../data/processed. "
            "Required for concept space visualizations (t-SNE, UMAP, Bailey diagram)."
        ),
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only generate figures (PDF), skip LaTeX tables.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only generate LaTeX tables, skip figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible t-SNE/UMAP visualizations.",
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

    setup_logger(log_level=args.log_level)
    set_global_seed(args.seed)

    logger.info("=" * 70)
    logger.info("CBM Variable Star Classification")
    logger.info("Script 08: Generate Figures and Tables")
    logger.info(f"  Results dir: {args.results_dir}")
    logger.info(f"  Output dir:  {args.output_dir}")
    logger.info(f"  Seed:        {args.seed}")
    logger.info("=" * 70)

    results_dir = args.results_dir
    output_dir = args.output_dir

    if not results_dir.exists():
        logger.error(
            f"Results directory not found: {results_dir}\n"
            "Please run script 07 first:\n"
            "  python scripts/07_run_experiments.py"
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine data directory
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "processed"
        if not data_dir.exists():
            data_dir = results_dir.parent / "data" / "processed"
    if not data_dir.exists():
        logger.warning(
            f"Data directory not found: {data_dir}. "
            "Concept space visualizations (t-SNE, UMAP, Bailey) will be skipped."
        )
        data_dir = None

    generated_figures = {}
    generated_tables = {}

    # ---- Figures ----
    if not args.tables_only:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nGenerating figures -> {figures_dir}")
        try:
            generated_figures = generate_all_figures(
                results_dir=results_dir,
                output_dir=figures_dir,
                data_dir=data_dir,
                random_seed=args.seed,
            )
            logger.info(f"\n{len(generated_figures)} figures generated.")
        except Exception as e:
            logger.error(f"Figure generation failed: {e}", exc_info=True)

    # ---- Tables ----
    if not args.figures_only:
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nGenerating LaTeX tables -> {tables_dir}")
        try:
            generated_tables = generate_all_tables(
                results_dir=results_dir,
                output_dir=output_dir,  # export_all_tables adds /tables subdir
            )
            logger.info(f"\n{len(generated_tables)} tables generated.")
        except Exception as e:
            logger.error(f"Table generation failed: {e}", exc_info=True)

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("GENERATION COMPLETE")

    if generated_figures:
        logger.info(f"\nFigures ({len(generated_figures)}):")
        for name, path in generated_figures.items():
            logger.info(f"  {name}: {path}")

    if generated_tables:
        logger.info(f"\nTables ({len(generated_tables)}):")
        for name in generated_tables:
            logger.info(f"  {name}")

    logger.info(f"\nOutput directory: {output_dir.resolve()}")
    logger.info("=" * 70)

    if not generated_figures and not generated_tables:
        logger.warning(
            "No outputs were generated. Check that results/ is populated "
            "and that required dependencies are installed."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
