"""
CBM Variable Star Classification -- One-Click Paper Figure/Table Generation

Entry point for regenerating all paper figures and LaTeX tables from
previously computed and saved results.

Usage:
    python -m cbm_variable_stars.paper.generate_all \\
        --results-dir results/ \\
        --output-dir paper/

    # Or from the project root:
    python scripts/08_generate_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_NAMES_12,
    RANDOM_SEED,
)
from cbm_variable_stars.shared.logger import logger, setup_logger
from cbm_variable_stars.shared.reproducibility import set_global_seed


# ---------------------------------------------------------------------------
# Helpers: load results JSON files
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None if not found."""
    if not path.exists():
        logger.debug(f"Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _load_features_labels(
    parquet_path: Path,
    concept_names: Optional[List[str]] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load feature matrix and labels from a parquet file."""
    if not parquet_path.exists():
        logger.warning(f"Parquet file not found: {parquet_path}")
        return None, None

    try:
        import pandas as pd

        if concept_names is None:
            concept_names = CONCEPT_NAMES_12

        df = pd.read_parquet(parquet_path)
        available = [c for c in concept_names if c in df.columns]
        if not available:
            logger.warning(f"No concept columns found in {parquet_path}")
            return None, None

        features = df[available].fillna(df[available].median()).values.astype(np.float32)
        labels = df["label"].values.astype(np.int64) if "label" in df.columns else None
        return features, labels
    except Exception as e:
        logger.warning(f"Could not load {parquet_path}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# generate_all_figures
# ---------------------------------------------------------------------------

def generate_all_figures(
    results_dir: Union[str, Path],
    output_dir: Union[str, Path],
    data_dir: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
    concept_names: Optional[List[str]] = None,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Path]:
    """
    Generate all paper figures from saved results.

    Parameters
    ----------
    results_dir : str or Path
        Root results directory containing subdirs:
            ablation/, intervention/, cross_survey/, learning_curve/,
            hard_cbm/cv_results.json, etc.
    output_dir : str or Path
        Where to save figures (PDF files).
    data_dir : str or Path or None
        Processed data directory (for t-SNE/UMAP/Bailey diagrams).
        Default: results_dir/../data/processed.
    class_names : list of str or None
    concept_names : list of str or None
    random_seed : int

    Returns
    -------
    dict
        {figure_name: Path to saved PDF}
    """
    from cbm_variable_stars.visualization.plots import (
        plot_ablation_comparison,
        plot_class_distribution,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_intervention_curve,
        plot_learning_curve,
        plot_training_curves,
    )
    from cbm_variable_stars.visualization.concept_space import (
        plot_bailey_diagram,
        plot_concept_distributions,
        plot_concept_tsne,
        plot_concept_umap,
    )

    set_global_seed(random_seed)

    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = CLASS_NAMES
    if concept_names is None:
        concept_names = CONCEPT_NAMES_12

    if data_dir is None:
        data_dir = results_dir.parent / "data" / "processed"
    data_dir = Path(data_dir)

    generated: Dict[str, Path] = {}

    logger.info("=" * 70)
    logger.info("Generating all paper figures")
    logger.info(f"  Results: {results_dir}")
    logger.info(f"  Output:  {output_dir}")
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Fig 1: Class distribution
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 1] Class distribution...")
    cv_pool_path = data_dir / "cv_pool.parquet"
    _, labels_pool = _load_features_labels(cv_pool_path)
    if labels_pool is not None:
        fig_path = output_dir / "fig01_class_distribution.pdf"
        try:
            plot_class_distribution(
                labels_pool,
                class_names=class_names,
                save_path=fig_path,
                title="Variable Star Class Distribution (CV Pool)",
            )
            generated["fig01_class_distribution"] = fig_path
        except Exception as e:
            logger.warning(f"  Fig 1 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 2: Confusion matrix
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 2] Confusion matrix...")
    cm_data = _load_json(results_dir / "hard_cbm" / "test_in_domain.json")
    if cm_data is None:
        cm_data = _load_json(results_dir / "hard_cbm" / "cv_results.json")

    if cm_data is not None:
        cm = (
            cm_data.get("confusion_matrix")
            or (cm_data.get("results") or {}).get("confusion_matrix")
            or (cm_data.get("aggregated") or {}).get("confusion_matrix_sum")
        )
        if cm is not None:
            fig_path = output_dir / "fig02_confusion_matrix.pdf"
            try:
                plot_confusion_matrix(
                    cm,
                    class_names=class_names,
                    title="HardCBM (Plan A) -- In-domain Test Set",
                    save_path=fig_path,
                    normalize=True,
                )
                generated["fig02_confusion_matrix"] = fig_path
            except Exception as e:
                logger.warning(f"  Fig 2 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 3: Training curves
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 3] Training curves...")
    cv_results = _load_json(results_dir / "hard_cbm" / "cv_results.json")
    if cv_results is not None:
        fold_results = cv_results.get("fold_results", [])
        if fold_results:
            # Use the history from the first fold
            history = fold_results[0].get("history") or fold_results[0].get("metrics", {}).get("history")
            if history:
                fig_path = output_dir / "fig03_training_curves.pdf"
                try:
                    plot_training_curves(
                        history,
                        save_path=fig_path,
                        title="HardCBM Training Curves (Fold 1)",
                    )
                    generated["fig03_training_curves"] = fig_path
                except Exception as e:
                    logger.warning(f"  Fig 3 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 4: Feature importance
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 4] Feature importance...")
    if cv_results is not None:
        agg = cv_results.get("aggregated", {})
        fi = agg.get("feature_importance_mean") or agg.get("concept_importance_mean")
        if fi is not None and len(fi) == len(concept_names):
            fig_path = output_dir / "fig04_feature_importance.pdf"
            try:
                plot_feature_importance(
                    fi,
                    concept_names=concept_names,
                    save_path=fig_path,
                    title="Concept Importance (HardCBM)",
                )
                generated["fig04_feature_importance"] = fig_path
            except Exception as e:
                logger.warning(f"  Fig 4 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 5: Ablation comparison
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 5] Ablation study comparison...")
    ablation_dir = results_dir / "ablation"

    # Build a simple results dict for the bar chart
    ablation_plot_data: Dict[str, Dict[str, float]] = {}

    # Full model baseline
    if cv_results is not None:
        agg = cv_results.get("aggregated", {})
        ablation_plot_data["Full (12 concepts)"] = {
            "macro_f1_mean": float(agg.get("macro_f1_mean", float("nan"))),
            "macro_f1_std": float(agg.get("macro_f1_std", float("nan"))),
            "accuracy_mean": float(agg.get("accuracy_mean", float("nan"))),
            "accuracy_std": float(agg.get("accuracy_std", float("nan"))),
        }

    for abl_key, label in [
        ("A1_remove_color.json", "No C11 (11 concepts)"),
        ("A2_minimal_concepts.json", "Minimal (4 concepts)"),
        ("A3_extended_20_concepts.json", "Extended (20 concepts)"),
    ]:
        abl_data = _load_json(ablation_dir / abl_key)
        if abl_data is None:
            continue

        # Get ablation model results
        abl_model = (
            abl_data.get("no_color_model")
            or abl_data.get("minimal_model")
            or abl_data.get("extended_20_model")
        )
        if abl_model and "results" in abl_model:
            agg = abl_model["results"]
            ablation_plot_data[label] = {
                "macro_f1_mean": float(agg.get("macro_f1_mean", float("nan"))),
                "macro_f1_std": float(agg.get("macro_f1_std", float("nan"))),
                "accuracy_mean": float(agg.get("accuracy_mean", float("nan"))),
                "accuracy_std": float(agg.get("accuracy_std", float("nan"))),
            }

    if len(ablation_plot_data) > 1:
        fig_path = output_dir / "fig05_ablation_comparison.pdf"
        try:
            plot_ablation_comparison(
                ablation_plot_data,
                save_path=fig_path,
                title="Ablation Study: Concept Subset Comparison",
                metric="macro_f1_mean",
            )
            generated["fig05_ablation_comparison"] = fig_path
        except Exception as e:
            logger.warning(f"  Fig 5 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 6: Intervention curves
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 6] Intervention curves...")
    intervention_dir = results_dir / "intervention"

    rand_data = _load_json(intervention_dir / "sequential_random.json")
    greedy_data = _load_json(intervention_dir / "sequential_greedy.json")

    if rand_data is not None or greedy_data is not None:
        combined = {}
        if rand_data:
            combined.update(rand_data)
        if greedy_data:
            combined["accuracies"] = greedy_data.get("accuracies", [])
            combined["concept_names_order"] = greedy_data.get("concept_names_order", [])

        fig_path = output_dir / "fig06_intervention_curve.pdf"
        try:
            plot_intervention_curve(
                combined,
                save_path=fig_path,
                title="Sequential Concept Intervention",
            )
            generated["fig06_intervention_curve"] = fig_path
        except Exception as e:
            logger.warning(f"  Fig 6 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 7: Learning curve
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 7] Learning curve...")
    lc_dir = results_dir / "learning_curve"
    lc_data = _load_json(lc_dir / "learning_curve.json")

    if lc_data is not None:
        fig_path = output_dir / "fig07_learning_curve.pdf"
        try:
            plot_learning_curve(
                lc_data,
                save_path=fig_path,
                title="Learning Curve (HardCBM)",
            )
            generated["fig07_learning_curve"] = fig_path
        except Exception as e:
            logger.warning(f"  Fig 7 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 8: Concept space t-SNE
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 8] t-SNE concept space...")
    features_cv, labels_cv = _load_features_labels(cv_pool_path)
    if features_cv is not None and labels_cv is not None:
        fig_path = output_dir / "fig08_concept_tsne.pdf"
        try:
            plot_concept_tsne(
                features_cv,
                labels_cv,
                class_names=class_names,
                save_path=fig_path,
                title="Concept Space (t-SNE)",
                max_samples=5000,
                random_state=random_seed,
            )
            generated["fig08_concept_tsne"] = fig_path
        except Exception as e:
            logger.warning(f"  Fig 8 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 9: UMAP
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 9] UMAP concept space...")
    if features_cv is not None and labels_cv is not None:
        fig_path = output_dir / "fig09_concept_umap.pdf"
        try:
            plot_concept_umap(
                features_cv,
                labels_cv,
                class_names=class_names,
                save_path=fig_path,
                title="Concept Space (UMAP)",
                max_samples=10000,
                random_state=random_seed,
            )
            generated["fig09_concept_umap"] = fig_path
        except Exception as e:
            logger.warning(f"  Fig 9 (UMAP) failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 10: Bailey diagram
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 10] Bailey diagram...")
    if features_cv is not None and labels_cv is not None:
        fig_path = output_dir / "fig10_bailey_diagram.pdf"
        try:
            plot_bailey_diagram(
                features_cv,
                labels_cv,
                class_names=class_names,
                concept_names=concept_names,
                save_path=fig_path,
                title="Bailey Diagram: Period–Amplitude",
            )
            generated["fig10_bailey_diagram"] = fig_path
        except Exception as e:
            logger.warning(f"  Fig 10 failed: {e}")

    # -----------------------------------------------------------------------
    # Fig 11: Per-class concept distributions
    # -----------------------------------------------------------------------
    logger.info("\n[Fig 11] Per-class concept distributions...")
    if features_cv is not None and labels_cv is not None:
        fig_path = output_dir / "fig11_concept_distributions.pdf"
        try:
            plot_concept_distributions(
                features_cv,
                labels_cv,
                concept_names=concept_names,
                class_names=class_names,
                save_path=fig_path,
                title="Per-Class Concept Distributions",
                plot_type="violin",
            )
            generated["fig11_concept_distributions"] = fig_path
        except Exception as e:
            logger.warning(f"  Fig 11 failed: {e}")

    logger.info(
        f"\nGenerated {len(generated)}/{11} figures in {output_dir}"
    )
    for name, path in generated.items():
        logger.info(f"  {name}: {path}")

    return generated


# ---------------------------------------------------------------------------
# generate_all_tables
# ---------------------------------------------------------------------------

def generate_all_tables(
    results_dir: Union[str, Path],
    output_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Generate all LaTeX tables from saved results.

    Parameters
    ----------
    results_dir : str or Path
    output_dir : str or Path
    class_names : list of str or None

    Returns
    -------
    dict
        {table_name: latex_string}
    """
    from cbm_variable_stars.visualization.latex_export import export_all_tables

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Generating all LaTeX tables")
    logger.info(f"  Results: {results_dir}")
    logger.info(f"  Output:  {output_dir}")
    logger.info("=" * 70)

    tables = export_all_tables(
        results_dir=results_dir,
        output_dir=output_dir,
        class_names=class_names,
    )

    logger.info(f"\nGenerated {len(tables)} LaTeX tables in {output_dir}")
    return tables


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Command-line entry point for generating paper figures and tables.

    Usage
    -----
    python -m cbm_variable_stars.paper.generate_all \\
        --results-dir results/ \\
        --output-dir paper/ \\
        --data-dir data/processed/ \\
        [--figures-only | --tables-only]
    """
    parser = argparse.ArgumentParser(
        description="Generate all paper figures and LaTeX tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root results directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper"),
        help="Output directory for figures and tables.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Processed data directory (for concept space visualizations).",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only generate figures, skip tables.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only generate tables, skip figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible t-SNE/UMAP.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    setup_logger(log_level=args.log_level)
    set_global_seed(args.seed)

    results_dir = args.results_dir
    output_dir = args.output_dir

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CBM Variable Star Classification -- Paper Generation")
    logger.info(f"  Results dir: {results_dir.resolve()}")
    logger.info(f"  Output dir:  {output_dir.resolve()}")
    logger.info("=" * 70)

    generated_items: Dict[str, Any] = {}

    if not args.tables_only:
        figures_dir = output_dir / "figures"
        figures = generate_all_figures(
            results_dir=results_dir,
            output_dir=figures_dir,
            data_dir=args.data_dir,
            random_seed=args.seed,
        )
        generated_items["figures"] = {k: str(v) for k, v in figures.items()}
        logger.info(f"\n{len(figures)} figures saved to {figures_dir}")

    if not args.figures_only:
        tables = generate_all_tables(
            results_dir=results_dir,
            output_dir=output_dir,
        )
        generated_items["tables"] = list(tables.keys())
        logger.info(f"\n{len(tables)} tables saved to {output_dir / 'tables'}")

    # Write manifest
    manifest_path = output_dir / "generation_manifest.json"
    import json as _json
    with open(manifest_path, "w") as f:
        _json.dump(generated_items, f, indent=2, default=str)

    logger.info(f"\nManifest saved to {manifest_path}")
    logger.info("Paper generation complete.")


if __name__ == "__main__":
    main()
