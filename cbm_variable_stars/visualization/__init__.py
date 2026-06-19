"""
CBM Variable Star Classification -- Visualization Pipeline

Exposes the public API for all visualization modules.

Modules
-------
plots           : Main plotting functions (confusion matrix, training curves,
                  ablation comparison, intervention curve, learning curve,
                  feature importance, class distribution)
concept_space   : Concept space visualization (t-SNE, UMAP, Bailey diagram,
                  per-class distributions)
latex_export    : LaTeX table generation (I5)
"""

from cbm_variable_stars.visualization.plots import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_ablation_comparison,
    plot_intervention_curve,
    plot_learning_curve,
    plot_feature_importance,
    plot_class_distribution,
    generate_standard_figures,
)
from cbm_variable_stars.visualization.concept_space import (
    plot_concept_tsne,
    plot_concept_umap,
    plot_bailey_diagram,
    plot_concept_distributions,
    plot_concept_radar,
)
from cbm_variable_stars.visualization.latex_export import (
    results_to_latex,
    confusion_matrix_to_latex,
    ablation_results_to_latex,
    per_class_results_to_latex,
    export_all_tables,
)

__all__ = [
    # plots
    "plot_confusion_matrix",
    "plot_training_curves",
    "plot_ablation_comparison",
    "plot_intervention_curve",
    "plot_learning_curve",
    "plot_feature_importance",
    "plot_class_distribution",
    "generate_standard_figures",
    # concept_space
    "plot_concept_tsne",
    "plot_concept_umap",
    "plot_bailey_diagram",
    "plot_concept_distributions",
    "plot_concept_radar",
    # latex_export
    "results_to_latex",
    "confusion_matrix_to_latex",
    "ablation_results_to_latex",
    "per_class_results_to_latex",
    "export_all_tables",
]
