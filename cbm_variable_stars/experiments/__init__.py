"""
CBM Variable Star Classification -- Experiment Pipeline

Exposes the public API for all experiment modules.

Modules
-------
ablation        : Ablation experiments A1-A5
intervention    : Concept intervention and noise injection experiments
cross_survey    : Cross-survey (Gaia -> OGLE) validation
learning_curve  : Sample-size learning curve experiment
correlation     : Concept correlation analysis (I7)
run_all         : Main orchestrator (run_all_experiments)
"""

from cbm_variable_stars.experiments.ablation import (
    run_ablation_A1,
    run_ablation_A2,
    run_ablation_A3,
    run_ablation_A4,
    run_ablation_A5,
    run_all_ablations,
)
from cbm_variable_stars.experiments.intervention import (
    intervene_sequential_random,
    intervene_sequential_greedy,
    run_noise_injection_experiment,
    run_case_studies,
    run_all_interventions,
)
from cbm_variable_stars.experiments.cross_survey import (
    run_cross_survey_evaluation,
    run_cross_survey_experiment,
)
from cbm_variable_stars.experiments.learning_curve import (
    run_learning_curve,
    run_learning_curve_comparison,
)
from cbm_variable_stars.experiments.correlation import (
    compute_concept_correlation,
    compute_concept_class_association,
    compute_concept_mutual_information,
    run_correlation_analysis,
)
from cbm_variable_stars.experiments.run_all import run_all_experiments

__all__ = [
    # Ablation
    "run_ablation_A1",
    "run_ablation_A2",
    "run_ablation_A3",
    "run_ablation_A4",
    "run_ablation_A5",
    "run_all_ablations",
    # Intervention
    "intervene_sequential_random",
    "intervene_sequential_greedy",
    "run_noise_injection_experiment",
    "run_case_studies",
    "run_all_interventions",
    # Cross-survey
    "run_cross_survey_evaluation",
    "run_cross_survey_experiment",
    # Learning curve
    "run_learning_curve",
    "run_learning_curve_comparison",
    # Correlation
    "compute_concept_correlation",
    "compute_concept_class_association",
    "compute_concept_mutual_information",
    "run_correlation_analysis",
    # Orchestrator
    "run_all_experiments",
]
