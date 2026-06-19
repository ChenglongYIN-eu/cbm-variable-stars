"""
Result reporting and aggregation for CBM variable star classification.

Provides:
    - aggregate_cv_results:    Aggregate fold results to mean +/- std
    - save_results:            Save results with metadata and timestamp
    - format_results_table:    Pretty-print results in markdown or plain text
    - generate_comparison_table: Multi-model comparison in CSV + LaTeX
    - save_domain_comparison:   In-domain vs out-of-domain comparison
"""

import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from cbm_variable_stars.shared.constants import CLASS_NAMES


def aggregate_cv_results(
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate 5-fold CV results to mean +/- std.

    Handles both 'metrics.val_accuracy' (from Trainer.validate()) and
    'accuracy' (from compute_all_metrics()) key naming conventions.

    Args:
        fold_results: List of per-fold result dicts from run_cross_validation()

    Returns:
        dict with per-metric sub-dicts: {mean, std, min, max, values}
    """
    metrics_keys = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]

    aggregated: Dict[str, Any] = {}
    for key in metrics_keys:
        values: List[float] = []
        for r in fold_results:
            metrics = r.get("metrics", r)
            # Try val_{key} first (Trainer output), then {key} (compute_all_metrics output)
            val = metrics.get(f"val_{key}", metrics.get(key, None))
            if val is not None:
                values.append(float(val))

        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    return aggregated


def save_results(
    results: Dict[str, Any],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save experiment results to JSON with metadata and timestamp.

    Args:
        results:  Results dict to save
        path:     Output file path (.json)
        metadata: Optional metadata dict (model name, hyperparams, date, etc.)

    Returns:
        Absolute path of saved file as string.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2, default=str, ensure_ascii=False)

    return str(output_path.resolve())


def format_results_table(
    results: Dict[str, Any],
    format: str = "markdown",
) -> str:
    """
    Format results dict as a pretty-printed table string.

    Args:
        results: Results dict with 'aggregated' key (from run_cross_validation)
                 or direct metrics dict.
        format:  Output format: "markdown" or "plain"

    Returns:
        Formatted table string.
    """
    # Extract aggregated metrics
    agg = results.get("aggregated", results)
    model_name = results.get("model_name", "model")

    # Normalize flat keys (accuracy_mean -> {"accuracy": {"mean": ...}}) for display
    normalized: Dict[str, Dict[str, float]] = {}
    for key, val in agg.items():
        if isinstance(val, dict) and "mean" in val:
            normalized[key] = val
        elif isinstance(val, (int, float)) and key.endswith("_mean"):
            base = key[: -len("_mean")]
            if base not in normalized:
                normalized[base] = {}
            normalized[base]["mean"] = float(val)
            std_key = f"{base}_std"
            if std_key in agg:
                normalized[base]["std"] = float(agg[std_key])

    lines: List[str] = []

    if format == "markdown":
        lines.append(f"## Results: {model_name}")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")

        for metric_key, vals in normalized.items():
            mean_val = vals.get("mean", float("nan"))
            std_val = vals.get("std", float("nan"))
            lines.append(
                f"| {metric_key} | {mean_val:.4f} | {std_val:.4f} |"
            )
    elif format == "plain":
        lines.append(f"Results: {model_name}")
        lines.append("-" * 50)
        for metric_key, vals in normalized.items():
            mean_val = vals.get("mean", float("nan"))
            std_val = vals.get("std", float("nan"))
            lines.append(
                f"  {metric_key:<25} {mean_val:.4f} +/- {std_val:.4f}"
            )
    else:
        raise ValueError(f"Unknown format: '{format}'. Choose 'markdown' or 'plain'.")

    return "\n".join(lines)


def generate_comparison_table(
    all_results: Dict[str, Dict[str, Any]],
    output_path: str = "results/comparison_table.csv",
) -> str:
    """
    Generate a multi-model comparison table saved as CSV and LaTeX.

    Args:
        all_results:  Dict of {model_name: result_dict} from all trained models
        output_path:  Output CSV file path (LaTeX saved with .tex extension)

    Returns:
        Path to the saved CSV file as string.
    """
    rows: List[Dict[str, str]] = []

    for model_name, results in all_results.items():
        agg = results.get("aggregated", {})

        row: Dict[str, str] = {"Model": model_name}

        # Try both dict-of-dicts (aggregate_cv_results) and flat (cross_val aggregated)
        def get_stat(key: str) -> str:
            if key in agg and isinstance(agg[key], dict):
                return (
                    f"{agg[key]['mean']:.3f} +/- {agg[key]['std']:.3f}"
                )
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in agg:
                return f"{agg[mean_key]:.3f} +/- {agg.get(std_key, 0.0):.3f}"
            return "N/A"

        row["Accuracy"] = get_stat("accuracy")
        row["Macro F1"] = get_stat("macro_f1")
        row["Weighted F1"] = get_stat("weighted_f1")

        rows.append(row)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Write LaTeX
        latex = _generate_latex_table(rows)
        with open(path.with_suffix(".tex"), "w", encoding="utf-8") as f:
            f.write(latex)

    return str(path.resolve())


def _generate_latex_table(rows: List[Dict[str, str]]) -> str:
    """Generate a paper-ready LaTeX table in booktabs format."""
    if not rows:
        return ""

    cols = list(rows[0].keys())
    n_cols = len(cols)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Classification performance comparison (5-fold CV, mean $\pm$ std).}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "c" * (n_cols - 1) + r"}",
        r"\toprule",
        " & ".join([r"\textbf{" + c + "}" for c in cols]) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        values: List[str] = []
        for c in cols:
            val = row[c]
            if isinstance(val, str):
                val = val.replace("+/-", r"$\pm$")
            values.append(str(val))
        lines.append(" & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def save_domain_comparison(
    in_domain_results: Dict[str, Any],
    out_domain_results: Dict[str, Any],
    output_path: str = "results/domain_comparison.json",
) -> None:
    """
    Save in-domain vs out-of-domain performance comparison.

    Args:
        in_domain_results:  Evaluation results on Gaia in-domain test set
        out_domain_results: Evaluation results on OGLE out-of-domain test set
        output_path:        Output JSON file path
    """
    # Compute performance drop
    drop: Dict[str, float] = {}
    for key in ["accuracy", "macro_f1"]:
        in_val = in_domain_results.get("metrics", in_domain_results).get(key, 0.0)
        out_val = out_domain_results.get("metrics", out_domain_results).get(key, 0.0)
        drop[key] = float(out_val - in_val)  # negative = performance dropped

    comparison: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "in_domain": {
            "source": "Gaia DR3 hold-out (15%)",
            **in_domain_results,
        },
        "out_domain": {
            "source": "OGLE-IV (out-of-domain)",
            **out_domain_results,
        },
        "performance_drop": drop,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str, ensure_ascii=False)
