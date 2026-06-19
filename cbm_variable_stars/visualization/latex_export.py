"""
CBM Variable Star Classification -- LaTeX Table Generation (I5)

Generates publication-ready LaTeX tables for the paper.

Functions:
    results_to_latex            -- Model comparison table
    confusion_matrix_to_latex   -- Confusion matrix as LaTeX booktabs table
    ablation_results_to_latex   -- Ablation study comparison table
    export_all_tables           -- Export all tables from results directory
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from cbm_variable_stars.shared.constants import CLASS_NAMES, CONCEPT_NAMES_12
from cbm_variable_stars.shared.logger import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_mean_std(
    mean: float,
    std: float,
    decimals: int = 3,
    bold_threshold: Optional[float] = None,
    bold_higher: bool = True,
) -> str:
    """
    Format a mean±std pair as LaTeX string.

    Example: '$0.943 \\pm 0.008$'
    Bold if mean exceeds bold_threshold (or is below, if bold_higher=False).
    """
    if np.isnan(mean):
        return "---"

    val_str = f"{mean:.{decimals}f}"
    std_str = f"{std:.{decimals}f}" if not np.isnan(std) else "---"

    cell = f"${val_str} \\pm {std_str}$"

    if bold_threshold is not None:
        condition = mean >= bold_threshold if bold_higher else mean <= bold_threshold
        if condition:
            cell = f"\\textbf{{{cell}}}"

    return cell


_LATEX_SPECIAL = {
    '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
    '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
}
_LATEX_PATTERN = re.compile('|'.join(re.escape(k) for k in _LATEX_SPECIAL))


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters while preserving LaTeX math environments.

    [Min6 FIX] Text inside $...$ delimiters is left untouched so that
    math expressions like $R_{31}$ are not corrupted by escaping.
    """
    text = str(text)
    # Split on math delimiters, preserving them
    parts = re.split(r'(\$[^$]*\$)', text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Inside $...$: preserve as-is
            result.append(part)
        else:
            # Outside math: escape special characters
            result.append(_LATEX_PATTERN.sub(
                lambda m: _LATEX_SPECIAL[m.group()], part
            ))
    return ''.join(result)


def _booktabs_table(
    header_row: List[str],
    data_rows: List[List[str]],
    caption: str,
    label: str,
    column_format: Optional[str] = None,
    position: str = "t",
) -> str:
    """
    Build a booktabs LaTeX table string.

    Parameters
    ----------
    header_row : list of str
        Column headers.
    data_rows : list of list of str
        Each inner list is one table row.
    caption : str
    label : str
    column_format : str or None
        LaTeX column spec (e.g. "lcccc"). Auto-generated if None.
    position : str
        Table float position.

    Returns
    -------
    str
        Complete LaTeX table environment string.
    """
    n_cols = len(header_row)
    if column_format is None:
        column_format = "l" + "c" * (n_cols - 1)

    lines = [
        f"\\begin{{table}}[{position}]",
        "  \\centering",
        f"  \\caption{{{_escape_latex(caption)}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{column_format}}}",
        "    \\toprule",
    ]

    # Header
    header_str = " & ".join(header_row) + " \\\\"
    lines.append(f"    {header_str}")
    lines.append("    \\midrule")

    # Data rows
    for row in data_rows:
        row_str = " & ".join(row) + " \\\\"
        lines.append(f"    {row_str}")

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# results_to_latex
# ---------------------------------------------------------------------------

def results_to_latex(
    results: Dict[str, Any],
    model_names: Optional[List[str]] = None,
    metric_names: Optional[List[str]] = None,
    caption: str = "Classification performance comparison (5-fold CV, mean $\\pm$ std).",
    label: str = "tab:model_comparison",
    bold_best: bool = True,
) -> str:
    """
    Generate a LaTeX model comparison table from CV results.

    Parameters
    ----------
    results : dict
        {model_name: {"aggregated": {"accuracy_mean": ..., "accuracy_std": ..., ...}}}
        or the direct "aggregated" dict if a single model.
    model_names : list of str or None
        Subset of model names to include. All if None.
    metric_names : list of str or None
        Metrics to include. Defaults to ["accuracy", "macro_f1"].
    caption : str
    label : str
    bold_best : bool
        Bold the best value in each metric column.

    Returns
    -------
    str
        LaTeX table string.
    """
    if metric_names is None:
        metric_names = ["accuracy", "macro_f1"]

    if model_names is None:
        model_names = list(results.keys())

    # Pretty-print metric names
    metric_labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "macro_precision": "Macro Prec.",
        "macro_recall": "Macro Rec.",
    }

    header = ["Model"] + [metric_labels.get(m, m.replace("_", " ").title()) for m in metric_names]

    # Extract values
    all_vals: Dict[str, Dict[str, float]] = {}
    for model_name in model_names:
        res = results.get(model_name, {})
        if "aggregated" in res:
            agg = res["aggregated"]
        else:
            agg = res

        row_vals: Dict[str, float] = {}
        for metric in metric_names:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            row_vals[f"{metric}_mean"] = float(agg.get(mean_key, float("nan")))
            row_vals[f"{metric}_std"] = float(agg.get(std_key, float("nan")))

        all_vals[model_name] = row_vals

    # Find best values per metric for bolding
    best_vals: Dict[str, float] = {}
    if bold_best:
        for metric in metric_names:
            vals = [
                all_vals[m][f"{metric}_mean"]
                for m in model_names
                if not np.isnan(all_vals[m][f"{metric}_mean"])
            ]
            if vals:
                best_vals[metric] = max(vals)

    # Build data rows
    # Pretty model name mapping
    model_name_map = {
        "hard_cbm": "HardCBM (Plan A)",
        "hard_cbm_linear": "HardCBM-Linear",
        "hard_cbm_cal": "HardCBM-Cal. (Plan B)",
        "soft_cbm": "SoftCBM",
        "cem": "CEM",
        "rf_baseline": "Random Forest",
        "xgb_baseline": "XGBoost",
        "mlp_baseline": "MLP Baseline",
    }

    data_rows = []
    for model_name in model_names:
        row_vals = all_vals[model_name]
        row = [model_name_map.get(model_name, model_name)]

        for metric in metric_names:
            mean = row_vals[f"{metric}_mean"]
            std = row_vals[f"{metric}_std"]
            bold_thresh = best_vals.get(metric) if bold_best else None
            row.append(_fmt_mean_std(mean, std, bold_threshold=bold_thresh))

        data_rows.append(row)

    return _booktabs_table(header, data_rows, caption, label)


# ---------------------------------------------------------------------------
# confusion_matrix_to_latex
# ---------------------------------------------------------------------------

def confusion_matrix_to_latex(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: Optional[List[str]] = None,
    caption: str = "Confusion matrix on the hold-out in-domain test set.",
    label: str = "tab:confusion_matrix",
    normalize: bool = True,
    decimals: int = 3,
) -> str:
    """
    Generate a LaTeX confusion matrix table (booktabs style).

    Parameters
    ----------
    cm : array-like, shape (n_classes, n_classes)
        Raw count confusion matrix.
    class_names : list of str or None
    caption : str
    label : str
    normalize : bool
        If True, normalize rows (show recall per class).
    decimals : int
        Decimal places for normalized values.

    Returns
    -------
    str
        LaTeX table string.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm = np.array(cm)
    n_classes = len(class_names)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(float) / row_sums
        fmt_fn = lambda v: f"{v:.{decimals}f}"
    else:
        cm_display = cm.astype(int)
        fmt_fn = lambda v: f"{int(v)}"

    short_names = [cn.replace("_", "-") for cn in class_names]

    # Header: "True / Pred" + class names
    header = ["True $\\backslash$ Pred"] + short_names

    data_rows = []
    for i, class_name in enumerate(class_names):
        row = [short_names[i]]
        for j in range(n_classes):
            val = float(cm_display[i, j])
            cell = fmt_fn(val)
            # Bold diagonal (correct predictions)
            if i == j:
                cell = f"\\textbf{{{cell}}}"
            row.append(cell)
        data_rows.append(row)

    n_cols = n_classes + 1
    col_fmt = "l" + "c" * n_classes

    return _booktabs_table(header, data_rows, caption, label, column_format=col_fmt)


# ---------------------------------------------------------------------------
# ablation_results_to_latex
# ---------------------------------------------------------------------------

def ablation_results_to_latex(
    results: Dict[str, Any],
    caption: str = "Ablation study results (5-fold CV macro F1, mean $\\pm$ std).",
    label: str = "tab:ablation",
    bold_best: bool = True,
) -> str:
    """
    Generate a LaTeX ablation comparison table.

    Parameters
    ----------
    results : dict
        {ablation_id: {"description": str, "performance_delta": {...},
         "no_color_model" / "minimal_model" / "extended_20_model": {...}}}
        Or: {model_variant_name: {"macro_f1_mean": ..., "macro_f1_std": ...}}
    caption : str
    label : str
    bold_best : bool

    Returns
    -------
    str
        LaTeX table string.
    """
    header = [
        "Ablation",
        "Description",
        "Accuracy (mean $\\pm$ std)",
        "Macro F1 (mean $\\pm$ std)",
        "$\\Delta$ F1",
    ]

    rows_data: List[Dict[str, Any]] = []

    for key, res in results.items():
        if not isinstance(res, dict) or res.get("skipped"):
            continue

        desc = res.get("description", key)
        abl_type = res.get("ablation_type", key)

        # Try to extract model results
        # Different structures depending on A1/A2/A3/A4
        if "no_color_model" in res:
            agg = res["no_color_model"].get("results", {})
            delta = res.get("performance_delta", {}).get("macro_f1", float("nan"))
        elif "minimal_model" in res:
            agg = res["minimal_model"].get("results", {})
            delta = res.get("performance_delta", {}).get("macro_f1", float("nan"))
        elif "extended_20_model" in res:
            agg = res["extended_20_model"].get("results", {})
            delta = res.get("performance_delta", {}).get("macro_f1", float("nan"))
        elif "comparison_table" in res:
            # A4: multiple models
            for arch_name, arch_res in res["comparison_table"].items():
                rows_data.append({
                    "key": f"{key}/{arch_name}",
                    "desc": f"{arch_name}",
                    "acc_mean": arch_res.get("accuracy_mean", float("nan")),
                    "acc_std": float("nan"),
                    "f1_mean": arch_res.get("macro_f1_mean", float("nan")),
                    "f1_std": float("nan"),
                    "delta": float("nan"),
                })
            continue
        else:
            agg = res.get("results", {})
            delta = res.get("performance_delta", {}).get("macro_f1", float("nan"))

        rows_data.append({
            "key": abl_type,
            "desc": desc[:60] if len(desc) > 60 else desc,
            "acc_mean": float(agg.get("accuracy_mean", float("nan"))),
            "acc_std": float(agg.get("accuracy_std", float("nan"))),
            "f1_mean": float(agg.get("macro_f1_mean", float("nan"))),
            "f1_std": float(agg.get("macro_f1_std", float("nan"))),
            "delta": float(delta),
        })

    if not rows_data:
        logger.warning("ablation_results_to_latex: no valid rows found.")
        return "% No ablation results to display."

    # Find best F1 for bolding
    best_f1 = float("nan")
    if bold_best:
        f1_vals = [r["f1_mean"] for r in rows_data if not np.isnan(r["f1_mean"])]
        if f1_vals:
            best_f1 = max(f1_vals)

    data_rows = []
    for r in rows_data:
        delta_str = (
            f"{r['delta']:+.3f}" if not np.isnan(r["delta"]) else "---"
        )
        row = [
            _escape_latex(r["key"]),
            _escape_latex(r["desc"]),
            _fmt_mean_std(r["acc_mean"], r["acc_std"]),
            _fmt_mean_std(
                r["f1_mean"], r["f1_std"],
                bold_threshold=best_f1 if bold_best else None,
            ),
            f"${delta_str}$" if delta_str != "---" else "---",
        ]
        data_rows.append(row)

    col_fmt = "llccc"
    return _booktabs_table(header, data_rows, caption, label, column_format=col_fmt)


# ---------------------------------------------------------------------------
# per_class_results_to_latex
# ---------------------------------------------------------------------------

def per_class_results_to_latex(
    per_class: Dict[str, Dict[str, Any]],
    model_name: str = "",
    caption: str = "Per-class classification metrics.",
    label: str = "tab:per_class",
    decimals: int = 3,
) -> str:
    """
    Generate per-class precision/recall/F1/support table.

    Parameters
    ----------
    per_class : dict
        {class_name: {"f1": float, "precision": float, "recall": float, "support": int}}
    model_name : str
    caption : str
    label : str
    decimals : int

    Returns
    -------
    str
    """
    header = ["Class", "Precision", "Recall", "F1", "Support"]

    data_rows = []
    for class_name, metrics in per_class.items():
        f1 = metrics.get("f1", float("nan"))
        prec = metrics.get("precision", float("nan"))
        rec = metrics.get("recall", float("nan"))
        sup = metrics.get("support", 0)

        row = [
            _escape_latex(class_name),
            f"{prec:.{decimals}f}" if not np.isnan(prec) else "---",
            f"{rec:.{decimals}f}" if not np.isnan(rec) else "---",
            f"{f1:.{decimals}f}" if not np.isnan(f1) else "---",
            f"{int(sup):,}",
        ]
        data_rows.append(row)

    if model_name:
        caption = f"{caption} Model: {_escape_latex(model_name)}."

    return _booktabs_table(header, data_rows, caption, label, column_format="lcccc")


# ---------------------------------------------------------------------------
# export_all_tables
# ---------------------------------------------------------------------------

def export_all_tables(
    results_dir: str | Path,
    output_dir: str | Path = "paper/tables",
    class_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Export all LaTeX tables from saved results JSON files.

    Reads from results_dir and writes .tex files to output_dir.

    Parameters
    ----------
    results_dir : str or Path
        Root results directory (must contain subdirs like ablation/, cv_results.json).
    output_dir : str or Path
        Where to write .tex files.
    class_names : list of str or None

    Returns
    -------
    dict
        {table_name: latex_string}
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = CLASS_NAMES

    all_tables: Dict[str, str] = {}

    # --- Table 1: Model comparison ---
    model_results: Dict[str, Any] = {}
    model_names_to_check = [
        "hard_cbm", "hard_cbm_linear", "hard_cbm_cal", "soft_cbm",
        "rf_baseline", "xgb_baseline",
    ]
    for model_name in model_names_to_check:
        cv_path = results_dir / model_name / "cv_results.json"
        if cv_path.exists():
            with open(cv_path) as f:
                model_results[model_name] = json.load(f)

    if model_results:
        table_model = results_to_latex(
            model_results,
            caption=(
                "Classification performance on Gaia DR3 in-domain test set "
                "(5-fold CV, mean $\\pm$ std)."
            ),
            label="tab:model_comparison",
        )
        all_tables["model_comparison"] = table_model
        out_path = output_dir / "tab_model_comparison.tex"
        with open(out_path, "w") as f:
            f.write(table_model)
        logger.info(f"Saved: {out_path}")

    # --- Table 2: Ablation results ---
    ablation_results: Dict[str, Any] = {}
    for ablation_key in ("A1", "A2", "A3", "A4", "A5"):
        json_files = list((results_dir / "ablation").glob(f"*{ablation_key}*.json"))
        for jf in json_files:
            with open(jf) as f:
                ablation_results[ablation_key] = json.load(f)
            break

    if ablation_results:
        table_abl = ablation_results_to_latex(
            ablation_results,
            caption="Ablation study: effect of concept subset on classification performance.",
            label="tab:ablation",
        )
        all_tables["ablation"] = table_abl
        out_path = output_dir / "tab_ablation.tex"
        with open(out_path, "w") as f:
            f.write(table_abl)
        logger.info(f"Saved: {out_path}")

    # --- Table 3: Confusion matrix (from test_in_domain.json) ---
    cm_path = results_dir / "hard_cbm" / "test_in_domain.json"
    if not cm_path.exists():
        # Try cv_results
        cm_path = results_dir / "hard_cbm" / "cv_results.json"

    if cm_path.exists():
        with open(cm_path) as f:
            cm_data = json.load(f)

        # Look for confusion matrix in various locations
        cm = None
        if "confusion_matrix" in cm_data:
            cm = cm_data["confusion_matrix"]
        elif "results" in cm_data and "confusion_matrix" in cm_data["results"]:
            cm = cm_data["results"]["confusion_matrix"]
        elif "aggregated" in cm_data and "confusion_matrix_sum" in cm_data["aggregated"]:
            cm = cm_data["aggregated"]["confusion_matrix_sum"]

        if cm is not None:
            table_cm = confusion_matrix_to_latex(
                cm,
                class_names=class_names,
                caption=(
                    "Confusion matrix on the in-domain test set (HardCBM, Plan A). "
                    "Row-normalized recall values; diagonal entries \\textbf{bolded}."
                ),
                label="tab:confusion_matrix",
                normalize=True,
            )
            all_tables["confusion_matrix"] = table_cm
            out_path = output_dir / "tab_confusion_matrix.tex"
            with open(out_path, "w") as f:
                f.write(table_cm)
            logger.info(f"Saved: {out_path}")

    # --- Table 4: Cross-survey results ---
    cs_files = list((results_dir / "cross_survey").glob("*.json"))
    for cs_file in cs_files:
        if "10dim" in cs_file.name or "12dim" in cs_file.name:
            with open(cs_file) as f:
                cs_data = json.load(f)

            mode = cs_data.get("mode", cs_file.stem)
            gaia_res = cs_data.get("gaia_test_results", {})
            ogle_res = cs_data.get("ogle_test_results", {})

            # Build a simple 2-row table
            header = ["Dataset", "Accuracy", "Macro F1", "Weighted F1"]
            data_rows = [
                ["Gaia (in-domain)",
                 f"{gaia_res.get('accuracy', float('nan')):.3f}",
                 f"{gaia_res.get('macro_f1', float('nan')):.3f}",
                 f"{gaia_res.get('weighted_f1', float('nan')):.3f}"],
                ["OGLE (cross-survey)",
                 f"{ogle_res.get('accuracy', float('nan')):.3f}",
                 f"{ogle_res.get('macro_f1', float('nan')):.3f}",
                 f"{ogle_res.get('weighted_f1', float('nan')):.3f}"],
            ]
            table_cs = _booktabs_table(
                header, data_rows,
                caption=f"Cross-survey performance (mode={mode}).",
                label=f"tab:cross_survey_{mode.replace('-', '_')}",
            )
            all_tables[f"cross_survey_{mode}"] = table_cs
            out_path = output_dir / f"tab_cross_survey_{mode}.tex"
            with open(out_path, "w") as f:
                f.write(table_cs)
            logger.info(f"Saved: {out_path}")

    logger.info(f"\nExported {len(all_tables)} LaTeX tables to {output_dir}")
    return all_tables
