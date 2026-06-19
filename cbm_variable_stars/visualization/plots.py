"""
CBM Variable Star Classification -- Main Plotting Functions

Publication-quality figures (dpi=300, PDF format) using matplotlib + seaborn.

Functions:
    plot_confusion_matrix       -- Heatmap confusion matrix
    plot_training_curves        -- Loss and metric curves vs epoch
    plot_ablation_comparison    -- Bar chart of ablation results
    plot_intervention_curve     -- Accuracy vs n_concepts_intervened
    plot_learning_curve         -- Accuracy / F1 vs sample size
    plot_feature_importance     -- Horizontal bar chart of concept importance
    plot_class_distribution     -- Class balance bar chart
    plot_pareto_frontier        -- Accuracy--interpretability Pareto frontier
    plot_importance_consensus   -- Multi-method concept importance rankings
    plot_intervention_analysis  -- Dual-panel noise injection + recovery
    plot_domain_shift           -- KS statistic cross-survey domain shift
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_NAMES_12,
)
from cbm_variable_stars.shared.logger import logger

# Non-interactive backend for server environments
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_STYLE_CONTEXT = "seaborn-v0_8-paper"
_PALETTE = "tab10"
_DPI = 300
_FIG_FORMAT = "pdf"

# A&A journal layout dimensions (inches)
AA_COLWIDTH = 3.46    # single column: 88 mm
AA_TEXTWIDTH = 7.09   # full page:    180 mm

_COLORS = {
    "RRAB": "#1f77b4",
    "RRC": "#ff7f0e",
    "DCEP": "#2ca02c",
    "DSCT_SXPHE": "#d62728",
    "ECL": "#9467bd",
    "MIRA_SR": "#8c564b",
}


def _setup_style() -> None:
    """Apply consistent style to all figures."""
    try:
        plt.style.use(_STYLE_CONTEXT)
    except OSError:
        plt.style.use("seaborn-paper")
    sns.set_palette(_PALETTE)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": _DPI,
            "savefig.dpi": _DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(
    fig: plt.Figure,
    save_path: Optional[Union[str, Path]],
    bbox_inches: str = "tight",
) -> None:
    """Save figure to save_path or show interactively."""
    if save_path is None:
        plt.show()
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Infer format from file extension, fallback to PDF
    fmt = save_path.suffix.lstrip('.') if save_path.suffix else _FIG_FORMAT
    fig.savefig(str(save_path), dpi=_DPI, bbox_inches=bbox_inches, format=fmt)
    plt.close(fig)
    logger.info(f"Figure saved: {save_path}")


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    fmt_normalized: str = ".2f",
    fmt_raw: str = "d",
    figsize: tuple = (8, 6),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot a publication-quality confusion matrix heatmap.

    Parameters
    ----------
    cm : array-like, shape (n_classes, n_classes)
        Confusion matrix (raw counts).
    class_names : list of str or None
        Class labels. Defaults to CLASS_NAMES.
    title : str
        Figure title.
    save_path : str or Path or None
        Save path (PDF). If None, shows interactively.
    normalize : bool
        If True, show row-normalized proportions (recall per class).
    fmt_normalized : str
        Format string for normalized values.
    fmt_raw : str
        Format string for raw counts.
    figsize : tuple
    cmap : str
        Colormap name.

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    if class_names is None:
        class_names = CLASS_NAMES

    cm = np.array(cm)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(float) / row_sums
        fmt = fmt_normalized
        vmin, vmax = 0.0, 1.0
        cbar_label = "Recall (proportion)"
    else:
        cm_display = cm
        fmt = fmt_raw
        vmin, vmax = 0, None
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": cbar_label, "shrink": 0.8},
    )

    ax.set_xlabel("Predicted Class", labelpad=10)
    ax.set_ylabel("True Class", labelpad=10)
    ax.set_title(title, pad=14)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
    figsize: tuple = (10, 7),
) -> plt.Figure:
    """
    Plot training loss, validation loss, and validation metrics vs epoch.

    Parameters
    ----------
    history : list of dict
        Training log -- each entry is one epoch with keys such as:
        "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1".
    save_path : str or Path or None
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    if not history:
        logger.warning("plot_training_curves: history is empty.")
        return plt.figure()

    epochs = [r.get("epoch", i) for i, r in enumerate(history)]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, y=1.02)

    # --- Left: Loss ---
    ax = axes[0]
    if "train_loss" in history[0]:
        train_losses = [r.get("train_loss", float("nan")) for r in history]
        ax.plot(epochs, train_losses, label="Train loss", color="tab:blue", linewidth=1.5)
    if "val_loss" in history[0]:
        val_losses = [r.get("val_loss", float("nan")) for r in history]
        ax.plot(
            epochs, val_losses, label="Val loss", color="tab:orange",
            linewidth=1.5, linestyle="--"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # --- Right: Metrics ---
    ax = axes[1]
    metric_colors = {
        "val_accuracy": ("tab:green", "Accuracy"),
        "val_macro_f1": ("tab:red", "Macro F1"),
        "train_accuracy": ("tab:blue", "Train Accuracy"),
    }
    for key, (color, label) in metric_colors.items():
        if key in history[0]:
            vals = [r.get(key, float("nan")) for r in history]
            ax.plot(epochs, vals, label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics")
    ax.set_ylim(0, 1.02)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_ablation_comparison
# ---------------------------------------------------------------------------

def plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Ablation Study: Model Comparison",
    metric: str = "macro_f1_mean",
    metric_label: Optional[str] = None,
    figsize: tuple = (9, 5),
    color_baseline: str = "tab:blue",
    color_ablation: str = "tab:orange",
) -> plt.Figure:
    """
    Bar chart comparing ablation results.

    Parameters
    ----------
    results : dict
        {model_label: {"macro_f1_mean": float, "macro_f1_std": float, ...}}
        The first entry is assumed to be the baseline (full model).
    save_path : str or Path or None
    title : str
    metric : str
        Key to plot (e.g., "macro_f1_mean", "accuracy_mean").
    metric_label : str or None
        Y-axis label.
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    if not results:
        logger.warning("plot_ablation_comparison: empty results dict.")
        return plt.figure()

    if metric_label is None:
        metric_label = metric.replace("_mean", "").replace("_", " ").title()

    labels = list(results.keys())
    values = [results[k].get(metric, 0.0) for k in labels]
    std_key = metric.replace("_mean", "_std")
    stds = [results[k].get(std_key, 0.0) for k in labels]

    colors = [color_baseline] + [color_ablation] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        range(len(labels)),
        values,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7,
        yerr=stds,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "black", "capthick": 1.2},
    )

    # Annotate bars
    for bar, val, std in zip(bars, values, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.003,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(title)

    y_min = max(0, min(values) - 0.05)
    y_max = min(1.0, max(values) + 0.06)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=color_baseline, label="Baseline (full model)"),
            Patch(color=color_ablation, label="Ablation variants"),
        ],
        loc="lower right",
        framealpha=0.9,
    )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_intervention_curve
# ---------------------------------------------------------------------------

def plot_intervention_curve(
    results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Concept Intervention Curve",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Line plot of accuracy vs number of concepts intervened.

    Supports both sequential-random (with error bands) and greedy curves.

    Parameters
    ----------
    results : dict
        Must contain one of:
        - "mean_accuracies" + "std_accuracies": from sequential random
        - "accuracies": from greedy
        - "concept_names_order": for greedy x-tick labels (optional)
    save_path : str or Path or None
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    if "mean_accuracies" in results:
        # Sequential random with error bands
        means = np.array(results["mean_accuracies"])
        stds = np.array(results["std_accuracies"])
        n_steps = len(means)
        xs = list(range(n_steps))

        ax.plot(xs, means, color="tab:blue", linewidth=2, label="Random (mean)")
        ax.fill_between(
            xs,
            means - stds,
            means + stds,
            alpha=0.25,
            color="tab:blue",
            label=r"$\pm$1 std",
        )

        # Mark baseline and fully intervened
        ax.axhline(
            means[0], color="gray", linestyle=":", linewidth=1.2, label="Baseline (no intervention)"
        )
        ax.axhline(
            means[-1], color="tab:green", linestyle="--", linewidth=1.2,
            label="All concepts intervened"
        )

    if "accuracies" in results:
        # Greedy curve
        accs = results["accuracies"]
        xs_g = list(range(len(accs)))
        ax.plot(
            xs_g, accs,
            color="tab:orange", linewidth=2, marker="o", markersize=5,
            label="Greedy"
        )

    ax.set_xlabel("Number of Concepts Intervened")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.02))

    # If concept_names_order is available and points match, add secondary ticks
    if "concept_names_order" in results and "accuracies" in results:
        names = results["concept_names_order"]
        n_names = len(names)
        if n_names <= 12:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(range(1, n_names + 1))
            ax2.set_xticklabels(names, rotation=45, ha="left", fontsize=8)
            ax2.set_xlabel("Concept (greedy order)", fontsize=9)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_learning_curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
    results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Learning Curve",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot accuracy (and optionally F1) vs training sample size.

    Parameters
    ----------
    results : dict
        Output of run_learning_curve(). Contains keys:
            "sample_sizes": list of int
            "results": {n: {"mean_accuracy", "std_accuracy", "mean_f1", "std_f1"}}
    save_path : str or Path or None
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    sample_sizes = results.get("sample_sizes", [])
    lc_data = results.get("results", {})

    if not sample_sizes or not lc_data:
        logger.warning("plot_learning_curve: no data to plot.")
        return plt.figure()

    # Normalize keys (may be int or str)
    def _get(n: int, key: str) -> float:
        v = lc_data.get(n) or lc_data.get(str(n)) or {}
        return v.get(key, float("nan"))

    accs = [_get(n, "mean_accuracy") for n in sample_sizes]
    acc_stds = [_get(n, "std_accuracy") for n in sample_sizes]
    f1s = [_get(n, "mean_f1") for n in sample_sizes]
    f1_stds = [_get(n, "std_f1") for n in sample_sizes]

    fig, ax = plt.subplots(figsize=figsize)

    xs = sample_sizes

    ax.plot(xs, accs, color="tab:blue", linewidth=2, marker="o", markersize=5, label="Accuracy")
    ax.fill_between(
        xs,
        np.array(accs) - np.array(acc_stds),
        np.array(accs) + np.array(acc_stds),
        alpha=0.2,
        color="tab:blue",
    )

    if any(not np.isnan(f) for f in f1s):
        ax.plot(
            xs, f1s, color="tab:orange", linewidth=2, marker="s",
            markersize=5, linestyle="--", label="Macro F1"
        )
        ax.fill_between(
            xs,
            np.array(f1s) - np.array(f1_stds),
            np.array(f1s) + np.array(f1_stds),
            alpha=0.2,
            color="tab:orange",
        )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_locator(mticker.FixedLocator(xs))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_feature_importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: Union[np.ndarray, Dict[str, float], List[float]],
    concept_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Concept Importance",
    figsize: tuple = (7, 6),
    color: str = "tab:blue",
    highlight_top_n: int = 3,
) -> plt.Figure:
    """
    Horizontal bar chart of concept importance scores.

    Parameters
    ----------
    importance : array-like or dict
        If array/list: importance[i] corresponds to concept_names[i].
        If dict: {concept_name: importance_score}.
    concept_names : list of str or None
        Defaults to CONCEPT_NAMES_12.
    save_path : str or Path or None
    title : str
    figsize : tuple
    color : str
    highlight_top_n : int
        Top N bars are highlighted.

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    if concept_names is None:
        concept_names = CONCEPT_NAMES_12

    if isinstance(importance, dict):
        # Align to concept_names order
        scores = np.array([importance.get(c, 0.0) for c in concept_names])
    else:
        scores = np.array(importance)
        if len(scores) != len(concept_names):
            logger.warning(
                f"importance length ({len(scores)}) != "
                f"concept_names length ({len(concept_names)}). "
                "Truncating/padding."
            )
            n = min(len(scores), len(concept_names))
            scores = scores[:n]
            concept_names = concept_names[:n]

    # Sort by importance (descending) for horizontal bar
    sort_idx = np.argsort(scores)
    sorted_names = [concept_names[i] for i in sort_idx]
    sorted_scores = scores[sort_idx]

    # [Mod8 FIX] Colors: highlight top-N by sorted position (importance rank),
    # not by original concept index.
    n_bars = len(sorted_scores)
    bar_colors = [
        "tab:red" if pos >= (n_bars - highlight_top_n) else color
        for pos in range(n_bars)
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        range(len(sorted_names)),
        sorted_scores,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Annotate
    for bar, val in zip(bars, sorted_scores):
        ax.text(
            bar.get_width() + max(sorted_scores) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="tab:red", label=f"Top {highlight_top_n}"),
            Patch(color=color, label="Others"),
        ],
        loc="lower right",
        framealpha=0.9,
    )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_class_distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    labels: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Class Distribution",
    figsize: tuple = (7, 5),
    show_percentages: bool = True,
) -> plt.Figure:
    """
    Bar chart of class label distribution.

    Parameters
    ----------
    labels : array-like of int
        Integer class labels.
    class_names : list of str or None
        Defaults to CLASS_NAMES.
    save_path : str or Path or None
    title : str
    figsize : tuple
    show_percentages : bool
        Annotate bars with count and percentage.

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    if class_names is None:
        class_names = CLASS_NAMES

    labels = np.asarray(labels)
    n_classes = len(class_names)
    counts = np.array(
        [int(np.sum(labels == i)) for i in range(n_classes)]
    )
    total = counts.sum()

    palette = [_COLORS.get(cn, f"C{i}") for i, cn in enumerate(class_names)]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        range(n_classes),
        counts,
        color=palette,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )

    if show_percentages and total > 0:
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.003,
                f"{count:,}\n({100 * count / total:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Number of Stars")
    ax.set_title(f"{title}  (N={total:,})")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Batch generation convenience
# ---------------------------------------------------------------------------

def generate_standard_figures(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    class_names: Optional[List[str]] = None,
    concept_names: Optional[List[str]] = None,
) -> None:
    """
    Generate the standard set of paper figures from a results dictionary.

    Parameters
    ----------
    results : dict
        Full results dict from run_all_experiments().
    output_dir : str or Path
    class_names : list of str or None
    concept_names : list of str or None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = CLASS_NAMES
    if concept_names is None:
        concept_names = CONCEPT_NAMES_12

    # Learning curve
    if "learning_curve" in results and results["learning_curve"]:
        plot_learning_curve(
            results["learning_curve"],
            save_path=output_dir / "fig_learning_curve.pdf",
        )

    # Class distribution from model_results (labels)
    # (requires raw labels which aren't stored in results summary)
    logger.info(f"Standard figures generated in {output_dir}")


# ---------------------------------------------------------------------------
# plot_pareto_frontier
# ---------------------------------------------------------------------------

# Display-friendly model names for paper figures
MODEL_DISPLAY = {
    "hard_cbm": "HardCBM",
    "hard_cbm_linear": "HardCBM-Lin",
    "hard_cbm_cal": "HardCBM-Cal",
    "soft_cbm": "SoftCBM",
    "cem": "CEM",
    "mlp": "MLP",
    "rf": "RF",
    "xgb": "XGBoost",
}

# Colorblind-friendly palette (Wong 2011)
_CB_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#D55E00",  # vermilion
    "#F0E442",  # yellow
    "#000000",  # black
]


def plot_pareto_frontier(
    data: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (4.5, 3.8),
) -> plt.Figure:
    """
    Scatter plot of accuracy vs interpretability with Pareto frontier line.

    Parameters
    ----------
    data : dict
        B5_pareto_frontier.json contents with "points" list.
    save_path : str or Path or None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    points = data.get("points", [])
    if not points:
        logger.warning("plot_pareto_frontier: no data points.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=figsize)

    # Separate Pareto front and dominated points
    pareto_pts = [p for p in points if p.get("on_pareto_front", False)]

    # Sort Pareto front by interpretability for line
    pareto_pts.sort(key=lambda p: p["interpretability"])

    # Draw Pareto frontier line
    pf_x = [p["interpretability"] for p in pareto_pts]
    pf_y = [p["accuracy"] * 100 for p in pareto_pts]
    ax.plot(pf_x, pf_y, color="gray", linewidth=1.0, linestyle="--", alpha=0.7,
            zorder=1)

    # Per-model label placement: (dx, dy, ha, va) to avoid overlap
    # xgb (0.1, 99.81) and rf (0.1, 99.79) are nearly coincident;
    # Pareto line runs rightward from xgb, so label goes above.
    _LABEL_POS = {
        "xgb":             ( 0.00,  0.6, "center", "bottom"),
        "rf":              (-0.03, -0.3, "right",  "top"),
        "soft_cbm":        ( 0.04,  0.4, "left",  "bottom"),
        "cem":             ( 0.04, -0.7, "left",  "top"),
        "hard_cbm_cal":    ( 0.04,  0.4, "left",  "bottom"),
        "mlp":             (-0.03, -0.5, "right", "top"),
        "hard_cbm":        ( 0.04,  0.4, "left",  "bottom"),
        "hard_cbm_linear": (-0.03,  0.3, "right", "bottom"),
    }

    # Plot all points with error bars
    for i, pt in enumerate(points):
        x = pt["interpretability"]
        y = pt["accuracy"] * 100
        yerr = pt.get("accuracy_std", 0) * 100
        on_front = pt.get("on_pareto_front", False)
        color = _CB_PALETTE[i % len(_CB_PALETTE)]
        marker = "o" if on_front else "x"
        ms = 7 if on_front else 6

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=marker, color=color, markersize=ms,
            capsize=3, capthick=1.0, elinewidth=1.0,
            zorder=3,
        )

        label = MODEL_DISPLAY.get(pt["model"], pt["model"])
        dx, dy, ha, va = _LABEL_POS.get(
            pt["model"], (0.03, 0.3, "left", "bottom")
        )
        ax.annotate(
            label, (x, y),
            xytext=(x + dx, y + dy),
            fontsize=8, ha=ha, va=va,
            zorder=4,
        )

    ax.set_xlabel("Interpretability Score")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(88.5, 101.0)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_importance_consensus
# ---------------------------------------------------------------------------

def plot_importance_consensus(
    data: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (7.08, 4.5),
) -> plt.Figure:
    """
    Grouped horizontal bar chart of concept importance across 5 methods.

    Parameters
    ----------
    data : dict
        B3_importance_consensus.json with "rankings" and "rank_arrays".
    save_path : str or Path or None
    figsize : tuple
        Default double-column A&A width (180mm = 7.08in).

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    rank_arrays = data.get("rank_arrays", {})
    method_names = data.get("method_names", list(rank_arrays.keys()))

    if not rank_arrays:
        logger.warning("plot_importance_consensus: no ranking data.")
        return plt.figure()

    concept_names = CONCEPT_NAMES_12
    n_concepts = len(concept_names)
    n_methods = len(method_names)

    # Build rank matrix: rows=concepts, cols=methods
    rank_matrix = np.zeros((n_concepts, n_methods))
    for j, method in enumerate(method_names):
        ranks = rank_arrays[method]
        for i in range(n_concepts):
            rank_matrix[i, j] = ranks[i]

    # Compute mean rank per concept, sort by it (best = lowest rank)
    mean_ranks = rank_matrix.mean(axis=1)
    sort_idx = np.argsort(mean_ranks)  # ascending = best first

    sorted_names = [concept_names[i] for i in sort_idx]
    sorted_matrix = rank_matrix[sort_idx]

    # Method display names
    method_display = {
        "loo_accuracy_drop": "LOO Drop",
        "anova_f_statistic": "ANOVA F",
        "mutual_information": "Mutual Info",
        "weight_norms": "Weight Norm",
        "noise_sensitivity": "Noise Sens.",
    }

    fig, ax = plt.subplots(figsize=figsize)

    bar_height = 0.15
    y_positions = np.arange(n_concepts)

    for j, method in enumerate(method_names):
        offsets = y_positions + (j - n_methods / 2 + 0.5) * bar_height
        ranks = sorted_matrix[:, j]
        color = _CB_PALETTE[j % len(_CB_PALETTE)]
        label = method_display.get(method, method)
        ax.barh(
            offsets, ranks, height=bar_height,
            color=color, alpha=0.85, edgecolor="white", linewidth=0.3,
            label=label,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Rank (1 = most important)")
    ax.set_xlim(0, n_concepts + 1)
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9, ncol=2)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_intervention_analysis
# ---------------------------------------------------------------------------

def plot_intervention_analysis(
    data: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (7.08, 3.5),
) -> plt.Figure:
    """
    Dual-panel intervention analysis.

    Panel (a): noise level (sigma) vs accuracy for multiple models.
    Panel (b): single-concept recovery bar chart at sigma=1.0.

    Parameters
    ----------
    data : dict
        B1_intervention.json contents.
    save_path : str or Path or None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)

    # --- Panel (a): sigma vs accuracy ---
    model_keys = [k for k in data.keys() if isinstance(data[k], dict) and "noise_injection" in data[k]]

    for idx, model_key in enumerate(model_keys):
        mdata = data[model_key]["noise_injection"]
        clean_acc = mdata["clean_accuracy"] * 100
        sigmas = [0.0] + mdata["noise_stds"]
        accs = [clean_acc]

        for sigma in mdata["noise_stds"]:
            s_key = str(sigma) if str(sigma) in mdata["per_noise_level"] else f"{sigma}"
            level_data = mdata["per_noise_level"].get(s_key, {})
            accs.append(level_data.get("accuracy_noisy_all", 0) * 100)

        color = _CB_PALETTE[idx % len(_CB_PALETTE)]
        label = MODEL_DISPLAY.get(model_key, model_key)
        ax_a.plot(sigmas, accs, marker="o", markersize=5, linewidth=1.5,
                  color=color, label=label)

    ax_a.set_xlabel(r"Noise level ($\sigma$)")
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.set_title("(a) Noise injection", fontsize=10)
    ax_a.legend(fontsize=8, framealpha=0.9)
    ax_a.grid(True, alpha=0.3, linestyle="--")

    # --- Panel (b): single-concept recovery at sigma=1.0 ---
    # Use first model (hard_cbm) for concept recovery
    first_model = model_keys[0] if model_keys else None
    if first_model:
        mdata = data[first_model]["noise_injection"]
        sigma_key = "1.0"
        if sigma_key in mdata["per_noise_level"]:
            recovery = mdata["per_noise_level"][sigma_key]["per_concept_recovery"]

            concepts = list(recovery.keys())
            drops = [recovery[c]["performance_drop"] * 100 for c in concepts]

            # Sort by drop (descending)
            sorted_pairs = sorted(zip(concepts, drops), key=lambda x: x[1], reverse=True)
            concepts_s, drops_s = zip(*sorted_pairs)

            colors = [_CB_PALETTE[0] if d > 5 else _CB_PALETTE[2] if d < 2 else _CB_PALETTE[1]
                      for d in drops_s]

            ax_b.barh(range(len(concepts_s)), drops_s, color=colors,
                      edgecolor="white", linewidth=0.3, alpha=0.85)
            ax_b.set_yticks(range(len(concepts_s)))
            ax_b.set_yticklabels(concepts_s, fontsize=8)
            ax_b.set_xlabel("Accuracy drop (%)")
            ax_b.set_title(r"(b) Per-concept drop ($\sigma=1.0$)", fontsize=10)
            ax_b.grid(axis="x", alpha=0.3, linestyle="--")
            ax_b.invert_yaxis()

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_domain_shift
# ---------------------------------------------------------------------------

def plot_domain_shift(
    data: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (3.54, 3.5),
) -> plt.Figure:
    """
    Horizontal bar chart of KS statistics colored by severity.

    Parameters
    ----------
    data : dict
        B8_cross_survey.json with "domain_shift" dict.
    save_path : str or Path or None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    domain_shift = data.get("domain_shift", {})
    if not domain_shift:
        logger.warning("plot_domain_shift: no domain shift data.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=figsize)

    # Collect concepts and KS values, skip NaN
    concepts = []
    ks_vals = []
    for concept, stats in domain_shift.items():
        ks = stats.get("ks_statistic")
        if ks is not None and not (isinstance(ks, float) and np.isnan(ks)):
            concepts.append(concept)
            ks_vals.append(ks)

    # Sort by KS statistic (ascending for horizontal bar)
    sorted_pairs = sorted(zip(concepts, ks_vals), key=lambda x: x[1])
    concepts_s, ks_s = zip(*sorted_pairs)

    # Color by severity: green < 0.2, orange 0.2-0.5, red > 0.5
    colors = []
    for ks in ks_s:
        if ks < 0.2:
            colors.append("#009E73")  # green
        elif ks < 0.5:
            colors.append("#E69F00")  # orange
        else:
            colors.append("#D55E00")  # red/vermilion

    bars = ax.barh(range(len(concepts_s)), ks_s, color=colors,
                   edgecolor="white", linewidth=0.3, alpha=0.85)

    # Annotate values
    for bar, val in zip(bars, ks_s):
        ax.text(
            bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8,
        )

    ax.set_yticks(range(len(concepts_s)))
    ax.set_yticklabels(concepts_s, fontsize=8)
    ax.set_xlabel("KS Statistic")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Severity threshold lines
    ax.axvline(0.2, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#009E73", label="Good (KS < 0.2)"),
            Patch(color="#E69F00", label="Moderate (0.2--0.5)"),
            Patch(color="#D55E00", label="Poor (KS > 0.5)"),
        ],
        fontsize=8, loc="lower right", framealpha=0.9,
    )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
