"""
CBM Variable Star Classification -- Concept Space Visualization

Dimensionality reduction and specialized astrophysical plots for the
12-dimensional concept space.

Functions:
    plot_concept_tsne           -- t-SNE of concept space colored by class
    plot_concept_umap           -- UMAP dimensionality reduction
    plot_bailey_diagram         -- Period-Amplitude (Bailey) diagram
    plot_concept_distributions  -- Per-class violin/box plots for each concept
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_NAMES_12,
)
from cbm_variable_stars.shared.logger import logger
from cbm_variable_stars.visualization.plots import _save_figure, _setup_style, _COLORS, _DPI

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_colors(labels: np.ndarray, class_names: List[str]) -> List[str]:
    """Map integer labels to color strings."""
    palette = [_COLORS.get(cn, f"C{i}") for i, cn in enumerate(class_names)]
    return [palette[int(l)] for l in labels]


def _add_class_legend(
    ax: plt.Axes,
    class_names: List[str],
    loc: str = "best",
) -> None:
    """Add a class-color legend to the axes."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_COLORS.get(cn, f"C{i}"),
            markersize=8,
            label=cn,
        )
        for i, cn in enumerate(class_names)
    ]
    ax.legend(handles=handles, loc=loc, framealpha=0.9, title="Class")


# ---------------------------------------------------------------------------
# plot_concept_tsne
# ---------------------------------------------------------------------------

def plot_concept_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "t-SNE of Concept Space",
    perplexity: float = 40.0,
    n_iter: int = 1000,
    random_state: int = 42,
    figsize: tuple = (8, 7),
    alpha: float = 0.5,
    s: float = 8.0,
    max_samples: int = 5000,
) -> plt.Figure:
    """
    t-SNE dimensionality reduction of the concept space, colored by class.

    Parameters
    ----------
    features : np.ndarray, shape (N, n_concepts)
        Concept feature matrix (standardized).
    labels : np.ndarray, shape (N,)
        Integer class labels.
    class_names : list of str or None
        Defaults to CLASS_NAMES.
    save_path : str or Path or None
    title : str
    perplexity : float
        t-SNE perplexity (typical 5-50; larger for bigger datasets).
    n_iter : int
        t-SNE iterations.
    random_state : int
    figsize : tuple
    alpha : float
        Point transparency.
    s : float
        Point size.
    max_samples : int
        Sub-sample if N > max_samples (for speed).

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    if class_names is None:
        class_names = CLASS_NAMES

    labels = np.asarray(labels)

    # Sub-sample for large datasets
    if len(features) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(features), size=max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
        logger.info(f"t-SNE: sub-sampled to {max_samples} points")

    # Fill NaN
    features = np.nan_to_num(features, nan=0.0)

    logger.info(f"t-SNE: fitting on {len(features)} points (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        n_jobs=-1,
    )
    embedding = tsne.fit_transform(features)
    logger.info("t-SNE done.")

    fig, ax = plt.subplots(figsize=figsize)

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=_COLORS.get(class_name, f"C{class_idx}"),
            label=class_name,
            alpha=alpha,
            s=s,
            linewidths=0,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"{title}\n(perplexity={perplexity}, N={len(features)})")
    ax.legend(
        loc="best",
        markerscale=2.5,
        framealpha=0.9,
        title="Class",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_concept_umap
# ---------------------------------------------------------------------------

def plot_concept_umap(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "UMAP of Concept Space",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    figsize: tuple = (8, 7),
    alpha: float = 0.5,
    s: float = 8.0,
    max_samples: int = 10000,
) -> plt.Figure:
    """
    UMAP dimensionality reduction of the concept space, colored by class.

    Requires umap-learn >= 0.5.4 (M7 fix: added to requirements.txt).

    Parameters
    ----------
    features : np.ndarray, shape (N, n_concepts)
    labels : np.ndarray, shape (N,)
    class_names : list of str or None
    save_path : str or Path or None
    title : str
    n_neighbors : int
        UMAP n_neighbors (typical 5-50).
    min_dist : float
        UMAP min_dist (0-1; smaller = tighter clusters).
    metric : str
    random_state : int
    figsize : tuple
    alpha : float
    s : float
    max_samples : int

    Returns
    -------
    matplotlib Figure
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for UMAP visualization. "
            "Install with: pip install umap-learn>=0.5.4"
        )

    _setup_style()

    if class_names is None:
        class_names = CLASS_NAMES

    labels = np.asarray(labels)

    if len(features) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(features), size=max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
        logger.info(f"UMAP: sub-sampled to {max_samples} points")

    features = np.nan_to_num(features, nan=0.0)

    logger.info(
        f"UMAP: fitting on {len(features)} points "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist})..."
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=-1,
        low_memory=False,
    )
    embedding = reducer.fit_transform(features)
    logger.info("UMAP done.")

    fig, ax = plt.subplots(figsize=figsize)

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=_COLORS.get(class_name, f"C{class_idx}"),
            label=class_name,
            alpha=alpha,
            s=s,
            linewidths=0,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        f"{title}\n(n_neighbors={n_neighbors}, min_dist={min_dist}, N={len(features)})"
    )
    ax.legend(
        loc="best",
        markerscale=2.5,
        framealpha=0.9,
        title="Class",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_bailey_diagram
# ---------------------------------------------------------------------------

def plot_bailey_diagram(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    concept_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Bailey Diagram (Period–Amplitude)",
    figsize: tuple = (9, 6),
    alpha: float = 0.5,
    s: float = 8.0,
    period_range: Optional[tuple] = None,
    amplitude_range: Optional[tuple] = None,
) -> plt.Figure:
    """
    Period-Amplitude (Bailey) diagram, the classic variable star locus plot.

    X-axis: log10(period [days])
    Y-axis: amplitude [mag]

    Parameters
    ----------
    features : np.ndarray, shape (N, n_concepts)
        Feature matrix. Must contain "period" and "amplitude" columns.
    labels : np.ndarray, shape (N,)
        Integer class labels.
    class_names : list of str or None
    concept_names : list of str or None
        Column names for `features`. Used to locate period/amplitude columns.
        Defaults to CONCEPT_NAMES_12.
    save_path : str or Path or None
    title : str
    figsize : tuple
    alpha : float
    s : float
    period_range : tuple or None
        (log10_period_min, log10_period_max). Auto-detected if None.
    amplitude_range : tuple or None
        (amp_min, amp_max). Auto-detected if None.

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    if class_names is None:
        class_names = CLASS_NAMES
    if concept_names is None:
        concept_names = CONCEPT_NAMES_12

    labels = np.asarray(labels)

    # Locate period and amplitude columns
    try:
        period_idx = concept_names.index("period")
        amplitude_idx = concept_names.index("amplitude")
    except ValueError as e:
        raise ValueError(
            f"'period' and 'amplitude' must be in concept_names. Error: {e}"
        )

    period_scaled = features[:, period_idx]
    amplitude_scaled = features[:, amplitude_idx]

    # Note: features are standardized (z-scores). For Bailey diagram we
    # want to show the scaled values with appropriate axis labels.
    # The x-axis is labeled as "Period (standardized)" or log(period) if
    # we have scaler information.  Since we only have z-scores here, we
    # use the z-score directly but label axes accordingly.

    fig, ax = plt.subplots(figsize=figsize)

    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            period_scaled[mask],
            amplitude_scaled[mask],
            c=_COLORS.get(class_name, f"C{class_idx}"),
            label=f"{class_name} (n={mask.sum():,})",
            alpha=alpha,
            s=s,
            linewidths=0,
        )

    ax.set_xlabel("Period (standardized, z-score)")
    ax.set_ylabel("Amplitude (standardized, z-score)")
    ax.set_title(title)

    if period_range is not None:
        ax.set_xlim(period_range)
    if amplitude_range is not None:
        ax.set_ylim(amplitude_range)

    ax.legend(
        loc="upper right",
        markerscale=2.5,
        framealpha=0.9,
        title="Class",
        fontsize=9,
        ncol=2,
    )
    ax.grid(True, alpha=0.25)

    # Annotation
    ax.text(
        0.02, 0.02,
        f"N={len(labels):,} stars",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_concept_distributions
# ---------------------------------------------------------------------------

def plot_concept_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    concept_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Per-Class Concept Distributions",
    plot_type: str = "violin",
    figsize: Optional[tuple] = None,
    n_cols: int = 4,
) -> plt.Figure:
    """
    Per-class violin (or box) plots for each concept.

    Creates a grid of subplots, one per concept, showing the distribution
    of each concept's values across the 6 variable star classes.

    Parameters
    ----------
    features : np.ndarray, shape (N, n_concepts)
    labels : np.ndarray, shape (N,)
    concept_names : list of str or None
        Defaults to CONCEPT_NAMES_12.
    class_names : list of str or None
        Defaults to CLASS_NAMES.
    save_path : str or Path or None
    title : str
    plot_type : str
        "violin" or "box".
    figsize : tuple or None
        Auto-computed if None based on n_concepts and n_cols.
    n_cols : int
        Number of subplot columns.

    Returns
    -------
    matplotlib Figure
    """
    import pandas as pd

    _setup_style()

    if concept_names is None:
        concept_names = CONCEPT_NAMES_12
    if class_names is None:
        class_names = CLASS_NAMES

    labels = np.asarray(labels)
    n_concepts = len(concept_names)

    # Build long-form DataFrame for seaborn
    df = pd.DataFrame(features[:, :n_concepts], columns=concept_names)
    df["class"] = [
        class_names[int(l)] if int(l) < len(class_names) else str(l)
        for l in labels
    ]

    n_rows = int(np.ceil(n_concepts / n_cols))
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3.0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    class_palette = {cn: _COLORS.get(cn, f"C{i}") for i, cn in enumerate(class_names)}

    for i, concept in enumerate(concept_names):
        ax = axes[i]

        # Remove extreme outliers for cleaner display (clip at 3 sigma)
        vals = df[concept].values
        q_lo, q_hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
        df_plot = df.copy()
        df_plot[concept] = df_plot[concept].clip(lower=q_lo, upper=q_hi)

        if plot_type == "violin":
            try:
                sns.violinplot(
                    data=df_plot,
                    x="class",
                    y=concept,
                    palette=class_palette,
                    ax=ax,
                    inner="quartile",
                    cut=0,
                    scale="width",
                    linewidth=0.8,
                )
            except Exception:
                sns.boxplot(
                    data=df_plot,
                    x="class",
                    y=concept,
                    palette=class_palette,
                    ax=ax,
                    linewidth=0.8,
                )
        else:
            sns.boxplot(
                data=df_plot,
                x="class",
                y=concept,
                palette=class_palette,
                ax=ax,
                linewidth=0.8,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
            )

        ax.set_title(concept, fontsize=10, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("z-score", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Hide unused subplots
    for j in range(n_concepts, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# plot_concept_radar
# ---------------------------------------------------------------------------

def plot_concept_radar(
    class_means: Dict[str, List[float]],
    concept_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Class Concept Profiles (Radar Chart)",
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Radar / spider chart showing mean concept values per class.

    Parameters
    ----------
    class_means : dict
        {class_name: [mean_concept_0, ..., mean_concept_n-1]}
        Typically from compute_concept_class_association().
    concept_names : list of str or None
    class_names : list of str or None
    save_path : str or Path or None
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    _setup_style()

    if concept_names is None:
        concept_names = CONCEPT_NAMES_12
    if class_names is None:
        class_names = list(class_means.keys())

    n_concepts = len(concept_names)
    angles = np.linspace(0, 2 * np.pi, n_concepts, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for class_name in class_names:
        if class_name not in class_means:
            continue
        values = list(class_means[class_name]) + [class_means[class_name][0]]
        color = _COLORS.get(class_name, None)
        ax.plot(angles, values, linewidth=1.8, label=class_name, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(concept_names, size=9)
    ax.set_title(title, pad=16, fontsize=13)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.3, -0.1),
        framealpha=0.9,
        title="Class",
        fontsize=9,
    )

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig
