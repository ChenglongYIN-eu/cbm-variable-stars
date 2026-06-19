"""
CBM Variable Star Classification -- Concept Correlation Analysis (I7)

Analyzes relationships between the 12 physical concepts to:
1. Identify redundant/correlated concepts (Pearson correlation matrix).
2. Characterize how each concept discriminates between variable star classes
   (per-class mean/std of each concept).

These analyses inform ablation experiment design and help interpret
CBM predictions in physical terms.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from cbm_variable_stars.shared.constants import (
    CLASS_NAMES,
    CONCEPT_GROUPS,
    CONCEPT_NAMES_12,
    LABEL_TO_IDX,
    RANDOM_SEED,
)
from cbm_variable_stars.shared.logger import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(
    model_name: str,
    n_concepts: int,
    concept_names: List[str],
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "n_concepts": n_concepts,
        "concept_names": concept_names,
    }


def _to_dataframe(
    features: Any,
    concept_names: List[str],
) -> pd.DataFrame:
    """
    Convert features to a DataFrame with concept_names as columns.

    Accepts np.ndarray, pd.DataFrame, or dict.
    """
    if isinstance(features, pd.DataFrame):
        # Make sure all concept columns are present
        missing = [c for c in concept_names if c not in features.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing concept columns: {missing}"
            )
        return features[concept_names].copy()
    elif isinstance(features, np.ndarray):
        if features.shape[1] != len(concept_names):
            raise ValueError(
                f"features has {features.shape[1]} columns, "
                f"but concept_names has {len(concept_names)} elements."
            )
        return pd.DataFrame(features, columns=concept_names)
    else:
        raise TypeError(f"Unsupported features type: {type(features)}")


# ---------------------------------------------------------------------------
# compute_concept_correlation
# ---------------------------------------------------------------------------

def compute_concept_correlation(
    features_df: Any,
    concept_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Pearson correlation matrix for the concept features.

    Parameters
    ----------
    features_df : pd.DataFrame or np.ndarray
        Feature matrix. If DataFrame, must contain concept_names as columns.
        If ndarray, shape (N, len(concept_names)).
    concept_names : list of str or None
        Active concept names. Defaults to CONCEPT_NAMES_12.

    Returns
    -------
    dict
        {
          "concept_names": list of str,
          "correlation_matrix": list of list of float  (n_concepts x n_concepts),
          "high_correlations": [{"concept_a", "concept_b", "r"}, ...],
              -- pairs with |r| > 0.7, sorted by |r| descending
          "low_correlations": [{"concept_a", "concept_b", "r"}, ...],
              -- pairs with |r| < 0.1
        }
    """
    if concept_names is None:
        concept_names = CONCEPT_NAMES_12

    df = _to_dataframe(features_df, concept_names)

    # Drop NaN rows for correlation computation
    df_clean = df.dropna()
    n_dropped = len(df) - len(df_clean)
    if n_dropped > 0:
        logger.info(f"Correlation: dropped {n_dropped} rows with NaN values.")

    corr_matrix = df_clean.corr(method="pearson")
    corr_array = corr_matrix.values.tolist()

    n = len(concept_names)
    high_corrs: List[Dict[str, Any]] = []
    low_corrs: List[Dict[str, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr_matrix.iloc[i, j])
            entry = {
                "concept_a": concept_names[i],
                "concept_b": concept_names[j],
                "r": r,
                "r_abs": abs(r),
            }
            if abs(r) > 0.7:
                high_corrs.append(entry)
            elif abs(r) < 0.1:
                low_corrs.append(entry)

    high_corrs.sort(key=lambda x: x["r_abs"], reverse=True)
    low_corrs.sort(key=lambda x: x["r_abs"])

    # Remove 'r_abs' from output (only used for sorting)
    for entry in high_corrs + low_corrs:
        entry.pop("r_abs", None)

    logger.info(
        f"Concept correlation: {len(high_corrs)} highly correlated pairs "
        f"(|r|>0.7), {len(low_corrs)} near-zero pairs (|r|<0.1)"
    )

    return {
        "concept_names": concept_names,
        "n_samples": len(df_clean),
        "correlation_matrix": corr_array,
        "high_correlations": high_corrs,
        "low_correlations": low_corrs,
        "concept_groups": CONCEPT_GROUPS,
    }


# ---------------------------------------------------------------------------
# compute_concept_class_association
# ---------------------------------------------------------------------------

def compute_concept_class_association(
    features_df: Any,
    labels: Any,
    concept_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute per-class mean and standard deviation for each concept.

    Also computes the between-class to within-class variance ratio
    (F-ratio / one-way ANOVA F-statistic) to rank concepts by their
    discriminative power.

    Parameters
    ----------
    features_df : pd.DataFrame or np.ndarray
        Feature matrix.
    labels : array-like of int
        Integer class labels aligned with features_df rows.
    concept_names : list of str or None
        Defaults to CONCEPT_NAMES_12.
    class_names : list of str or None
        Defaults to CLASS_NAMES.

    Returns
    -------
    dict
        {
          "concept_names": [...],
          "class_names": [...],
          "per_class_stats": {
              concept_name: {
                  class_name: {"mean": float, "std": float, "n": int},
                  ...
              },
              ...
          },
          "discriminative_power": {
              concept_name: {"f_statistic": float, "p_value": float, "rank": int},
              ...
          },
          "concept_ranking": [concept_name, ...]  -- sorted by F-statistic desc
        }
    """
    from scipy import stats as scipy_stats
    from scipy.stats import levene, kruskal

    if concept_names is None:
        concept_names = CONCEPT_NAMES_12
    if class_names is None:
        class_names = CLASS_NAMES

    df = _to_dataframe(features_df, concept_names)

    labels_arr = np.asarray(labels, dtype=np.int64)
    if len(labels_arr) != len(df):
        raise ValueError(
            f"labels length ({len(labels_arr)}) != "
            f"features rows ({len(df)})"
        )

    df = df.copy()
    df["_label"] = labels_arr

    per_class_stats: Dict[str, Dict[str, Any]] = {}
    discriminative_power: Dict[str, Any] = {}

    for concept in concept_names:
        per_class_stats[concept] = {}
        class_groups: List[np.ndarray] = []

        for class_idx, class_name in enumerate(class_names):
            mask = df["_label"] == class_idx
            values = df.loc[mask, concept].dropna().values

            if len(values) == 0:
                per_class_stats[concept][class_name] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "n": 0,
                }
            else:
                per_class_stats[concept][class_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "n": int(len(values)),
                }
                class_groups.append(values)

        # Assumption checks + ANOVA F-statistic + non-parametric alternative
        if len(class_groups) >= 2 and all(len(g) > 0 for g in class_groups):
            # 1. Levene's test for homogeneity of variance
            try:
                levene_stat, levene_p = levene(*class_groups)
            except Exception:
                levene_stat, levene_p = float('nan'), float('nan')

            # 2. Kruskal-Wallis H test (non-parametric alternative)
            try:
                h_stat, kw_p = kruskal(*class_groups)
            except Exception:
                h_stat, kw_p = float('nan'), float('nan')

            # 3. Original ANOVA F-test
            try:
                f_stat, p_val = scipy_stats.f_oneway(*class_groups)
            except Exception:
                f_stat, p_val = float('nan'), float('nan')

            discriminative_power[concept] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "kruskal_wallis_h": float(h_stat),
                "kruskal_wallis_p": float(kw_p),
                "levene_statistic": float(levene_stat),
                "levene_p_value": float(levene_p),
                "variance_homogeneous": bool(levene_p > 0.05),
            }
        else:
            discriminative_power[concept] = {
                "f_statistic": float("nan"),
                "p_value": float("nan"),
                "kruskal_wallis_h": float("nan"),
                "kruskal_wallis_p": float("nan"),
                "levene_statistic": float("nan"),
                "levene_p_value": float("nan"),
                "variance_homogeneous": False,
            }

    # Rank concepts by F-statistic (highest = most discriminative)
    valid_f = [
        (c, discriminative_power[c]["f_statistic"])
        for c in concept_names
        if not np.isnan(discriminative_power[c]["f_statistic"])
    ]
    valid_f.sort(key=lambda x: x[1], reverse=True)
    concept_ranking = [c for c, _ in valid_f]

    for rank, (concept, _) in enumerate(valid_f):
        discriminative_power[concept]["rank"] = rank + 1

    logger.info(
        f"Concept class association: top discriminative concept = "
        f"{concept_ranking[0] if concept_ranking else 'N/A'}"
    )

    return {
        "concept_names": concept_names,
        "class_names": class_names,
        "n_samples": len(df),
        "per_class_stats": per_class_stats,
        "discriminative_power": discriminative_power,
        "concept_ranking": concept_ranking,
    }


# ---------------------------------------------------------------------------
# compute_concept_mi
# ---------------------------------------------------------------------------

def compute_concept_mutual_information(
    features_df: Any,
    labels: Any,
    concept_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute mutual information between each concept and the class label.

    Uses scikit-learn's mutual_info_classif with continuous features.

    Parameters
    ----------
    features_df : pd.DataFrame or np.ndarray
    labels : array-like of int
    concept_names : list of str or None

    Returns
    -------
    dict
        {
          "concept_names": [...],
          "mutual_information": {concept_name: float, ...},
          "mi_ranking": [concept_name, ...]  -- sorted by MI desc
        }
    """
    from sklearn.feature_selection import mutual_info_classif

    if concept_names is None:
        concept_names = CONCEPT_NAMES_12

    df = _to_dataframe(features_df, concept_names)
    labels_arr = np.asarray(labels, dtype=np.int64)

    # Fill NaN with column medians for MI computation
    df_filled = df.fillna(df.median())
    X = df_filled.values
    y = labels_arr

    mi = mutual_info_classif(X, y, discrete_features=False, random_state=RANDOM_SEED)

    mi_dict = {c: float(mi[i]) for i, c in enumerate(concept_names)}

    sorted_concepts = sorted(concept_names, key=lambda c: mi_dict[c], reverse=True)

    logger.info(
        f"Mutual information: top concept = "
        f"{sorted_concepts[0] if sorted_concepts else 'N/A'} "
        f"(MI={mi_dict.get(sorted_concepts[0], float('nan')):.4f})"
    )

    return {
        "concept_names": concept_names,
        "mutual_information": mi_dict,
        "mi_ranking": sorted_concepts,
    }


# ---------------------------------------------------------------------------
# run_correlation_analysis
# ---------------------------------------------------------------------------

def run_correlation_analysis(
    features_df: Any,
    labels: Any,
    output_dir: str | Path = "results/correlation",
    concept_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full concept correlation and class association analysis (I7).

    Runs:
        1. Pearson correlation matrix between all concept pairs.
        2. Per-class mean/std of each concept + ANOVA F-statistic.
        3. Mutual information between each concept and the class label.

    Parameters
    ----------
    features_df : pd.DataFrame or np.ndarray
        Feature matrix. If DataFrame, must contain concept_names columns
        plus a "label" column.
    labels : array-like of int
        Integer class labels aligned with features_df.
    output_dir : str or Path
        Directory for saving results JSON.
    concept_names : list of str or None
        Defaults to CONCEPT_NAMES_12.
    class_names : list of str or None
        Defaults to CLASS_NAMES.

    Returns
    -------
    dict
        {
          "_metadata": {...},
          "correlation": {...},
          "class_association": {...},
          "mutual_information": {...},
        }
    """
    if concept_names is None:
        concept_names = CONCEPT_NAMES_12
    if class_names is None:
        class_names = CLASS_NAMES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Concept Correlation Analysis (I7)")
    logger.info(f"  Concepts: {concept_names}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    # Convert labels to array
    labels_arr = np.asarray(labels, dtype=np.int64)

    # 1. Pearson correlation
    logger.info("\n[1/3] Computing Pearson correlation matrix...")
    corr_result = compute_concept_correlation(features_df, concept_names)

    corr_path = output_dir / "concept_correlation.json"
    with open(corr_path, "w") as f:
        json.dump(corr_result, f, indent=2, default=str)
    logger.info(f"  Saved to {corr_path}")

    # 2. Class association
    logger.info("\n[2/3] Computing per-class concept statistics...")
    assoc_result = compute_concept_class_association(
        features_df, labels_arr, concept_names, class_names
    )

    assoc_path = output_dir / "concept_class_association.json"
    with open(assoc_path, "w") as f:
        json.dump(assoc_result, f, indent=2, default=str)
    logger.info(f"  Saved to {assoc_path}")

    # 3. Mutual information
    logger.info("\n[3/3] Computing mutual information with class labels...")
    try:
        mi_result = compute_concept_mutual_information(
            features_df, labels_arr, concept_names
        )
        mi_path = output_dir / "concept_mutual_information.json"
        with open(mi_path, "w") as f:
            json.dump(mi_result, f, indent=2, default=str)
        logger.info(f"  Saved to {mi_path}")
    except Exception as e:
        logger.warning(f"  Mutual information failed: {e}")
        mi_result = {"error": str(e)}

    result = {
        "_metadata": _make_metadata(
            model_name="correlation_analysis",
            n_concepts=len(concept_names),
            concept_names=concept_names,
        ),
        "correlation": corr_result,
        "class_association": assoc_result,
        "mutual_information": mi_result,
    }

    # Combined summary
    summary_path = output_dir / "correlation_analysis.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"\nFull correlation analysis saved to {summary_path}")

    # Print top insights
    logger.info("\n=== Top Insights ===")

    if "high_correlations" in corr_result and corr_result["high_correlations"]:
        top = corr_result["high_correlations"][0]
        logger.info(
            f"  Highest correlation: {top['concept_a']} <-> {top['concept_b']} "
            f"(r={top['r']:.3f})"
        )

    if "concept_ranking" in assoc_result and assoc_result["concept_ranking"]:
        top_disc = assoc_result["concept_ranking"][0]
        dp = assoc_result["discriminative_power"][top_disc]
        f_val = dp["f_statistic"]
        logger.info(
            f"  Most discriminative concept (ANOVA): {top_disc} "
            f"(F={f_val:.1f})"
        )
        # Kruskal-Wallis non-parametric result
        kw_h = dp.get("kruskal_wallis_h", float("nan"))
        kw_p = dp.get("kruskal_wallis_p", float("nan"))
        logger.info(
            f"    Kruskal-Wallis H={kw_h:.1f}, p={kw_p:.2e}"
        )
        # Variance homogeneity summary across all concepts
        n_homogeneous = sum(
            1 for c in assoc_result["concept_ranking"]
            if assoc_result["discriminative_power"][c].get(
                "variance_homogeneous", False
            )
        )
        n_total = len(assoc_result["concept_ranking"])
        logger.info(
            f"    Variance homogeneity (Levene's test): "
            f"{n_homogeneous}/{n_total} concepts have homogeneous variance "
            f"(p > 0.05)"
        )

    if "mi_ranking" in mi_result and mi_result["mi_ranking"]:
        top_mi = mi_result["mi_ranking"][0]
        mi_val = mi_result["mutual_information"][top_mi]
        logger.info(f"  Highest MI concept: {top_mi} (MI={mi_val:.4f})")

    return result
