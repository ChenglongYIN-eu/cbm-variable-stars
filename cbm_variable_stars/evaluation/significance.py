"""
Statistical significance testing for model comparison.

[Fix M8] Uses scipy.stats.binomtest instead of deprecated binom_test.
[Fix C2] Nadeau-Bengio corrected paired t-test for overlapping CV folds.
[Fix C3] Holm-Bonferroni multiple comparison correction.

Provides:
    - mcnemar_test:                  McNemar test for pairwise model comparison
    - paired_cv_ttest:               Nadeau-Bengio corrected paired t-test on CV fold scores
    - holm_bonferroni:               Multiple comparison correction
    - binomial_test:                 Binomial test using scipy.stats.binomtest
    - bootstrap_confidence_interval: BCa bootstrap CI for out-of-domain datasets
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Callable, Union, Tuple


def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> Dict[str, Any]:
    """
    McNemar test -- test whether two models have significantly different error rates.

    Builds a 2x2 contingency table and tests whether the two models make
    the same types of errors.

    [Fix M8] Uses scipy.stats.binomtest (replaces deprecated binom_test)
    for exact test when n_discordant < 25.

    Args:
        preds_a:      Predictions from model A, shape (N,)
        preds_b:      Predictions from model B, shape (N,)
        labels:       Ground truth labels, shape (N,)
        model_a_name: Name of model A (for reporting)
        model_b_name: Name of model B (for reporting)

    Returns:
        dict with keys:
            "test":               test type used
            "model_a", "model_b": model names
            "contingency_table":  2x2 table counts
            "statistic":          test statistic
            "p_value":            p-value
            "significant_at_005": bool
            "significant_at_001": bool
    """
    y_true = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    correct_a = (preds_a == y_true).astype(int)
    correct_b = (preds_b == y_true).astype(int)

    # Contingency table entries
    n00 = int(np.sum((correct_a == 1) & (correct_b == 1)))  # both correct
    n01 = int(np.sum((correct_a == 1) & (correct_b == 0)))  # A correct, B wrong
    n10 = int(np.sum((correct_a == 0) & (correct_b == 1)))  # A wrong, B correct
    n11 = int(np.sum((correct_a == 0) & (correct_b == 0)))  # both wrong

    if n01 + n10 == 0:
        return {
            "test": "mcnemar",
            "model_a": model_a_name,
            "model_b": model_b_name,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_005": False,
            "significant_at_001": False,
            "interpretation": "No discordant pairs -- models have identical predictions",
        }

    if n01 + n10 < 25:
        # [Fix M8] Exact test using scipy.stats.binomtest (replaces deprecated binom_test)
        result = stats.binomtest(n01, n01 + n10, p=0.5)
        p_value = result.pvalue
        statistic = n01 / (n01 + n10)
        test_type = "mcnemar_exact (binomtest)"
    else:
        # Chi-squared approximation with continuity correction
        statistic = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
        p_value = float(stats.chi2.sf(statistic, df=1))
        test_type = "mcnemar_chi2"

    return {
        "test": test_type,
        "model_a": model_a_name,
        "model_b": model_b_name,
        "contingency_table": {
            "both_correct": n00,
            "a_correct_b_wrong": n01,
            "a_wrong_b_correct": n10,
            "both_wrong": n11,
        },
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
    }


def paired_cv_ttest(
    scores_a: List[float],
    scores_b: List[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    n_train: int = 0,
    n_test: int = 0,
) -> Dict[str, Any]:
    """
    Nadeau-Bengio corrected paired t-test for k-fold CV model comparison.

    Uses the corrected resampled t-test from Nadeau & Bengio (2003),
    "Inference for the Generalization Error", Machine Learning, 52(3):239-281.

    Formula:
        t = mean_diff / sqrt((1/k + n_test/n_train) * var_diff)

    where:
        - mean_diff is the mean of the k fold-wise score differences,
        - var_diff uses ddof=1 (sample variance of k fold differences),
        - k is the number of folds,
        - n_test and n_train are the per-fold test and training set sizes.

    [Fix C2] Standard paired t-test underestimates variance due to overlapping
    training sets across CV folds. The Nadeau-Bengio correction inflates
    the variance estimate by a factor of (1/k + n_test/n_train).

    Args:
        scores_a:     List of fold scores for model A (e.g., macro F1 per fold)
        scores_b:     List of fold scores for model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        n_train:      Training set size per fold (if 0, estimated from k-fold)
        n_test:       Test set size per fold (if 0, estimated from k-fold)

    Returns:
        dict with keys:
            "test": "paired_ttest_nadeau_bengio"
            "model_a", "model_b": model names
            "mean_diff", "std_diff", "t_statistic", "p_value"
            "cohens_d", "effect_size_interpretation"
            "significant_at_005": bool
            "correction_factor": Nadeau-Bengio variance inflation factor
    """
    scores_a_arr = np.array(scores_a, dtype=float)
    scores_b_arr = np.array(scores_b, dtype=float)
    k = len(scores_a_arr)

    diff = scores_a_arr - scores_b_arr
    mean_diff = float(np.mean(diff))
    var_diff = float(np.var(diff, ddof=1))

    # Nadeau-Bengio correction: inflate variance for overlapping training sets
    # For k-fold CV: n_test = N/k, n_train = N*(k-1)/k
    if n_train == 0 or n_test == 0:
        # Estimate from k-fold structure: n_test/n_train = 1/(k-1)
        test_train_ratio = 1.0 / (k - 1)
    else:
        test_train_ratio = n_test / n_train

    correction_factor = (1.0 / k) + test_train_ratio
    corrected_var = correction_factor * var_diff

    # Corrected t-statistic
    if corrected_var > 0:
        t_stat = mean_diff / np.sqrt(corrected_var)
    else:
        t_stat = 0.0

    # Two-sided p-value with df = k-1
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=k - 1))

    std_diff = float(np.std(diff, ddof=1))
    cohens_d = float(mean_diff / std_diff) if std_diff > 0.0 else 0.0

    effect_size_label = (
        "negligible" if abs(cohens_d) < 0.2 else
        "small" if abs(cohens_d) < 0.5 else
        "medium" if abs(cohens_d) < 0.8 else
        "large"
    )

    return {
        "test": "paired_ttest_nadeau_bengio",
        "model_a": model_a_name,
        "model_b": model_b_name,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": float(t_stat),
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size_interpretation": effect_size_label,
        "significant_at_005": bool(p_value < 0.05),
        "correction_factor": correction_factor,
    }


def holm_bonferroni(
    p_values: List[Tuple[str, float]],
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    [Fix C3] Controls family-wise error rate (FWER) when performing
    multiple hypothesis tests simultaneously.

    Args:
        p_values: List of (comparison_name, p_value) tuples.
        alpha:    Desired family-wise significance level (default 0.05).

    Returns:
        List of dicts with keys:
            "comparison": comparison name
            "p_value":    original p-value
            "adjusted_alpha": Holm-corrected threshold for this comparison
            "significant": whether the comparison is significant after correction
    """
    m = len(p_values)
    sorted_pvals = sorted(p_values, key=lambda x: x[1])

    results = []
    rejected_so_far = True
    for i, (name, p) in enumerate(sorted_pvals):
        adjusted_alpha = alpha / (m - i)
        significant = rejected_so_far and (p < adjusted_alpha)
        if not significant:
            rejected_so_far = False
        results.append({
            "comparison": name,
            "p_value": float(p),
            "adjusted_alpha": float(adjusted_alpha),
            "rank": i + 1,
            "significant": significant,
        })

    return results


def binomial_test(
    n_correct_a: int,
    n_correct_b: int,
    n_total: int,
) -> Dict[str, Any]:
    """
    Binomial test comparing accuracy of two models.

    [Fix M8] Uses scipy.stats.binomtest (replaces deprecated binom_test).

    Tests the null hypothesis that both models have the same true accuracy
    given their observed correct prediction counts.

    Args:
        n_correct_a: Number of correct predictions by model A
        n_correct_b: Number of correct predictions by model B
        n_total:     Total number of test samples

    Returns:
        dict with keys:
            "test": "binomial"
            "n_correct_a", "n_correct_b", "n_total"
            "accuracy_a", "accuracy_b"
            "p_value": p-value
            "significant_at_005": bool
    """
    # Ensure integer inputs (scipy.stats.binomtest requires exact int)
    n_correct_a = int(n_correct_a)
    n_correct_b = int(n_correct_b)
    n_total = int(n_total)

    # Test whether n_correct_a follows Binomial(n_total, n_correct_b / n_total)
    p_null = n_correct_b / n_total
    result = stats.binomtest(n_correct_a, n_total, p=p_null)

    return {
        "test": "binomial",
        "n_correct_a": n_correct_a,
        "n_correct_b": n_correct_b,
        "n_total": n_total,
        "accuracy_a": n_correct_a / n_total,
        "accuracy_b": n_correct_b / n_total,
        "p_value": float(result.pvalue),
        "significant_at_005": bool(result.pvalue < 0.05),
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    method: str = "bca",
) -> Dict[str, float]:
    """
    Bootstrap confidence interval estimation with BCa correction.

    Particularly suited for out-of-domain test sets (OGLE, no CV structure).

    [Fix C2] Uses BCa (Bias-Corrected and Accelerated) bootstrap by default,
    which corrects for both bias and skewness in the bootstrap distribution.
    Falls back to percentile method if BCa computation fails.

    Args:
        y_true:           Ground truth labels
        y_pred:           Predicted labels
        metric_fn:        Function (y_true, y_pred) -> float
        n_bootstrap:      Number of bootstrap resamples (default 10000)
        confidence_level: Confidence level (default 0.95)
        random_seed:      Random seed for reproducibility
        method:           "bca" (default) or "percentile"

    Returns:
        dict with keys:
            "point_estimate": Metric value on full data
            "mean":           Mean across bootstrap samples
            "std":            Std across bootstrap samples
            "ci_{level}":     (lower, upper) confidence interval tuple
            "method":         CI method used
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    point_estimate = float(metric_fn(y_true, y_pred))

    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = 1.0 - confidence_level
    ci_key = f"ci_{confidence_level}"
    used_method = method

    if method == "bca":
        try:
            # Bias correction factor z_0
            # Clip proportion to avoid -inf/+inf from ppf(0) or ppf(1)
            prop_below = np.mean(scores < point_estimate)
            prop_below = np.clip(prop_below, 1.0 / (n_bootstrap + 1), 1.0 - 1.0 / (n_bootstrap + 1))
            z_0 = stats.norm.ppf(prop_below)

            # Acceleration factor via jackknife
            jackknife_scores = np.empty(n)
            for i in range(n):
                idx_jack = np.concatenate([np.arange(i), np.arange(i + 1, n)])
                jackknife_scores[i] = metric_fn(y_true[idx_jack], y_pred[idx_jack])
            jack_mean = jackknife_scores.mean()
            jack_diff = jack_mean - jackknife_scores
            denom = np.sum(jack_diff ** 2) ** 1.5
            a = np.sum(jack_diff ** 3) / (6.0 * denom) if denom > 0 else 0.0

            # BCa adjusted percentiles
            z_alpha_lo = stats.norm.ppf(alpha / 2.0)
            z_alpha_hi = stats.norm.ppf(1.0 - alpha / 2.0)

            def _bca_percentile(z_alpha: float) -> float:
                numer = z_0 + z_alpha
                denom_bca = 1.0 - a * numer
                if abs(denom_bca) < 1e-10:
                    denom_bca = 1e-10
                return float(stats.norm.cdf(z_0 + numer / denom_bca))

            adj_lo = _bca_percentile(z_alpha_lo)
            adj_hi = _bca_percentile(z_alpha_hi)

            # Validate BCa percentiles are finite and in range
            if not (np.isfinite(adj_lo) and np.isfinite(adj_hi)
                    and 0 < adj_lo < 1 and 0 < adj_hi < 1):
                raise ValueError("BCa percentiles out of range")

            ci = (
                float(np.percentile(scores, 100.0 * adj_lo)),
                float(np.percentile(scores, 100.0 * adj_hi)),
            )
        except (ValueError, ZeroDivisionError, FloatingPointError):
            # Fallback to percentile method
            used_method = "percentile_fallback"
            ci = (
                float(np.percentile(scores, 100.0 * alpha / 2.0)),
                float(np.percentile(scores, 100.0 * (1.0 - alpha / 2.0))),
            )
    else:
        ci = (
            float(np.percentile(scores, 100.0 * alpha / 2.0)),
            float(np.percentile(scores, 100.0 * (1.0 - alpha / 2.0))),
        )

    return {
        "point_estimate": point_estimate,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        ci_key: ci,
        "method": used_method,
    }
