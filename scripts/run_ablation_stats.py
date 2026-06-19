#!/usr/bin/env python3
"""
Compute statistical tests for concept ablation results and OGLE bootstrap CIs.

Part 1: Paired t-tests and standard deviations for leave-one-out concept ablation.
         Reads per-fold accuracy values from individual cv_results.json files.
Part 2: Bootstrap confidence intervals for OGLE cross-survey evaluation.
         Uses confusion matrices to reconstruct per-sample correctness vectors.

Outputs:
    results/supplementary/ablation_statistics.json
    results/supplementary/ogle_bootstrap_ci.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = PROJECT_ROOT / "results" / "ablation_detailed"
ABLATION_JSON = ABLATION_DIR / "all_ablation_results.json"
OGLE_PATH = PROJECT_ROOT / "results" / "supplementary" / "B8_cross_survey.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "supplementary"

# 12 concepts in canonical order
CONCEPT_NAMES = [
    "period", "amplitude", "rise_fraction", "R21", "R31", "phi21",
    "skewness", "kurtosis", "stetson_K", "period_snr", "color_bp_rp", "mean_mag",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_per_fold_accuracies(cv_results_path: Path) -> list[float]:
    """Extract per-fold val_accuracy from a cv_results.json file."""
    with open(cv_results_path) as f:
        data = json.load(f)
    fold_results = data.get("fold_results", [])
    return [fr["metrics"]["val_accuracy"] for fr in fold_results]


def load_per_fold_macro_f1(cv_results_path: Path) -> list[float]:
    """Extract per-fold val_macro_f1 from a cv_results.json file."""
    with open(cv_results_path) as f:
        data = json.load(f)
    fold_results = data.get("fold_results", [])
    return [fr["metrics"]["val_macro_f1"] for fr in fold_results]


def paired_t_test(baseline_vals: list[float], ablated_vals: list[float]) -> dict:
    """Compute paired t-test and difference statistics."""
    assert len(baseline_vals) == len(ablated_vals), (
        f"Fold count mismatch: {len(baseline_vals)} vs {len(ablated_vals)}"
    )
    diffs = [b - a for b, a in zip(baseline_vals, ablated_vals)]
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    t_stat, p_value = ttest_rel(baseline_vals, ablated_vals)
    return {
        "mean_delta": mean_diff,
        "std_delta": std_diff,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "n_folds": len(diffs),
        "per_fold_deltas": [float(d) for d in diffs],
    }


def bootstrap_accuracy_ci(
    confusion_matrix: list[list[int]],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute BCa bootstrap CI for accuracy from a confusion matrix.

    Reconstructs a per-sample correctness vector (1 = correct, 0 = wrong)
    from the confusion matrix, then applies bootstrap resampling.
    """
    cm = np.array(confusion_matrix)
    n_classes = cm.shape[0]
    # Build correctness vector: for each class i, cm[i][i] correct and
    # sum(cm[i]) - cm[i][i] incorrect predictions
    correct_vec = []
    for i in range(n_classes):
        n_correct = cm[i][i]
        n_total = int(cm[i].sum())
        n_wrong = n_total - n_correct
        correct_vec.extend([1] * n_correct + [0] * n_wrong)

    correct_vec = np.array(correct_vec, dtype=np.float64)
    n_samples = len(correct_vec)
    observed_acc = correct_vec.mean()

    rng = np.random.default_rng(seed)
    boot_accs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_accs[b] = correct_vec[idx].mean()

    # Percentile CI
    alpha = 1 - ci_level
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)
    ci_lower_pct = float(np.percentile(boot_accs, lo_pct))
    ci_upper_pct = float(np.percentile(boot_accs, hi_pct))

    # BCa correction
    # z0: bias correction
    prop_below = (boot_accs < observed_acc).mean()
    # Clamp to avoid infinite z0
    prop_below = np.clip(prop_below, 1.0 / (n_bootstrap + 1), 1.0 - 1.0 / (n_bootstrap + 1))
    z0 = float(norm_ppf(prop_below))

    # a: acceleration (jackknife)
    jack_accs = np.empty(n_samples)
    for i in range(n_samples):
        jack_accs[i] = (correct_vec.sum() - correct_vec[i]) / (n_samples - 1)
    jack_mean = jack_accs.mean()
    jack_diff = jack_mean - jack_accs
    a_hat = float((jack_diff ** 3).sum() / (6.0 * ((jack_diff ** 2).sum()) ** 1.5))
    if np.isnan(a_hat) or np.isinf(a_hat):
        a_hat = 0.0

    z_alpha_lo = norm_ppf(alpha / 2)
    z_alpha_hi = norm_ppf(1 - alpha / 2)

    def bca_quantile(z_alpha):
        num = z0 + z_alpha
        adj = z0 + num / (1 - a_hat * num)
        return float(norm_cdf(adj))

    bca_lo_p = np.clip(bca_quantile(z_alpha_lo), 0.0001, 0.9999)
    bca_hi_p = np.clip(bca_quantile(z_alpha_hi), 0.0001, 0.9999)

    ci_lower_bca = float(np.percentile(boot_accs, 100 * bca_lo_p))
    ci_upper_bca = float(np.percentile(boot_accs, 100 * bca_hi_p))

    return {
        "accuracy": float(observed_acc),
        "n_samples": n_samples,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
        "percentile_ci": [ci_lower_pct, ci_upper_pct],
        "bca_ci": [ci_lower_bca, ci_upper_bca],
        "bootstrap_mean": float(boot_accs.mean()),
        "bootstrap_std": float(boot_accs.std()),
    }


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (probit). Uses scipy if available, else approximation."""
    from scipy.stats import norm
    return float(norm.ppf(p))


def norm_cdf(z: float) -> float:
    """Normal CDF."""
    from scipy.stats import norm
    return float(norm.cdf(z))


# ---------------------------------------------------------------------------
# Part 1: Ablation statistical tests
# ---------------------------------------------------------------------------

def run_ablation_statistics() -> dict:
    """Load per-fold results and compute paired t-tests for LOO ablation."""

    # Load the aggregated ablation JSON for reference
    with open(ABLATION_JSON) as f:
        agg_data = json.load(f)

    # --- Baseline per-fold accuracies ---
    baseline_cv = ABLATION_DIR / "A0_baseline" / "hard_cbm" / "cv_results.json"
    if not baseline_cv.exists():
        print(f"ERROR: Baseline cv_results.json not found at {baseline_cv}", file=sys.stderr)
        sys.exit(1)

    baseline_accs = load_per_fold_accuracies(baseline_cv)
    baseline_f1s = load_per_fold_macro_f1(baseline_cv)
    n_folds = len(baseline_accs)
    print(f"Baseline: {n_folds}-fold CV")
    print(f"  Accuracy per fold: {[f'{a:.4f}' for a in baseline_accs]}")
    print(f"  Mean accuracy:     {np.mean(baseline_accs):.6f} +/- {np.std(baseline_accs, ddof=1):.6f}")
    print()

    # --- Leave-one-out ablation per concept ---
    loo_results = {}
    concept_dir_map = {c: ABLATION_DIR / f"A2b_no_{c}" / "hard_cbm" / "cv_results.json"
                       for c in CONCEPT_NAMES}

    per_fold_available = True
    for concept in CONCEPT_NAMES:
        cv_path = concept_dir_map[concept]
        if not cv_path.exists():
            print(f"  WARNING: cv_results.json not found for {concept}, will use aggregated stats")
            per_fold_available = False
            break

    if per_fold_available:
        print("Per-fold results available -- computing paired t-tests\n")
        for concept in CONCEPT_NAMES:
            cv_path = concept_dir_map[concept]
            concept_accs = load_per_fold_accuracies(cv_path)
            concept_f1s = load_per_fold_macro_f1(cv_path)

            acc_stats = paired_t_test(baseline_accs, concept_accs)
            f1_stats = paired_t_test(baseline_f1s, concept_f1s)

            # Reference aggregated delta from all_ablation_results.json
            agg_ref = agg_data.get("A2b_leave_one_out", {}).get(concept, {})
            agg_delta = agg_ref.get("delta_accuracy", None)

            loo_results[concept] = {
                "accuracy": {
                    "baseline_mean": float(np.mean(baseline_accs)),
                    "ablated_mean": float(np.mean(concept_accs)),
                    "mean_delta": acc_stats["mean_delta"],
                    "std_delta": acc_stats["std_delta"],
                    "t_statistic": acc_stats["t_statistic"],
                    "p_value": acc_stats["p_value"],
                    "significant_0.05": acc_stats["p_value"] < 0.05,
                    "per_fold_deltas": acc_stats["per_fold_deltas"],
                    "aggregated_delta_ref": agg_delta,
                },
                "macro_f1": {
                    "baseline_mean": float(np.mean(baseline_f1s)),
                    "ablated_mean": float(np.mean(concept_f1s)),
                    "mean_delta": f1_stats["mean_delta"],
                    "std_delta": f1_stats["std_delta"],
                    "t_statistic": f1_stats["t_statistic"],
                    "p_value": f1_stats["p_value"],
                    "significant_0.05": f1_stats["p_value"] < 0.05,
                    "per_fold_deltas": f1_stats["per_fold_deltas"],
                },
            }
    else:
        # Fallback: use aggregated stats with error propagation
        print("Using aggregated stats with error propagation (no per-fold data)\n")
        baseline_agg = agg_data["A0_baseline"]["aggregated"]
        b_mean = baseline_agg["accuracy_mean"]
        b_std = baseline_agg["accuracy_std"]

        for concept in CONCEPT_NAMES:
            c_data = agg_data.get("A2b_leave_one_out", {}).get(concept, {})
            if not c_data:
                continue
            c_mean = c_data["accuracy_mean"]
            c_std = c_data["accuracy_std"]
            delta = c_data.get("delta_accuracy", b_mean - c_mean)
            # Error propagation: std_delta ~ sqrt(std_b^2 + std_c^2)
            std_delta = float(np.sqrt(b_std ** 2 + c_std ** 2))

            loo_results[concept] = {
                "accuracy": {
                    "baseline_mean": b_mean,
                    "ablated_mean": c_mean,
                    "mean_delta": delta,
                    "std_delta_approx": std_delta,
                    "note": "Approximate std from error propagation (no per-fold data)",
                },
            }

    # --- Also include A1 (remove color) and A2 (minimal) if per-fold available ---
    extra_ablations = {}
    for label, subdir in [("A1_remove_color", "A1_no_color"), ("A2_minimal", "A2_minimal")]:
        cv_path = ABLATION_DIR / subdir / "hard_cbm" / "cv_results.json"
        if cv_path.exists():
            extra_accs = load_per_fold_accuracies(cv_path)
            extra_f1s = load_per_fold_macro_f1(cv_path)
            if len(extra_accs) == n_folds:
                acc_stats = paired_t_test(baseline_accs, extra_accs)
                f1_stats = paired_t_test(baseline_f1s, extra_f1s)
                extra_ablations[label] = {
                    "accuracy": {
                        "baseline_mean": float(np.mean(baseline_accs)),
                        "ablated_mean": float(np.mean(extra_accs)),
                        "mean_delta": acc_stats["mean_delta"],
                        "std_delta": acc_stats["std_delta"],
                        "t_statistic": acc_stats["t_statistic"],
                        "p_value": acc_stats["p_value"],
                        "significant_0.05": acc_stats["p_value"] < 0.05,
                        "per_fold_deltas": acc_stats["per_fold_deltas"],
                    },
                    "macro_f1": {
                        "baseline_mean": float(np.mean(baseline_f1s)),
                        "ablated_mean": float(np.mean(extra_f1s)),
                        "mean_delta": f1_stats["mean_delta"],
                        "std_delta": f1_stats["std_delta"],
                        "t_statistic": f1_stats["t_statistic"],
                        "p_value": f1_stats["p_value"],
                        "significant_0.05": f1_stats["p_value"] < 0.05,
                        "per_fold_deltas": f1_stats["per_fold_deltas"],
                    },
                }

    output = {
        "description": "Paired t-tests for leave-one-out concept ablation (HardCBM, 5-fold CV)",
        "n_folds": n_folds,
        "baseline_per_fold_accuracy": [float(a) for a in baseline_accs],
        "baseline_per_fold_macro_f1": [float(f) for f in baseline_f1s],
        "leave_one_out": loo_results,
    }
    if extra_ablations:
        output["extra_ablations"] = extra_ablations

    return output


# ---------------------------------------------------------------------------
# Part 2: OGLE Bootstrap CIs
# ---------------------------------------------------------------------------

def run_ogle_bootstrap() -> dict:
    """Compute bootstrap CIs for OGLE cross-survey accuracy from confusion matrices."""

    if not OGLE_PATH.exists():
        print(f"WARNING: OGLE cross-survey results not found at {OGLE_PATH}", file=sys.stderr)
        return {"error": "B8_cross_survey.json not found"}

    with open(OGLE_PATH) as f:
        ogle_data = json.load(f)

    model_eval = ogle_data.get("model_evaluation", {})
    finetune = ogle_data.get("finetune_ogle", {})

    bootstrap_results = {}

    # Zero-shot OGLE evaluation: models with confusion matrices
    print("=" * 60)
    print("OGLE Bootstrap CIs (zero-shot, from confusion matrices)")
    print("=" * 60)

    for model_name, model_data in model_eval.items():
        cm = model_data.get("confusion_matrix_ogle")
        if cm is not None:
            ci = bootstrap_accuracy_ci(cm, n_bootstrap=10000, seed=42)
            bootstrap_results[f"{model_name}_zero_shot"] = ci
            print(f"\n  {model_name} (zero-shot):")
            print(f"    Accuracy:       {ci['accuracy']:.4f}")
            print(f"    95% BCa CI:     [{ci['bca_ci'][0]:.4f}, {ci['bca_ci'][1]:.4f}]")
            print(f"    95% Percentile: [{ci['percentile_ci'][0]:.4f}, {ci['percentile_ci'][1]:.4f}]")
        else:
            # No confusion matrix -- report aggregated values
            acc_12 = model_data.get("ogle_12dim", {}).get("accuracy")
            acc_10 = model_data.get("ogle_10dim", {}).get("accuracy")
            bootstrap_results[f"{model_name}_zero_shot"] = {
                "accuracy_12dim": acc_12,
                "accuracy_10dim": acc_10,
                "note": "No confusion matrix available; cannot compute bootstrap CI.",
            }
            print(f"\n  {model_name} (zero-shot): acc_12={acc_12}, acc_10={acc_10} (no CM)")

    # Fine-tuned results: no confusion matrices available, report as-is
    print("\n" + "=" * 60)
    print("OGLE Fine-tuned results (aggregated, no per-sample data)")
    print("=" * 60)

    for model_name in ["hard_cbm", "hard_cbm_cal", "rf", "xgb"]:
        ft_data = finetune.get(model_name, {})
        if ft_data:
            zs_acc = ft_data.get("zero_shot_accuracy")
            ft_acc = ft_data.get("finetuned_accuracy")
            ft_f1 = ft_data.get("finetuned_macro_f1")
            recovery = ft_data.get("recovery")

            bootstrap_results[f"{model_name}_finetuned"] = {
                "zero_shot_accuracy": zs_acc,
                "finetuned_accuracy": ft_acc,
                "finetuned_macro_f1": ft_f1,
                "recovery": recovery,
                "note": "No per-sample predictions available; bootstrap CI not computable for fine-tuned results.",
            }
            print(f"\n  {model_name}:")
            print(f"    Zero-shot acc:  {zs_acc}")
            print(f"    Fine-tuned acc: {ft_acc}")
            print(f"    Recovery:       {recovery}")

    # N samples info
    bootstrap_results["metadata"] = {
        "n_ogle_samples": ogle_data.get("n_ogle_samples", 1200),
        "n_bootstrap": 10000,
        "ci_level": 0.95,
        "method": "BCa bootstrap (bias-corrected and accelerated) where confusion matrices available",
    }

    return bootstrap_results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(ablation_stats: dict) -> None:
    """Print a formatted summary of concept importance with +/- std and p-values."""
    print("\n" + "=" * 90)
    print("CONCEPT IMPORTANCE SUMMARY (Leave-One-Out Ablation, HardCBM)")
    print("=" * 90)
    header = f"{'Concept':<16} {'Delta Acc':>10} {'Std':>10} {'t-stat':>10} {'p-value':>12} {'Sig.':>6}"
    print(header)
    print("-" * 90)

    loo = ablation_stats.get("leave_one_out", {})
    # Sort by absolute delta (largest drop first)
    sorted_concepts = sorted(
        loo.keys(),
        key=lambda c: abs(loo[c].get("accuracy", {}).get("mean_delta", 0)),
        reverse=True,
    )

    for concept in sorted_concepts:
        acc = loo[concept].get("accuracy", {})
        delta = acc.get("mean_delta", 0)
        std = acc.get("std_delta", acc.get("std_delta_approx", float("nan")))
        t_stat = acc.get("t_statistic", float("nan"))
        p_val = acc.get("p_value", float("nan"))
        sig = "*" if acc.get("significant_0.05", False) else ""

        if not np.isnan(t_stat):
            print(f"{concept:<16} {delta:>+10.6f} {std:>10.6f} {t_stat:>10.4f} {p_val:>12.6f} {sig:>6}")
        else:
            print(f"{concept:<16} {delta:>+10.6f} {std:>10.6f} {'N/A':>10} {'N/A':>12} {sig:>6}")

    print("-" * 90)
    print("  Delta > 0 means baseline is better (removing the concept hurts performance)")
    print("  * = significant at alpha=0.05 (two-sided paired t-test)")

    # Extra ablations
    extras = ablation_stats.get("extra_ablations", {})
    if extras:
        print(f"\n{'--- Additional ablations ---':^90}")
        for label, data in extras.items():
            acc = data.get("accuracy", {})
            delta = acc.get("mean_delta", 0)
            std = acc.get("std_delta", float("nan"))
            t_stat = acc.get("t_statistic", float("nan"))
            p_val = acc.get("p_value", float("nan"))
            sig = "*" if acc.get("significant_0.05", False) else ""
            print(f"{label:<16} {delta:>+10.6f} {std:>10.6f} {t_stat:>10.4f} {p_val:>12.6f} {sig:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Ablation Statistical Tests & OGLE Bootstrap CIs")
    print("=" * 60)

    # ---- Part 1: Ablation ----
    print("\n>>> Part 1: Ablation paired t-tests\n")
    ablation_stats = run_ablation_statistics()

    out_ablation = OUTPUT_DIR / "ablation_statistics.json"
    out_ablation.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ablation, "w") as f:
        json.dump(ablation_stats, f, indent=2)
    print(f"\nSaved ablation statistics to: {out_ablation}")

    # ---- Part 2: OGLE Bootstrap ----
    print("\n>>> Part 2: OGLE bootstrap CIs\n")
    ogle_results = run_ogle_bootstrap()

    out_ogle = OUTPUT_DIR / "ogle_bootstrap_ci.json"
    with open(out_ogle, "w") as f:
        json.dump(ogle_results, f, indent=2)
    print(f"\nSaved OGLE bootstrap CIs to: {out_ogle}")

    # ---- Summary ----
    print_summary_table(ablation_stats)

    print("\nDone.")


if __name__ == "__main__":
    main()
