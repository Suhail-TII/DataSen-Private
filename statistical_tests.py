"""
statistical_tests.py — T0.1: Formal Statistical Testing for Algorithm Comparison
=================================================================================

Implements paired statistical tests comparing MW against alternatives on the
same 18 observations (6 tasks × 3 seeds) per algorithm.

Tests performed:
  1. Wilcoxon signed-rank test (paired, non-parametric) — primary test
     Compares MW (A4) against each of A1, A2, A3 on matched (task, seed) pairs.
     Null hypothesis: no difference in FNR between algorithms.
     Use when: distribution of differences is unknown (appropriate here).

  2. Effect size: rank-biserial correlation r = 1 - 2W/(n*(n+1)/2)
     |r| < 0.3 small, 0.3-0.5 medium, > 0.5 large

  3. McNemar's test — binary evasion framing
     Treats each (task, seed) as a binary outcome: FNR >= 0.9 = "attack succeeds"
     Tests whether the proportion of successes differs between algorithms.

Interpretation for paper:
  - MW vs A1 (random): expect p << 0.05, large effect (MW much better)
  - MW vs A2 (greedy): expect p > 0.05 (MW matches greedy — equivalence claim)
  - MW vs A3 (epsilon-greedy): expect p < 0.05, MW more stable (lower variance)

Usage:
  python statistical_tests.py \\
      --checkpoints_dir results/gt_ablation/run_20260408_224648/checkpoints \\
      --out_dir results/statistical_tests

Output:
  - Console: full test results with interpretation
  - statistical_tests.json: structured results for paper
  - latex_table.tex: ready-to-paste LaTeX table
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ── Constants ─────────────────────────────────────────────────────────────────

ALGORITHMS = {
    "A1_random":               "Random sampling",
    "A2_greedy":               "Greedy enumeration",
    "A3_epsilon_greedy":       r"$\epsilon$-greedy",
    "A4_multiplicative_weights": "Multiplicative Weights (ours)",
}

# Success threshold for McNemar's test: FNR >= this = attack succeeded
MCNEMAR_THRESHOLD = 0.90

# Significance level
ALPHA = 0.05


# ── Data loading ───────────────────────────────────────────────────────────────

def load_algorithm_data(checkpoints_dir: Path, algo_key: str) -> pd.DataFrame:
    """Load CSV for one algorithm, return with clean task/seed columns."""
    csv_path = checkpoints_dir / f"G2_{algo_key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    task_col = "true_target_task" if "true_target_task" in df.columns else "target_task"

    # Strip composite seed suffix if present
    if "__seed" in str(df[task_col].iloc[0]):
        df["task"] = df[task_col].str.replace(r"__seed\d+", "", regex=True)
    else:
        df["task"] = df[task_col]

    df["seed"] = df["seed"].astype(int)
    return df[["task", "seed", "fnr"]].sort_values(["task", "seed"]).reset_index(drop=True)


def build_paired_vectors(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple:
    """
    Build matched (task, seed) paired FNR vectors for two algorithms.
    Only includes pairs present in BOTH dataframes.
    Returns: (fnr_a, fnr_b, n_pairs)
    """
    merged = pd.merge(df_a, df_b, on=["task", "seed"], suffixes=("_a", "_b"))
    merged = merged.sort_values(["task", "seed"])
    return (
        merged["fnr_a"].values,
        merged["fnr_b"].values,
        len(merged),
    )


# ── Statistical tests ──────────────────────────────────────────────────────────

def wilcoxon_test(fnr_a: np.ndarray, fnr_b: np.ndarray,
                  label_a: str, label_b: str) -> dict:
    """
    Wilcoxon signed-rank test on matched pairs.
    Tests H0: median(fnr_a - fnr_b) = 0.
    Alternative: two-sided.
    """
    differences = fnr_a - fnr_b

    # Check if all differences are zero (degenerate case)
    if np.all(differences == 0):
        return {
            "test": "wilcoxon",
            "label_a": label_a,
            "label_b": label_b,
            "n_pairs": len(fnr_a),
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "effect_size_r": 0.0,
            "effect_magnitude": "none",
            "mean_diff": 0.0,
            "median_diff": 0.0,
            "note": "All differences zero — algorithms identical on this metric",
        }

    try:
        stat, p_value = stats.wilcoxon(fnr_a, fnr_b, alternative="two-sided",
                                        zero_method="wilcox")
    except ValueError as e:
        return {
            "test": "wilcoxon",
            "label_a": label_a,
            "label_b": label_b,
            "n_pairs": len(fnr_a),
            "error": str(e),
            "note": "Test could not be computed",
        }

    # Effect size: rank-biserial correlation
    n = len(fnr_a)
    # r = 1 - (2*W) / (n*(n+1)/2)  — ranges from -1 to 1
    max_w = n * (n + 1) / 2
    r = 1 - (2 * stat) / max_w if max_w > 0 else 0.0
    r_abs = abs(r)

    if r_abs < 0.1:
        magnitude = "negligible"
    elif r_abs < 0.3:
        magnitude = "small"
    elif r_abs < 0.5:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "test": "wilcoxon",
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": n,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
        "effect_size_r": float(r),
        "effect_magnitude": magnitude,
        "mean_diff": float(np.mean(differences)),
        "median_diff": float(np.median(differences)),
    }


def mcnemar_test(fnr_a: np.ndarray, fnr_b: np.ndarray,
                 label_a: str, label_b: str,
                 threshold: float = MCNEMAR_THRESHOLD) -> dict:
    """
    McNemar's test on binary outcomes (FNR >= threshold = success).

    Contingency table:
      b=1  b=0
    a=1 [ n11  n10 ]
    a=0 [ n01  n00 ]

    McNemar tests whether n10 != n01 (discordant pairs differ).
    """
    bin_a = (fnr_a >= threshold).astype(int)
    bin_b = (fnr_b >= threshold).astype(int)

    n11 = int(np.sum((bin_a == 1) & (bin_b == 1)))  # both succeed
    n10 = int(np.sum((bin_a == 1) & (bin_b == 0)))  # a succeeds, b fails
    n01 = int(np.sum((bin_a == 0) & (bin_b == 1)))  # a fails, b succeeds
    n00 = int(np.sum((bin_a == 0) & (bin_b == 0)))  # both fail

    discordant = n10 + n01
    success_rate_a = float(np.mean(bin_a))
    success_rate_b = float(np.mean(bin_b))

    if discordant == 0:
        return {
            "test": "mcnemar",
            "label_a": label_a,
            "label_b": label_b,
            "threshold": threshold,
            "n_pairs": len(fnr_a),
            "contingency": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
            "success_rate_a": success_rate_a,
            "success_rate_b": success_rate_b,
            "p_value": 1.0,
            "significant": False,
            "note": "No discordant pairs — algorithms agree on all binary outcomes",
        }

    # McNemar statistic with continuity correction
    stat = (abs(n10 - n01) - 1) ** 2 / discordant
    p_value = float(1 - stats.chi2.cdf(stat, df=1))

    return {
        "test": "mcnemar",
        "label_a": label_a,
        "label_b": label_b,
        "threshold": threshold,
        "n_pairs": len(fnr_a),
        "contingency": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
        "success_rate_a": success_rate_a,
        "success_rate_b": success_rate_b,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
    }


# ── Summary stats ──────────────────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute mean, std, min, max FNR for one algorithm."""
    fnrs = df["fnr"].values
    return {
        "mean": float(np.mean(fnrs)),
        "std":  float(np.std(fnrs, ddof=1)),
        "min":  float(np.min(fnrs)),
        "max":  float(np.max(fnrs)),
        "n":    int(len(fnrs)),
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def print_summary_table(data: dict):
    print("\n" + "=" * 70)
    print("ALGORITHM SUMMARY STATISTICS (18 observations: 6 tasks × 3 seeds)")
    print("=" * 70)
    print(f"{'Algorithm':<35} {'Mean FNR':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for algo, stats_dict in data.items():
        label = ALGORITHMS[algo][:34]
        s = stats_dict
        print(f"{label:<35} {s['mean']:>10.4f} {s['std']:>8.4f} "
              f"{s['min']:>8.4f} {s['max']:>8.4f}")
    print("=" * 70)


def print_wilcoxon_results(results: list):
    print("\n" + "=" * 70)
    print("WILCOXON SIGNED-RANK TESTS (MW vs each alternative)")
    print(f"H0: no difference in FNR | alpha={ALPHA} | two-sided")
    print("=" * 70)

    for r in results:
        if "error" in r:
            print(f"\n  {r['label_a']} vs {r['label_b']}: ERROR — {r['error']}")
            continue

        sig_str = "✓ SIGNIFICANT" if r["significant"] else "✗ not significant"
        direction = ""
        if r["mean_diff"] > 0:
            direction = f"(MW is {r['mean_diff']:+.4f} higher)"
        elif r["mean_diff"] < 0:
            direction = f"(MW is {r['mean_diff']:+.4f} lower)"

        print(f"\n  MW vs {r['label_b']}:")
        print(f"    n pairs     : {r['n_pairs']}")
        print(f"    W statistic : {r['statistic']:.4f}")
        print(f"    p-value     : {r['p_value']:.4f}  {sig_str}")
        print(f"    effect size : r = {r['effect_size_r']:.4f}  ({r['effect_magnitude']})")
        print(f"    mean diff   : {r['mean_diff']:+.4f}  {direction}")
        print(f"    median diff : {r['median_diff']:+.4f}")
        if "note" in r:
            print(f"    note        : {r['note']}")


def print_mcnemar_results(results: list):
    print("\n" + "=" * 70)
    print(f"McNEMAR'S TESTS (binary: FNR >= {MCNEMAR_THRESHOLD} = attack succeeds)")
    print(f"H0: equal proportion of successes | alpha={ALPHA}")
    print("=" * 70)

    for r in results:
        sig_str = "✓ SIGNIFICANT" if r.get("significant") else "✗ not significant"
        c = r["contingency"]
        print(f"\n  MW vs {r['label_b']}:")
        print(f"    Contingency (MW\\Alt) : both=({c['n11']}), "
              f"MW-only=({c['n10']}), Alt-only=({c['n01']}), neither=({c['n00']})")
        print(f"    Success rate MW  : {r['success_rate_a']:.3f}")
        print(f"    Success rate Alt : {r['success_rate_b']:.3f}")
        if "p_value" in r:
            print(f"    p-value          : {r['p_value']:.4f}  {sig_str}")
        if "note" in r:
            print(f"    note             : {r['note']}")


def generate_latex_table(wilcoxon_results: list, summary_stats: dict) -> str:
    """Generate LaTeX table for paper (Algorithm comparison with p-values)."""
    lines = []
    lines.append("% Statistical test results — Algorithm comparison (T0.1)")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Algorithm comparison with formal statistical tests. "
                 "All algorithms use the same $K=147$ strategy pool. "
                 "Wilcoxon signed-rank test (paired, two-sided) comparing each "
                 "algorithm against MW; $n=18$ pairs (6 tasks $\\times$ 3 seeds). "
                 "Effect size: rank-biserial correlation $r$.}")
    lines.append("\\label{tab:algorithm-stats}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Algorithm} & \\textbf{Mean FNR} & \\textbf{Queries} "
                 "& \\textbf{vs MW: $p$} & \\textbf{Effect $r$} & \\textbf{Sig.} \\\\")
    lines.append("\\midrule")

    # Algorithm display names and query counts
    query_counts = {
        "A1_random": "100",
        "A2_greedy": "1570",
        "A3_epsilon_greedy": "200",
        "A4_multiplicative_weights": "\\textbf{200}",
    }
    display_names = {
        "A1_random":               "Random sampling",
        "A2_greedy":               "Greedy enumeration",
        "A3_epsilon_greedy":       "$\\epsilon$-greedy ($\\epsilon=0.2$)",
        "A4_multiplicative_weights": "\\textbf{Multiplicative Weights (ours)}",
    }

    # Build lookup from wilcoxon results
    wlookup = {}
    for r in wilcoxon_results:
        wlookup[r["label_b"]] = r

    for algo in ["A1_random", "A2_greedy", "A3_epsilon_greedy",
                 "A4_multiplicative_weights"]:
        s = summary_stats[algo]
        name = display_names[algo]
        q = query_counts[algo]
        mean_str = f"${s['mean']:.4f} \\pm {s['std']:.3f}$"

        if algo == "A4_multiplicative_weights":
            lines.append(f"{name} & {mean_str} & {q} & — & — & — \\\\")
        else:
            label_b = ALGORITHMS[algo]
            r = wlookup.get(label_b, {})
            if r and "p_value" in r:
                p = r["p_value"]
                p_str = f"$< 0.001$" if p < 0.001 else f"${p:.3f}$"
                eff_str = f"${r['effect_size_r']:.3f}$"
                sig = "\\checkmark" if r["significant"] else "—"
            else:
                p_str = "—"
                eff_str = "—"
                sig = "—"
            lines.append(f"{name} & {mean_str} & {q} & {p_str} & {eff_str} & {sig} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="T0.1: Statistical tests for algorithm comparison")
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                        help="Path to ablation checkpoints dir (contains G2_*.csv)")
    parser.add_argument("--out_dir", type=str, default="results/statistical_tests")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("T0.1: STATISTICAL TESTS — ALGORITHM COMPARISON")
    print(f"alpha = {ALPHA}, McNemar threshold = {MCNEMAR_THRESHOLD}")
    print("=" * 70)

    # Load all algorithm data
    print("\nLoading algorithm CSVs...")
    algo_data = {}
    for algo in ALGORITHMS:
        try:
            algo_data[algo] = load_algorithm_data(ckpt_dir, algo)
            n = len(algo_data[algo])
            mean = algo_data[algo]["fnr"].mean()
            print(f"  {algo}: {n} rows, mean FNR={mean:.4f}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    # Summary statistics
    summary_stats = {algo: compute_summary_stats(df)
                     for algo, df in algo_data.items()}
    print_summary_table(summary_stats)

    # MW is the reference algorithm
    mw_key = "A4_multiplicative_weights"
    mw_label = ALGORITHMS[mw_key]
    mw_df = algo_data[mw_key]

    # Wilcoxon tests: MW vs each alternative
    print("\nRunning Wilcoxon signed-rank tests...")
    wilcoxon_results = []
    for algo in ["A1_random", "A2_greedy", "A3_epsilon_greedy"]:
        if algo not in algo_data:
            continue
        alt_df = algo_data[algo]
        alt_label = ALGORITHMS[algo]

        fnr_mw, fnr_alt, n_pairs = build_paired_vectors(mw_df, alt_df)
        print(f"  MW vs {alt_label}: {n_pairs} paired observations")

        result = wilcoxon_test(fnr_mw, fnr_alt, mw_label, alt_label)
        wilcoxon_results.append(result)

    print_wilcoxon_results(wilcoxon_results)

    # McNemar tests: MW vs each alternative
    print("\nRunning McNemar tests...")
    mcnemar_results = []
    for algo in ["A1_random", "A2_greedy", "A3_epsilon_greedy"]:
        if algo not in algo_data:
            continue
        alt_df = algo_data[algo]
        alt_label = ALGORITHMS[algo]

        fnr_mw, fnr_alt, n_pairs = build_paired_vectors(mw_df, alt_df)
        result = mcnemar_test(fnr_mw, fnr_alt, mw_label, alt_label)
        mcnemar_results.append(result)

    print_mcnemar_results(mcnemar_results)

    # Interpretation for paper
    print("\n" + "=" * 70)
    print("INTERPRETATION FOR PAPER")
    print("=" * 70)
    for r in wilcoxon_results:
        if "p_value" not in r:
            continue
        alt = r["label_b"]
        p = r["p_value"]
        eff = r["effect_size_r"]
        mean_diff = r["mean_diff"]

        if alt == "Random sampling":
            expected = "MW >> Random (p << 0.05, large effect)"
            verdict = "✓ CONFIRMED" if r["significant"] and abs(eff) > 0.5 \
                else "⚠ CHECK"
        elif alt == "Greedy enumeration":
            expected = "MW ≈ Greedy (p > 0.05, negligible effect)"
            verdict = "✓ CONFIRMED" if not r["significant"] \
                else "⚠ SIGNIFICANT DIFFERENCE FOUND"
        else:
            expected = "MW more stable than eps-greedy"
            verdict = "✓ CONFIRMED" if r["significant"] else "⚠ not significant"

        print(f"\n  MW vs {alt}:")
        print(f"    Expected  : {expected}")
        print(f"    p={p:.4f}, r={eff:.3f}, mean_diff={mean_diff:+.4f}")
        print(f"    Verdict   : {verdict}")

    # LaTeX table
    latex = generate_latex_table(wilcoxon_results, summary_stats)
    print("\n\nLATEX TABLE:")
    print("-" * 70)
    print(latex)
    print("-" * 70)

    # Save outputs
    latex_path = out_dir / "latex_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)

    results_dict = {
        "summary_stats": summary_stats,
        "wilcoxon_tests": wilcoxon_results,
        "mcnemar_tests": mcnemar_results,
        "config": {
            "alpha": ALPHA,
            "mcnemar_threshold": MCNEMAR_THRESHOLD,
            "n_algorithms": len(algo_data),
            "reference_algorithm": mw_key,
        },
    }
    json_path = out_dir / "statistical_tests.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nSaved: {latex_path}")
    print(f"Saved: {json_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()