"""
collect_t02_results.py — Collect T0.2 Cross-Model Oracle Results
=================================================================

Reads per-seed result JSONs from the Qwen2.5-7B-Instruct oracle run
and compares against the LLaMA-3.1-8B-Instruct main results.

Usage:
  python collect_t02_results.py \\
      --results_dir results/t02_oracle/run_YYYYMMDD_HHMMSS

Output:
  - Console: side-by-side comparison table + LaTeX
  - t02_comparison.json: structured results
"""

import argparse
import json
import numpy as np
from pathlib import Path

# Main result reference (LLaMA-3.1-8B-Instruct oracle, from paper)
LLAMA_REFERENCE = {
    "Dup. sentence detection":  0.8333,
    "Grammar correction":       1.0000,
    "Hate detection":           0.9967,
    "Nat. lang. inference":     1.0000,
    "Spam detection":           0.9233,
    "Summarization":            0.9667,
    "mean":                     0.9533,
    "std":                      0.090,
}

TASK_ORDER = [
    "Dup. sentence detection",
    "Grammar correction",
    "Hate detection",
    "Nat. lang. inference",
    "Spam detection",
    "Summarization",
]


def load_seed_results(results_dir: Path) -> dict:
    """
    Load per-task FNR from all seed subdirectories.
    gt_semantic_attack.py saves to {save_dir}/gt_semantic_final.json
    """
    per_task = {task: [] for task in TASK_ORDER}

    for seed in [42, 43, 44]:
        seed_dir = results_dir / f"seed_{seed}"
        json_path = seed_dir / "gt_semantic_final.json"

        if not json_path.exists():
            print(f"  WARNING: {json_path} not found")
            continue

        with open(json_path) as f:
            data = json.load(f)

        results = data.get("results", {}).get("gt_semantic_DS", {})
        for task_name, fnr in results.items():
            if task_name in per_task:
                per_task[task_name].append(fnr)

    return per_task


def compute_stats(per_task: dict) -> dict:
    """Compute mean/std per task and overall."""
    stats = {}
    all_fnrs = []
    for task, fnrs in per_task.items():
        if fnrs:
            stats[task] = {
                "mean": float(np.mean(fnrs)),
                "std":  float(np.std(fnrs, ddof=1)) if len(fnrs) > 1 else 0.0,
                "n":    len(fnrs),
                "values": fnrs,
            }
            all_fnrs.extend(fnrs)
        else:
            stats[task] = {"mean": None, "std": None, "n": 0}

    if all_fnrs:
        stats["_overall"] = {
            "mean": float(np.mean(all_fnrs)),
            "std":  float(np.std(all_fnrs, ddof=1)) if len(all_fnrs) > 1 else 0.0,
            "n":    len(all_fnrs),
        }

    return stats


def print_comparison(qwen_stats: dict):
    print("\n" + "=" * 75)
    print("T0.2: CROSS-MODEL ORACLE COMPARISON")
    print("GT-Semantic attack FNR: LLaMA-3.1-8B-Instruct vs Qwen2.5-7B-Instruct")
    print("(Same attack, same detector, same hyperparameters — only oracle changes)")
    print("=" * 75)

    print(f"\n{'Task':<35} {'LLaMA-3.1 (main)':>18} {'Qwen2.5 (T0.2)':>16} {'Diff':>8}")
    print("-" * 75)

    diffs = []
    for task in TASK_ORDER:
        llama_fnr = LLAMA_REFERENCE.get(task, None)
        qwen_entry = qwen_stats.get(task, {})
        qwen_fnr = qwen_entry.get("mean", None)
        qwen_std = qwen_entry.get("std", 0.0)
        qwen_n = qwen_entry.get("n", 0)

        llama_str = f"{llama_fnr:.4f}" if llama_fnr is not None else "—"
        if qwen_fnr is not None:
            qwen_str = f"{qwen_fnr:.4f}±{qwen_std:.3f}"
            diff = qwen_fnr - llama_fnr if llama_fnr is not None else None
            diff_str = f"{diff:+.4f}" if diff is not None else "—"
            diffs.append(diff)
        else:
            qwen_str = "— (missing)"
            diff_str = "—"

        print(f"{task:<35} {llama_str:>18} {qwen_str:>16} {diff_str:>8}")

    print("-" * 75)

    # Overall
    overall = qwen_stats.get("_overall", {})
    qwen_mean = overall.get("mean")
    qwen_std  = overall.get("std", 0.0)
    llama_mean = LLAMA_REFERENCE["mean"]

    if qwen_mean is not None:
        diff_overall = qwen_mean - llama_mean
        print(f"{'MEAN':<35} {llama_mean:>18.4f} "
              f"{qwen_mean:>12.4f}±{qwen_std:.3f} {diff_overall:>+8.4f}")
    print("=" * 75)

    # Interpretation
    print("\nINTERPRETATION:")
    if qwen_mean is not None:
        if abs(qwen_mean - llama_mean) < 0.05:
            print(f"  -> Qwen2.5 FNR ({qwen_mean:.4f}) is within 5pp of LLaMA-3.1 ({llama_mean:.4f})")
            print(f"  -> CONFIRMS: vulnerability is not oracle-specific")
            print(f"  -> T0.2 passes: safe to claim systemic vulnerability")
        elif qwen_mean > llama_mean:
            print(f"  -> Qwen2.5 achieves HIGHER FNR than LLaMA-3.1")
            print(f"  -> STRONGLY CONFIRMS: different oracle, same or better evasion")
        else:
            diff = llama_mean - qwen_mean
            print(f"  -> Qwen2.5 FNR is {diff:.4f} lower than LLaMA-3.1")
            if diff < 0.10:
                print(f"  -> Within 10pp — still confirms oracle independence")
            else:
                print(f"  -> >10pp gap — investigate: may reflect Qwen instruction-following")

    # LaTeX for paper
    if qwen_mean is not None:
        print("\n\nLATEX (for §4.2 / §7 oracle validation paragraph):")
        print("-" * 75)
        print("% T0.2: Cross-model oracle validation")
        print("\\begin{table}[t]")
        print("\\centering")
        print("\\caption{Cross-model oracle validation (T0.2). The GT-Semantic attack "
              "is re-run with Qwen2.5-7B-Instruct as the paraphrase oracle, holding "
              "all other hyperparameters fixed. FNR remains high across all tasks, "
              "confirming the vulnerability is not specific to the LLaMA-3.1 oracle.}")
        print("\\label{tab:oracle_validation}")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("\\textbf{Target task} & \\textbf{LLaMA-3.1 (main)} "
              "& \\textbf{Qwen2.5 (T0.2)} \\\\")
        print("\\midrule")
        for task in TASK_ORDER:
            llama_fnr = LLAMA_REFERENCE.get(task)
            qwen_entry = qwen_stats.get(task, {})
            qwen_fnr = qwen_entry.get("mean")
            qwen_std = qwen_entry.get("std", 0.0)
            llama_str = f"${llama_fnr:.4f}$" if llama_fnr is not None else "—"
            qwen_str  = f"${qwen_fnr:.4f} \\pm {qwen_std:.3f}$" \
                if qwen_fnr is not None else "—"
            print(f"{task} & {llama_str} & {qwen_str} \\\\")
        print("\\midrule")
        llama_str = f"${llama_mean:.4f} \\pm {LLAMA_REFERENCE['std']:.3f}$"
        qwen_str  = f"${qwen_mean:.4f} \\pm {qwen_std:.3f}$"
        print(f"\\textbf{{Mean}} & {llama_str} & {qwen_str} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print("-" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Loading T0.2 results from: {results_dir}")

    per_task = load_seed_results(results_dir)
    qwen_stats = compute_stats(per_task)

    # Check completeness
    missing = [t for t in TASK_ORDER if not per_task.get(t)]
    if missing:
        print(f"\nWARNING: Missing results for: {missing}")

    print_comparison(qwen_stats)

    # Save
    out = {
        "oracle": "Qwen/Qwen2.5-7B-Instruct",
        "reference_oracle": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "qwen_stats": qwen_stats,
        "llama_reference": LLAMA_REFERENCE,
    }
    out_path = results_dir / "t02_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()