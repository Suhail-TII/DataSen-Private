"""
collect_phase1_results.py — Assemble Phase 1 cross-detector comparison table
=============================================================================

Run after all Phase 1 jobs complete. Reads per-detector checkpoint CSVs,
computes mean ± std FNR per task across seeds, and prints the paper table.

Usage:
  python collect_phase1_results.py --results_dir results/phase1/run_YYYYMMDD_HHMMSS

Output:
  - Console: LaTeX-ready comparison table
  - phase1_table.json: structured results for paper
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DETECTOR_ORDER = ["kad", "nllmd", "promptguard", "evd", "ssftd", "ssftd_g"]
DETECTOR_DISPLAY = {
    "kad":         "KAD",
    "nllmd":       "NLLMD",
    "promptguard": "PromptGuard",
    "evd":         "EVD",
    "ssftd":       "SSFTD",
    "ssftd_g":     "SSFTD-G",
}

# DataSentinel reference (from main paper experiments)
DATASENTINEL_REFERENCE = {
    "fnr_mean": 0.9533,
    "fnr_std":  0.090,
    "fpr_avg":  0.0,
}


def load_detector_results(results_dir: Path, detector: str):
    """Load FNR and FPR results for one detector from its checkpoint CSV."""
    ckpt_dir = results_dir / "checkpoints"
    fnr_csv = ckpt_dir / f"{detector}_fnr.csv"
    fpr_csv = ckpt_dir / f"{detector}_fpr.csv"

    if not fnr_csv.exists():
        return None

    df_fnr = pd.read_csv(fnr_csv)
    task_col = "true_target_task" if "true_target_task" in df_fnr.columns else "target_task"

    # Per-task mean/std across seeds
    per_task = {}
    for task_name, grp in df_fnr.groupby(task_col):
        per_task[task_name] = {
            "mean": float(grp["fnr"].mean()),
            "std":  float(grp["fnr"].std()) if len(grp) > 1 else 0.0,
            "n":    int(len(grp)),
        }

    overall_mean = float(df_fnr["fnr"].mean())
    overall_std  = float(df_fnr["fnr"].std()) if len(df_fnr) > 1 else 0.0

    # FPR
    fpr_avg = None
    if fpr_csv.exists():
        df_fpr = pd.read_csv(fpr_csv)
        if "fpr_avg" in df_fpr.columns:
            fpr_avg = float(df_fpr["fpr_avg"].iloc[0])

    # Also try the summary JSON
    summary_json = results_dir / f"{detector}_summary.json"
    if summary_json.exists():
        with open(summary_json) as f:
            s = json.load(f)
        fpr_avg = s.get("fpr_avg", fpr_avg)

    n_tasks = len(per_task)
    n_seeds = max((v["n"] for v in per_task.values()), default=0)

    return {
        "detector": detector,
        "fnr_mean": overall_mean,
        "fnr_std":  overall_std,
        "fpr_avg":  fpr_avg,
        "per_task": per_task,
        "n_tasks":  n_tasks,
        "n_seeds":  n_seeds,
    }


def print_comparison_table(results: dict):
    """Print the cross-detector comparison table (console + LaTeX)."""
    dets_available = [d for d in DETECTOR_ORDER if d in results]

    # Include DataSentinel as reference row
    all_dets = dets_available  # DataSentinel printed separately

    print("\n" + "=" * 75)
    print("PHASE 1: CROSS-DETECTOR GT-SEMANTIC ATTACK RESULTS")
    print("(Attack: MW + Φ_LLM, T=100, η=0.3, seeds 42/43/44)")
    print("=" * 75)

    col_w = 18
    header = f"{'Detector':<16} {'FNR (mean)':>{col_w}} {'FNR (std)':>{col_w}} {'FPR':>{col_w}} {'#Tasks':>8} {'#Seeds':>8}"
    print(header)
    print("-" * 75)

    # DataSentinel reference first
    ref = DATASENTINEL_REFERENCE
    ref_std_str = f"+-{ref['fnr_std']:.4f}"
    print(f"{'DataSentinel':<16} {ref['fnr_mean']:>{col_w}.4f} "
          f"{ref_std_str:>{col_w}} {ref['fpr_avg']:>{col_w}.4f} "
          f"{'6':>8} {'3':>8}  <- reference")
    print("-" * 75)

    for det in dets_available:
        r = results[det]
        label = DETECTOR_DISPLAY[det]
        fnr_std_str = f"±{r['fnr_std']:.4f}" if r['fnr_std'] > 0 else "—"
        fpr_str = f"{r['fpr_avg']:.4f}" if r['fpr_avg'] is not None else "—"
        print(f"{label:<16} {r['fnr_mean']:>{col_w}.4f} "
              f"{fnr_std_str:>{col_w}} {fpr_str:>{col_w}} "
              f"{r['n_tasks']:>8} {r['n_seeds']:>8}")

    print("=" * 75)

    # Per-task breakdown
    print("\n\nPER-TASK FNR BREAKDOWN (mean across seeds)")
    print("=" * 75)

    # Collect all task names
    all_tasks = set()
    for r in results.values():
        all_tasks.update(r.get("per_task", {}).keys())
    all_tasks = sorted(all_tasks)

    task_col_w = 35
    display_with_ds = {"datasentinel": "DataSentinel", **DETECTOR_DISPLAY}
    header2 = f"{'Task':<{task_col_w}}" + "".join(
        f"{display_with_ds[d]:>12}" for d in ["datasentinel"] + dets_available
    )
    print(header2)
    print("-" * (task_col_w + 12 * (1 + len(dets_available))))

    for task in all_tasks:
        row = f"{task:<{task_col_w}}"
        # DataSentinel reference: task-level not available, show overall
        row += f"{'~0.953':>12}"
        for det in dets_available:
            r = results[det]
            if task in r.get("per_task", {}):
                m = r["per_task"][task]["mean"]
                row += f"{m:>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # LaTeX table
    print("\n\nLATEX TABLE (for §6.6 Cross-Detector Evaluation):")
    print("-" * 75)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{GT-Semantic attack FNR across detection-based defenders. "
          "All attacks use MW + $\\Phi_{\\mathrm{LLM}}$ with $T=100$, $\\eta=0.3$, "
          "mean $\\pm$ std over seeds \\{42,43,44\\}.}")
    print("\\label{tab:cross_detector}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Defender} & \\textbf{FNR} (\\uparrow attack success) "
          "& \\textbf{FPR} (\\downarrow better) \\\\")
    print("\\midrule")

    ref = DATASENTINEL_REFERENCE
    print(f"DataSentinel~\\cite{{datasentinel2025}} & "
          f"${ref['fnr_mean']:.4f} \\pm {ref['fnr_std']:.3f}$ & "
          f"${ref['fpr_avg']:.4f}$ \\\\")

    for det in dets_available:
        r = results[det]
        label = DETECTOR_DISPLAY[det]
        fpr_str = f"{r['fpr_avg']:.4f}" if r['fpr_avg'] is not None else "—"
        std_str = f"{r['fnr_std']:.3f}" if r['fnr_std'] > 0 else "0.000"
        print(f"{label} & ${r['fnr_mean']:.4f} \\pm {std_str}$ & ${fpr_str}$ \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to Phase 1 run directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return

    print(f"Loading Phase 1 results from: {results_dir}")

    results = {}
    for det in DETECTOR_ORDER:
        r = load_detector_results(results_dir, det)
        if r is not None:
            results[det] = r
            fpr_display = f"{r['fpr_avg']:.4f}" if r['fpr_avg'] is not None else "--"
            print(f"  ✓ {DETECTOR_DISPLAY[det]}: "
                  f"FNR={r['fnr_mean']:.4f}+-{r['fnr_std']:.4f}, "
                  f"FPR={fpr_display}, "
                  f"{r['n_tasks']} tasks, {r['n_seeds']} seeds")
        else:
            print(f"  ✗ {DETECTOR_DISPLAY[det]}: no results found")

    if not results:
        print("No results found. Are the jobs finished?")
        return

    print_comparison_table(results)

    # Save structured output
    out_path = results_dir / "phase1_table.json"
    with open(out_path, "w") as f:
        json.dump({
            "datasentinel_reference": DATASENTINEL_REFERENCE,
            "detectors": results,
        }, f, indent=2)
    print(f"\nStructured results saved → {out_path}")


if __name__ == "__main__":
    main()