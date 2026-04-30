"""
generate_transfer_heatmap.py — 6×6 Cross-Task Transfer Heatmap
===============================================================

Reads Group 3 (T2_argmax_transfer) checkpoint CSVs from all 6 source tasks
and generates a publication-ready heatmap of argmax transfer FNR.

Each cell (row=source, col=target) shows mean FNR across 3 seeds when
the argmax strategy learned on the source task is deployed on the target task.
Diagonal is blank (source == target not evaluated).

Usage:
  python generate_transfer_heatmap.py \\
      --run_dir results/gt_ablation/run_YYYYMMDD_HHMMSS \\
      --out_dir Figures

Output:
  Figures/fig_transfer_heatmap.pdf
  Figures/fig_transfer_heatmap.png
  transfer_heatmap_data.json   (structured data for paper)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# ── Task display names (short, for axis labels) ───────────────────────────────
TASK_SHORT = {
    "Dup. sentence detection": "MRPC",
    "Grammar correction":      "Jfleg",
    "Hate detection":          "HSOL",
    "Nat. lang. inference":    "RTE",
    "Spam detection":          "SMS",
    "Summarization":           "Gigaword",
}

# Canonical display order (same as paper tables)
TASK_ORDER = [
    "Dup. sentence detection",
    "Grammar correction",
    "Hate detection",
    "Nat. lang. inference",
    "Spam detection",
    "Summarization",
]

# Source task name → dataset key used in --transfer_source
DISPLAY_TO_DATASET = {
    "Dup. sentence detection": "mrpc",
    "Grammar correction":      "jfleg",
    "Hate detection":          "hsol",
    "Nat. lang. inference":    "rte",
    "Spam detection":          "sms_spam",
    "Summarization":           "gigaword",
}


def load_transfer_matrix(run_dir: Path) -> tuple:
    """
    Load T2_argmax_transfer CSVs for all source tasks.

    The ablation script writes one CSV per (group, condition), keyed by
    condition name. For Group 3, the condition is T2_argmax_transfer and
    the 'source_task' column identifies which source was used.

    Returns:
        matrix: (6, 6) array of mean FNR, nan on diagonal
        std_matrix: (6, 6) array of std FNR
        n_matrix: (6, 6) array of seed counts
    """
    ckpt_dir = run_dir / "checkpoints"
    csv_path = ckpt_dir / "G3_T2_argmax_transfer.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"T2 CSV not found: {csv_path}\n"
            f"Make sure all 6 source task jobs have completed."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # Determine task column
    task_col = "true_target_task" if "true_target_task" in df.columns else "target_task"

    # Strip composite seed suffix from target_task if present
    if "__seed" in str(df[task_col].iloc[0]):
        df[task_col] = df[task_col].str.replace(r"__seed\d+", "", regex=True)

    n = len(TASK_ORDER)
    matrix     = np.full((n, n), np.nan)
    std_matrix = np.full((n, n), np.nan)
    n_matrix   = np.zeros((n, n), dtype=int)

    for src_idx, src_task in enumerate(TASK_ORDER):
        # Filter rows for this source task
        src_mask = df["source_task"] == src_task
        src_df = df[src_mask]

        if src_df.empty:
            print(f"  WARNING: No data for source={src_task}")
            continue

        for tgt_idx, tgt_task in enumerate(TASK_ORDER):
            if src_idx == tgt_idx:
                continue  # diagonal blank

            tgt_mask = src_df[task_col] == tgt_task
            tgt_df = src_df[tgt_mask]

            if tgt_df.empty:
                print(f"  WARNING: No data for source={src_task} -> target={tgt_task}")
                continue

            matrix[src_idx, tgt_idx]     = float(tgt_df["fnr"].mean())
            std_matrix[src_idx, tgt_idx] = float(tgt_df["fnr"].std()) \
                if len(tgt_df) > 1 else 0.0
            n_matrix[src_idx, tgt_idx]   = int(len(tgt_df))

    return matrix, std_matrix, n_matrix


def plot_heatmap(matrix: np.ndarray, std_matrix: np.ndarray,
                 out_dir: Path, show_std: bool = True):
    """
    Plot publication-ready 6×6 transfer heatmap.

    Colormap: white (0.0 FNR) → dark red (1.0 FNR)
    Diagonal: hatched gray (not evaluated)
    Annotations: mean (±std) in each cell
    """
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.labelsize":   12,
        "axes.titlesize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "text.usetex":      False,
    })

    # Custom colormap: light yellow → orange → dark red
    cmap = LinearSegmentedColormap.from_list(
        "fnr_cmap",
        [(0.0, "#ffffff"),   # 0.0 FNR = white
         (0.4, "#fdd49e"),   # 0.4 = light orange
         (0.7, "#d7301f"),   # 0.7 = red
         (1.0, "#490000")],  # 1.0 = dark red
        N=256,
    )

    n = len(TASK_ORDER)
    short_labels = [TASK_SHORT[t] for t in TASK_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    # Plot heatmap (mask diagonal)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Hatch diagonal cells
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=True, facecolor="#d0d0d0", hatch="////",
            edgecolor="#999999", linewidth=0.5, zorder=2,
        ))

    # Annotate cells
    for src_i in range(n):
        for tgt_j in range(n):
            if src_i == tgt_j:
                continue
            val = matrix[src_i, tgt_j]
            if np.isnan(val):
                ax.text(tgt_j, src_i, "—", ha="center", va="center",
                        fontsize=9, color="#888888")
                continue
            std = std_matrix[src_i, tgt_j]
            # Choose text color based on background darkness
            text_color = "white" if val > 0.65 else "black"
            if show_std and std > 0:
                label = f"{val:.2f}\n±{std:.2f}"
                fs = 8
            else:
                label = f"{val:.2f}"
                fs = 9
            ax.text(tgt_j, src_i, label, ha="center", va="center",
                    fontsize=fs, color=text_color, fontweight="normal",
                    zorder=3)

    # Axes
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Target task", labelpad=8)
    ax.set_ylabel("Source task (MW trained on)", labelpad=8)
    ax.set_title(
        "Argmax Transfer FNR: strategy learned on source, deployed on target",
        pad=10, fontsize=11,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("FNR (↑ = attack succeeds)", rotation=270, labelpad=14)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5, zorder=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig_transfer_heatmap.pdf"
    png_path = out_dir / "fig_transfer_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


def print_matrix_table(matrix: np.ndarray, std_matrix: np.ndarray):
    """Print the matrix as a console table and LaTeX tabular."""
    short = [TASK_SHORT[t] for t in TASK_ORDER]
    n = len(TASK_ORDER)

    print("\n" + "=" * 70)
    print("ARGMAX TRANSFER FNR MATRIX (mean across seeds)")
    print("Rows = source task (MW trained on), Cols = target task")
    print("=" * 70)

    # Console table
    src_tgt_label = "Source\\Target"
    header = f"{src_tgt_label:<12}" + "".join(f"{s:>10}" for s in short)
    print(header)
    print("-" * 70)
    for i, src in enumerate(TASK_ORDER):
        row = f"{short[i]:<12}"
        for j in range(n):
            if i == j:
                row += f"{'—':>10}"
            elif np.isnan(matrix[i, j]):
                row += f"{'?':>10}"
            else:
                row += f"{matrix[i, j]:>10.3f}"
        print(row)
    print("=" * 70)

    # Per-row stats
    print("\nPer-source-task statistics (excluding diagonal):")
    for i, src in enumerate(TASK_ORDER):
        row_vals = [matrix[i, j] for j in range(n) if i != j and not np.isnan(matrix[i, j])]
        if row_vals:
            print(f"  {short[i]:<10}: mean={np.mean(row_vals):.3f}  "
                  f"min={np.min(row_vals):.3f}  max={np.max(row_vals):.3f}")

    # LaTeX table
    print("\n\nLATEX TABULAR (6×6 heatmap numerical table for appendix):")
    print("-" * 70)
    col_spec = "l" + "c" * n
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    header_latex = "Source $\\backslash$ Target & " + \
                   " & ".join(f"\\textbf{{{s}}}" for s in short) + " \\\\"
    print(header_latex)
    print("\\midrule")
    for i, src in enumerate(TASK_ORDER):
        row = f"\\textbf{{{short[i]}}}"
        for j in range(n):
            if i == j:
                row += " & —"
            elif np.isnan(matrix[i, j]):
                row += " & ?"
            else:
                row += f" & {matrix[i,j]:.2f}"
        row += " \\\\"
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to ablation run dir containing all G3 CSVs")
    parser.add_argument("--out_dir", type=str, default="Figures",
                        help="Output directory for figures")
    parser.add_argument("--no_std", action="store_true",
                        help="Show only mean FNR without ±std in cells")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)

    print(f"Loading transfer data from: {run_dir}")

    matrix, std_matrix, n_matrix = load_transfer_matrix(run_dir)

    # Check completeness
    n = len(TASK_ORDER)
    missing = [(TASK_ORDER[i], TASK_ORDER[j])
               for i in range(n) for j in range(n)
               if i != j and np.isnan(matrix[i, j])]
    if missing:
        print(f"\nWARNING: {len(missing)} cells still missing data:")
        for src, tgt in missing[:10]:
            print(f"  {TASK_SHORT[src]} -> {TASK_SHORT[tgt]}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
        print("Run remaining source tasks before generating final figure.\n")

    # Print tables
    print_matrix_table(matrix, std_matrix)

    # Generate figure
    plot_heatmap(matrix, std_matrix, out_dir, show_std=not args.no_std)

    # Save structured data
    data = {
        "task_order": TASK_ORDER,
        "task_short": TASK_SHORT,
        "matrix": {
            TASK_ORDER[i]: {
                TASK_ORDER[j]: {
                    "mean": float(matrix[i, j]) if not np.isnan(matrix[i, j]) else None,
                    "std":  float(std_matrix[i, j]) if not np.isnan(std_matrix[i, j]) else None,
                    "n":    int(n_matrix[i, j]),
                }
                for j in range(n) if i != j
            }
            for i in range(n)
        },
        "aggregate": {
            "overall_mean": float(np.nanmean(matrix)),
            "overall_std":  float(np.nanstd(matrix)),
        },
    }
    json_path = run_dir / "transfer_heatmap_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nStructured data saved -> {json_path}")


if __name__ == "__main__":
    main()