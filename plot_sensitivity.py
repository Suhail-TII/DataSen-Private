"""
plot_sensitivity.py — Generate three sensitivity figures for the paper
========================================================================

Produces Figures 3, 4, 5 of the paper (mirroring DataSentinel's Figures 3-5
structure):
  - Figure 3: FNR vs. T (MW learning rounds)
  - Figure 4: FNR vs. eta (MW learning rate)
  - Figure 5: FNR vs. |Phi_LLM| (paraphrase pool size)

Each figure shows:
  - Mean FNR as a solid line with markers
  - Shaded ±1 std band around the mean
  - A horizontal reference line for the main-result FNR (0.9533)
  - For Figure 4: vertical dashed line at theoretical optimal η = √(ln K / T)

USAGE (on the HPC login node or any machine with matplotlib):
  python plot_sensitivity.py \
      --sweep_json /dpc/kuex0037/Open-Prompt-Injection/results/gt_sweep/run_20260408_230232/sweep_summary.json \
      --output_dir ./figures

Outputs three PDF files and three PNG files in the output directory.
PDFs are vector (for LaTeX); PNGs are raster (for quick preview).
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Style — publication-quality defaults
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    # Remove top/right spines by default for cleaner look
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colors (colorblind-friendly)
COLOR_LINE = '#1f77b4'      # blue
COLOR_FILL = '#1f77b4'      # blue (with alpha for fill)
COLOR_REF  = '#d62728'      # red (reference line)
COLOR_OPT  = '#2ca02c'      # green (theoretical optimum)

# Main-result FNR reference (from C4 ablation)
MAIN_RESULT_FNR = 0.9533

# Figure dimensions (width, height) in inches — single-column IEEE format
FIG_SIZE = (4.0, 2.8)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_T_sweep(data, output_dir: Path):
    """Figure 3: FNR vs T (MW rounds)."""
    xs = [row['T'] for row in data]
    means = [row['mean_fnr'] for row in data]
    stds = [row['std_fnr'] for row in data]
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Shaded std band
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr,
                    color=COLOR_FILL, alpha=0.2, linewidth=0,
                    label=r'$\pm 1$ std')
    
    # Mean line with markers
    ax.plot(xs, means, 'o-', color=COLOR_LINE, linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=1.8,
            label='Mean FNR')
    
    # Horizontal reference for main result
    ax.axhline(y=MAIN_RESULT_FNR, color=COLOR_REF, linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f'Main result ({MAIN_RESULT_FNR:.3f})')
    
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(r'MW learning rounds $T$')
    ax.set_ylabel('False Negative Rate')
    ax.set_ylim(0.85, 1.02)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out = output_dir / f'fig3_sweep_T.{ext}'
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


def plot_eta_sweep(data, output_dir: Path, K: int = 147, T: int = 100):
    """
    Figure 4: FNR vs η (MW learning rate).
    Includes vertical line at theoretical optimum √(ln K / T).
    """
    xs = [row['eta'] for row in data]
    means = [row['mean_fnr'] for row in data]
    stds = [row['std_fnr'] for row in data]
    
    eta_theory = math.sqrt(math.log(K) / T)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Shaded std band
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr,
                    color=COLOR_FILL, alpha=0.2, linewidth=0,
                    label=r'$\pm 1$ std')
    
    # Mean line
    ax.plot(xs, means, 'o-', color=COLOR_LINE, linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=1.8,
            label='Mean FNR')
    
    # Theoretical optimum vertical line
    ax.axvline(x=eta_theory, color=COLOR_OPT, linestyle=':', linewidth=1.5,
               alpha=0.85,
               label=rf'Theory: $\sqrt{{\ln K/T}} \approx {eta_theory:.2f}$')
    
    # Main result horizontal line
    ax.axhline(y=MAIN_RESULT_FNR, color=COLOR_REF, linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f'Main result ({MAIN_RESULT_FNR:.3f})')
    
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x:g}' for x in xs])
    ax.set_xlabel(r'MW learning rate $\eta$')
    ax.set_ylabel('False Negative Rate')
    ax.set_ylim(0.75, 1.02)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out = output_dir / f'fig4_sweep_eta.{ext}'
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


def plot_K_sweep(data, output_dir: Path):
    """Figure 5: FNR vs |Φ_LLM|."""
    xs = [row['num_paraphrases'] for row in data]
    means = [row['mean_fnr'] for row in data]
    stds = [row['std_fnr'] for row in data]
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Shaded std band
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr,
                    color=COLOR_FILL, alpha=0.2, linewidth=0,
                    label=r'$\pm 1$ std')
    
    # Mean line
    ax.plot(xs, means, 'o-', color=COLOR_LINE, linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=1.8,
            label='Mean FNR')
    
    # Main result reference
    ax.axhline(y=MAIN_RESULT_FNR, color=COLOR_REF, linestyle='--',
               linewidth=1.2, alpha=0.7,
               label=f'Main result ({MAIN_RESULT_FNR:.3f})')
    
    # Annotate oracle saturation point — K=21 is the cap
    # Find the K=21 point if it exists
    for x, m in zip(xs, means):
        if x == 21:
            ax.annotate(
                'Oracle\nsaturation',
                xy=(21, m),
                xytext=(30, m - 0.05),
                fontsize=8,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='gray',
                                lw=0.8, alpha=0.7),
            )
            break
    
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(r'Paraphrase pool size $|\Phi_{\mathrm{LLM}}|$')
    ax.set_ylabel('False Negative Rate')
    ax.set_ylim(0.80, 1.02)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=9)
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out = output_dir / f'fig5_sweep_K.{ext}'
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


def plot_combined(summaries, output_dir: Path, K: int = 147, T: int = 100):
    """
    Optional: 3-panel combined figure (one row, three panels).
    Useful if the paper prefers a single wide figure over three small ones.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    
    eta_theory = math.sqrt(math.log(K) / T)
    
    # ── Panel 1: T ─────────────────────────────────────────────────────────
    data = summaries['S1_T']
    xs = [r['T'] for r in data]
    means = np.array([r['mean_fnr'] for r in data])
    stds = np.array([r['std_fnr'] for r in data])
    ax = axes[0]
    ax.fill_between(xs, means - stds, means + stds, color=COLOR_FILL, alpha=0.2, linewidth=0)
    ax.plot(xs, means, 'o-', color=COLOR_LINE, linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=1.8)
    ax.axhline(MAIN_RESULT_FNR, color=COLOR_REF, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(r'MW rounds $T$')
    ax.set_ylabel('False Negative Rate')
    ax.set_ylim(0.85, 1.02)
    ax.set_title('(a) Sensitivity to $T$')
    ax.grid(True, alpha=0.3, axis='y')
    
    # ── Panel 2: η ─────────────────────────────────────────────────────────
    data = summaries['S2_eta']
    xs = [r['eta'] for r in data]
    means = np.array([r['mean_fnr'] for r in data])
    stds = np.array([r['std_fnr'] for r in data])
    ax = axes[1]
    ax.fill_between(xs, means - stds, means + stds, color=COLOR_FILL, alpha=0.2, linewidth=0)
    ax.plot(xs, means, 'o-', color=COLOR_LINE, linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=1.8)
    ax.axvline(eta_theory, color=COLOR_OPT, linestyle=':', linewidth=1.5, alpha=0.85,
               label=rf'$\sqrt{{\ln K/T}} \approx {eta_theory:.2f}$')
    ax.axhline(MAIN_RESULT_FNR, color=COLOR_REF, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x:g}' for x in xs])
    ax.set_xlabel(r'Learning rate $\eta$')
    ax.set_ylim(0.75, 1.02)
    ax.set_title(r'(b) Sensitivity to $\eta$')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower center', fontsize=8, frameon=True, framealpha=0.95)
    
    # ── Panel 3: K ─────────────────────────────────────────────────────────
    data = summaries['S3_K']
    xs = [r['num_paraphrases'] for r in data]
    means = np.array([r['mean_fnr'] for r in data])
    stds = np.array([r['std_fnr'] for r in data])
    ax = axes[2]
    ax.fill_between(xs, means - stds, means + stds, color=COLOR_FILL, alpha=0.2, linewidth=0)
    ax.plot(xs, means, 'o-', color=COLOR_LINE, linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=1.8)
    ax.axhline(MAIN_RESULT_FNR, color=COLOR_REF, linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'Main ({MAIN_RESULT_FNR:.3f})')
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(r'Pool size $|\Phi_{\mathrm{LLM}}|$')
    ax.set_ylim(0.80, 1.02)
    ax.set_title(r'(c) Sensitivity to $|\Phi_{\mathrm{LLM}}|$')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        out = output_dir / f'fig_sensitivity_combined.{ext}'
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot sensitivity figures from sweep results")
    parser.add_argument("--sweep_json", type=str, required=True,
                        help="Path to sweep_summary.json produced by the aggregation script")
    parser.add_argument("--output_dir", type=str, default="./figures",
                        help="Directory to write PDF/PNG figures to")
    parser.add_argument("--K", type=int, default=147,
                        help="Strategy pool size for theoretical eta calculation")
    parser.add_argument("--T", type=int, default=100,
                        help="MW rounds for theoretical eta calculation")
    parser.add_argument("--combined", action="store_true",
                        help="Also produce a single 3-panel combined figure")
    args = parser.parse_args()
    
    sweep_json_path = Path(args.sweep_json)
    if not sweep_json_path.exists():
        raise FileNotFoundError(f"Sweep JSON not found: {sweep_json_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading sweep results from: {sweep_json_path}")
    with open(sweep_json_path) as f:
        summaries = json.load(f)
    
    print(f"\nAvailable sweeps: {list(summaries.keys())}")
    print(f"Output directory: {output_dir}\n")
    
    # ── Figure 3: T sweep ──────────────────────────────────────────────────
    if 'S1_T' in summaries:
        print("Plotting Figure 3 (T sweep)...")
        plot_T_sweep(summaries['S1_T'], output_dir)
    else:
        print("WARNING: S1_T sweep not found")
    
    # ── Figure 4: eta sweep ────────────────────────────────────────────────
    if 'S2_eta' in summaries:
        print("\nPlotting Figure 4 (eta sweep)...")
        plot_eta_sweep(summaries['S2_eta'], output_dir, K=args.K, T=args.T)
    else:
        print("WARNING: S2_eta sweep not found")
    
    # ── Figure 5: K sweep ──────────────────────────────────────────────────
    if 'S3_K' in summaries:
        print("\nPlotting Figure 5 (K sweep)...")
        plot_K_sweep(summaries['S3_K'], output_dir)
    else:
        print("WARNING: S3_K sweep not found")
    
    # ── Optional combined figure ───────────────────────────────────────────
    if args.combined and all(k in summaries for k in ['S1_T', 'S2_eta', 'S3_K']):
        print("\nPlotting combined 3-panel figure...")
        plot_combined(summaries, output_dir, K=args.K, T=args.T)
    
    print(f"\nAll figures saved to: {output_dir.absolute()}")
    print("Done!")


if __name__ == "__main__":
    main()
