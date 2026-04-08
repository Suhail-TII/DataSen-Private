"""
gt_semantic_sweep.py — Hyperparameter Sweep for GT-Semantic Attack
====================================================================

Sweeps the three most paper-relevant hyperparameters of the game-theoretic
attack and produces sensitivity curves analogous to DataSentinel's Figures 3-5.

SWEEPS
─────────────────────────────────────────────────────────────────────────────

  S1. MW LEARNING ROUNDS (T)        — does convergence happen? Is 100 enough?
      T ∈ {25, 50, 100, 200, 400}
      Fixed: η=0.3, num_paraphrases=20
      Plot: avg FNR vs T  →  expected to plateau, validating the no-regret bound

  S2. MW LEARNING RATE (η)          — convergence speed vs. stability tradeoff
      η ∈ {0.1, 0.2, 0.3, 0.5, 1.0}
      Fixed: T=100, num_paraphrases=20
      Plot: avg FNR vs η  →  sweet spot expected near theory's √(ln K / T)

  S3. STRATEGY POOL SIZE (K)        — does action-space expansion help monotonically?
      num_paraphrases ∈ {5, 10, 20, 30, 50}
      Fixed: T=100, η=0.3
      Plot: avg FNR vs |Φ_LLM|  →  expected to grow then saturate

For each sweep, all 6 target tasks are evaluated under the swept value.
Each (sweep, value, target_task) is independently checkpointed.

USAGE
─────────────────────────────────────────────────────────────────────────────
  python gt_semantic_sweep.py \
      --ckpt_detector DataSentinel_Models/detector_large/checkpoint-5000/ \
      --paraphrase_model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --repo_dir . --data_num 100 \
      --sweeps T,eta,K \
      --save_dir results/gt_sweep \
      --resume

  # Run only one sweep
  python gt_semantic_sweep.py ... --sweeps T

  # Override sweep values
  python gt_semantic_sweep.py ... --sweeps T --T_values 50,100,200
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch

# Imports from gt_semantic_attack.py (must sit alongside this file)
from gt_semantic_attack import (
    DATASETS, DATASET_TO_TASK_NAME, INJECT_DATASET,
    DATASENTINEL_SEPARATORS,
    Strategy, MWState,
    set_seed,
    load_all_tasks, load_detector, load_paraphrase_model,
    generate_paraphrase_pool, build_strategy_pool, apply_strategy,
    multiplicative_weights_learn, exploit_learned_strategy,
)

# Reuse the CheckpointManager from the ablation script
from gt_semantic_ablation import CheckpointManager


# ─────────────────────────────────────────────────────────────────────────────
# Default sweep values
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_T_VALUES = [25, 50, 100, 200, 400]
DEFAULT_ETA_VALUES = [0.1, 0.2, 0.3, 0.5, 1.0]
DEFAULT_K_VALUES = [5, 10, 20, 30, 50]   # num_paraphrases values


# ═════════════════════════════════════════════════════════════════════════════
#  CORE: Run one (T, η, num_para) configuration on one target task
# ═════════════════════════════════════════════════════════════════════════════

def run_single_config(
    detector, paraphrases: List[str],
    target_task, target_dataset: str,
    T: int, eta: float, data_num: int,
) -> Dict:
    """
    Run the game-theoretic attack with one specific (T, η, |Φ_LLM|) configuration.
    Returns a result row including the swept hyperparameters.
    """
    strategies = build_strategy_pool(DATASENTINEL_SEPARATORS, paraphrases)
    K = len(strategies)
    
    mw_state = multiplicative_weights_learn(
        detector, strategies, target_task, data_num,
        T=T, eta=eta, verbose=False,
    )
    fnr, _ = exploit_learned_strategy(
        detector, strategies, mw_state, target_task, data_num, mode="best",
    )
    
    regret_bound = math.sqrt(math.log(K) / max(T, 1))
    
    return {
        "target_task": DATASET_TO_TASK_NAME[target_dataset],
        "T": T,
        "eta": eta,
        "num_paraphrases": len(paraphrases),
        "K": K,
        "regret_bound": regret_bound,
        "fnr": fnr,
        "best_strategy_idx": int(mw_state.best_strategy_idx()),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SWEEPS
# ═════════════════════════════════════════════════════════════════════════════

def sweep_T(
    detector, para_model, para_tok, device: str,
    target_tasks_dict: Dict, target_datasets: List[str],
    inject_instr: str, inject_data: str,
    T_values: List[int], eta_fixed: float, num_paraphrases_fixed: int,
    data_num: int, save_dir: Path, resume: bool, seeds: List[int],
):
    """S1: Sweep MW learning rounds T (across multiple seeds)."""
    print("\n" + "=" * 75)
    print(f"SWEEP S1: MW Learning Rounds (T)")
    print(f"  T values: {T_values}")
    print(f"  Fixed: η={eta_fixed}, |Φ_LLM|={num_paraphrases_fixed}")
    print(f"  Seeds:  {seeds}")
    print("=" * 75)
    
    ckpt = CheckpointManager(save_dir, "S1_T", "sweep")
    done = ckpt.get_completed_tasks() if resume else set()
    
    for seed in seeds:
        set_seed(seed)
        print(f"\n  === seed={seed} ===")
        
        # Generate paraphrase pool ONCE per seed — same Φ_LLM across all T values
        # for fair comparison within a seed. Different across seeds (intentional —
        # we want to measure variance from both Φ_LLM and MW sampling).
        print(f"  Generating Φ_LLM (|Φ|={num_paraphrases_fixed}) for seed={seed}...")
        paraphrases = generate_paraphrase_pool(
            para_model, para_tok, device, inject_instr, inject_data,
            num_paraphrases=num_paraphrases_fixed,
        )
        
        for T in T_values:
            for dataset in target_datasets:
                task_name = DATASET_TO_TASK_NAME[dataset]
                done_key = f"{task_name}__T{T}__seed{seed}"
                
                if done_key in done:
                    print(f"  ⊙ T={T}, {task_name}, seed={seed} (skipped — checkpoint)")
                    continue
                
                print(f"  → T={T}, {task_name}, seed={seed}...", end=" ", flush=True)
                try:
                    row = run_single_config(
                        detector, paraphrases, target_tasks_dict[dataset], dataset,
                        T=T, eta=eta_fixed, data_num=data_num,
                    )
                    row["sweep"] = "S1_T"
                    row["seed"] = seed
                    row["target_task"] = done_key   # composite key for checkpoint resume
                    row["true_target_task"] = task_name
                    ckpt.save_row(row)
                    print(f"FNR={row['fnr']:.4f}")
                except Exception as e:
                    print(f"FAILED — {e}")
                    import traceback; traceback.print_exc()


def sweep_eta(
    detector, para_model, para_tok, device: str,
    target_tasks_dict: Dict, target_datasets: List[str],
    inject_instr: str, inject_data: str,
    eta_values: List[float], T_fixed: int, num_paraphrases_fixed: int,
    data_num: int, save_dir: Path, resume: bool, seeds: List[int],
):
    """S2: Sweep MW learning rate η (across multiple seeds)."""
    print("\n" + "=" * 75)
    print(f"SWEEP S2: MW Learning Rate (η)")
    print(f"  η values: {eta_values}")
    print(f"  Fixed: T={T_fixed}, |Φ_LLM|={num_paraphrases_fixed}")
    print(f"  Seeds:  {seeds}")
    print("=" * 75)
    
    ckpt = CheckpointManager(save_dir, "S2_eta", "sweep")
    done = ckpt.get_completed_tasks() if resume else set()
    
    for seed in seeds:
        set_seed(seed)
        print(f"\n  === seed={seed} ===")
        
        print(f"  Generating Φ_LLM (|Φ|={num_paraphrases_fixed}) for seed={seed}...")
        paraphrases = generate_paraphrase_pool(
            para_model, para_tok, device, inject_instr, inject_data,
            num_paraphrases=num_paraphrases_fixed,
        )
        
        for eta in eta_values:
            for dataset in target_datasets:
                task_name = DATASET_TO_TASK_NAME[dataset]
                done_key = f"{task_name}__eta{eta}__seed{seed}"
                
                if done_key in done:
                    print(f"  ⊙ η={eta}, {task_name}, seed={seed} (skipped — checkpoint)")
                    continue
                
                print(f"  → η={eta}, {task_name}, seed={seed}...", end=" ", flush=True)
                try:
                    row = run_single_config(
                        detector, paraphrases, target_tasks_dict[dataset], dataset,
                        T=T_fixed, eta=eta, data_num=data_num,
                    )
                    row["sweep"] = "S2_eta"
                    row["seed"] = seed
                    row["target_task"] = done_key
                    row["true_target_task"] = task_name
                    ckpt.save_row(row)
                    print(f"FNR={row['fnr']:.4f}")
                except Exception as e:
                    print(f"FAILED — {e}")
                    import traceback; traceback.print_exc()


def sweep_K(
    detector, para_model, para_tok, device: str,
    target_tasks_dict: Dict, target_datasets: List[str],
    inject_instr: str, inject_data: str,
    K_values: List[int], T_fixed: int, eta_fixed: float,
    data_num: int, save_dir: Path, resume: bool, seeds: List[int],
):
    """S3: Sweep strategy pool size via num_paraphrases (across multiple seeds)."""
    print("\n" + "=" * 75)
    print(f"SWEEP S3: Strategy Pool Size (|Φ_LLM|)")
    print(f"  num_paraphrases values: {K_values}")
    print(f"  Fixed: T={T_fixed}, η={eta_fixed}")
    print(f"  Seeds:  {seeds}")
    print("=" * 75)
    
    ckpt = CheckpointManager(save_dir, "S3_K", "sweep")
    done = ckpt.get_completed_tasks() if resume else set()
    
    for seed in seeds:
        set_seed(seed)
        print(f"\n  === seed={seed} ===")
        
        # Generate the LARGEST pool once per seed, then truncate for smaller K values.
        # Smaller pools are subsets of the larger ones (cleaner monotonicity analysis).
        max_K = max(K_values)
        print(f"  Generating max paraphrase pool (|Φ|={max_K}) for seed={seed}...")
        full_paraphrases = generate_paraphrase_pool(
            para_model, para_tok, device, inject_instr, inject_data,
            num_paraphrases=max_K,
        )
        print(f"  Generated {len(full_paraphrases)} paraphrases")
        
        for K in K_values:
            # Truncate to first K paraphrases
            paraphrases = full_paraphrases[:K]
            
            for dataset in target_datasets:
                task_name = DATASET_TO_TASK_NAME[dataset]
                done_key = f"{task_name}__K{K}__seed{seed}"
                
                if done_key in done:
                    print(f"  ⊙ K={K}, {task_name}, seed={seed} (skipped — checkpoint)")
                    continue
                
                print(f"  → K={K}, {task_name}, seed={seed}...", end=" ", flush=True)
                try:
                    row = run_single_config(
                        detector, paraphrases, target_tasks_dict[dataset], dataset,
                        T=T_fixed, eta=eta_fixed, data_num=data_num,
                    )
                    row["sweep"] = "S3_K"
                    row["seed"] = seed
                    row["target_task"] = done_key
                    row["true_target_task"] = task_name
                    ckpt.save_row(row)
                    print(f"FNR={row['fnr']:.4f}")
                except Exception as e:
                    print(f"FAILED — {e}")
                    import traceback; traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_sweep_results(save_dir: Path):
    """Aggregate sweep results into summary tables with mean ± std across seeds."""
    print("\n" + "=" * 75)
    print("SWEEP ANALYSIS (mean ± std across seeds)")
    print("=" * 75)
    
    summaries = {}
    
    for sweep_name, swept_param in [("S1_T", "T"), ("S2_eta", "eta"), ("S3_K", "num_paraphrases")]:
        csv_path = save_dir / "checkpoints" / f"{sweep_name}_sweep.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        
        has_seeds = "seed" in df.columns
        
        print(f"\n{sweep_name} — sweep over {swept_param}:")
        if has_seeds:
            n_seeds = df["seed"].nunique()
            print(f"  ({n_seeds} seed(s): {sorted(df['seed'].unique().tolist())})")
            print(f"{'Value':>10} {'Mean FNR':>12} {'Std':>10} {'Min':>10} {'Max':>10} {'#Tasks':>8} {'#Seeds':>8}")
            print("-" * 75)
        else:
            print(f"{'Value':>10} {'Avg FNR':>10} {'Min FNR':>10} {'Max FNR':>10} {'N tasks':>10}")
            print("-" * 55)
        
        sweep_summary = []
        for value in sorted(df[swept_param].unique()):
            subset = df[df[swept_param] == value]
            avg = float(subset["fnr"].mean())
            std = float(subset["fnr"].std()) if len(subset) > 1 else 0.0
            mn = float(subset["fnr"].min())
            mx = float(subset["fnr"].max())
            
            if has_seeds:
                # Per-task mean across seeds, then average across tasks
                task_col = "true_target_task" if "true_target_task" in subset.columns else "target_task"
                per_task_means = subset.groupby(task_col)["fnr"].mean()
                n_tasks = len(per_task_means)
                n_seeds_here = subset["seed"].nunique()
                std_str = f"±{std:.4f}" if std > 0 else "—"
                print(f"{value:>10} {avg:>12.4f} {std_str:>10} {mn:>10.4f} {mx:>10.4f} {n_tasks:>8} {n_seeds_here:>8}")
                sweep_summary.append({
                    swept_param: float(value),
                    "mean_fnr": avg, "std_fnr": std,
                    "min_fnr": mn, "max_fnr": mx,
                    "n_tasks": int(n_tasks), "n_seeds": int(n_seeds_here),
                    "n_rows": len(subset),
                })
            else:
                n = len(subset)
                print(f"{value:>10} {avg:>10.4f} {mn:>10.4f} {mx:>10.4f} {n:>10}")
                sweep_summary.append({
                    swept_param: float(value), "avg_fnr": avg,
                    "min_fnr": mn, "max_fnr": mx, "n_tasks": int(n),
                })
        
        summaries[sweep_name] = sweep_summary
    
    # Save summary
    with open(save_dir / "sweep_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    
    print(f"\nSweep summary saved to: {save_dir}/sweep_summary.json")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GT-Semantic Attack Hyperparameter Sweep")
    parser.add_argument("--ckpt_detector", type=str, required=True)
    parser.add_argument("--paraphrase_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--repo_dir", type=str, default=".")
    parser.add_argument("--data_num", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/gt_sweep")
    
    parser.add_argument("--sweeps", type=str, default="T,eta,K",
                        help="Which sweeps to run: T, eta, K (comma-separated)")
    
    # Sweep value overrides
    parser.add_argument("--T_values", type=str, default=",".join(map(str, DEFAULT_T_VALUES)))
    parser.add_argument("--eta_values", type=str, default=",".join(map(str, DEFAULT_ETA_VALUES)))
    parser.add_argument("--K_values", type=str, default=",".join(map(str, DEFAULT_K_VALUES)))
    
    # Fixed values when not the swept variable
    parser.add_argument("--T_fixed", type=int, default=100)
    parser.add_argument("--eta_fixed", type=float, default=0.3)
    parser.add_argument("--num_paraphrases_fixed", type=int, default=20)
    
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip running, just aggregate existing results")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated list of seeds (e.g. '42,43,44') for mean±std reporting")
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    sweeps_to_run = [s.strip() for s in args.sweeps.split(",")]
    T_values = [int(x) for x in args.T_values.split(",")]
    eta_values = [float(x) for x in args.eta_values.split(",")]
    K_values = [int(x) for x in args.K_values.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    print("=" * 75)
    print("GT-SEMANTIC ATTACK — HYPERPARAMETER SWEEP")
    print("=" * 75)
    print(f"  Sweeps     : {sweeps_to_run}")
    print(f"  T values   : {T_values}")
    print(f"  η values   : {eta_values}")
    print(f"  K values   : {K_values}")
    print(f"  Fixed T    : {args.T_fixed}")
    print(f"  Fixed η    : {args.eta_fixed}")
    print(f"  Fixed |Φ|  : {args.num_paraphrases_fixed}")
    print(f"  Seeds      : {seeds}")
    print(f"  data_num   : {args.data_num}")
    print(f"  Save dir   : {save_dir}")
    print(f"  Resume     : {args.resume}")
    print("=" * 75)
    
    if args.analyze_only:
        analyze_sweep_results(save_dir)
        return
    
    # ── Load data and models ────────────────────────────────────────────────
    print("\n[Setup] Loading data...")
    target_tasks, inject_tasks = load_all_tasks(args.repo_dir, args.data_num)
    
    print("\n[Setup] Loading detector...")
    detector = load_detector(args.ckpt_detector, args.device)
    
    print("\n[Setup] Loading paraphrase model...")
    para_model, para_tok = load_paraphrase_model(args.paraphrase_model, args.device)
    
    inject_task = inject_tasks[INJECT_DATASET]
    inject_instr = inject_task.get_instruction()
    inject_data_0, _ = inject_task[0]
    
    target_datasets = [d for d in DATASETS if d != INJECT_DATASET]
    
    # ── Run sweeps ──────────────────────────────────────────────────────────
    if "T" in sweeps_to_run:
        sweep_T(
            detector, para_model, para_tok, args.device,
            target_tasks, target_datasets, inject_instr, inject_data_0,
            T_values=T_values, eta_fixed=args.eta_fixed,
            num_paraphrases_fixed=args.num_paraphrases_fixed,
            data_num=args.data_num, save_dir=save_dir, resume=args.resume,
            seeds=seeds,
        )
    
    if "eta" in sweeps_to_run:
        sweep_eta(
            detector, para_model, para_tok, args.device,
            target_tasks, target_datasets, inject_instr, inject_data_0,
            eta_values=eta_values, T_fixed=args.T_fixed,
            num_paraphrases_fixed=args.num_paraphrases_fixed,
            data_num=args.data_num, save_dir=save_dir, resume=args.resume,
            seeds=seeds,
        )
    
    if "K" in sweeps_to_run:
        sweep_K(
            detector, para_model, para_tok, args.device,
            target_tasks, target_datasets, inject_instr, inject_data_0,
            K_values=K_values, T_fixed=args.T_fixed, eta_fixed=args.eta_fixed,
            data_num=args.data_num, save_dir=save_dir, resume=args.resume,
            seeds=seeds,
        )
    
    # ── Analyze ─────────────────────────────────────────────────────────────
    analyze_sweep_results(save_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()