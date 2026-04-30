"""
phase2_adaptive_defender.py — Adaptive Defender Experiment (Phase 2)
=====================================================================

Tests whether DataSentinel can be patched against the GT-Semantic attack
by augmenting its training pool with the discovered semantic paraphrases,
and whether the patched detector remains vulnerable to fresh paraphrases.

Protocol (3 rounds):
  Round 0: Published checkpoint (FNR=0.9533, from Phase 1 — no retraining)
  Round 1: Retrain(checkpoint-5000 + P0) → Attack(fresh seeds 45,46,47) → FNR1, P1
  Round 2: Retrain(Round1 ckpt + P0+P1) → Attack(fresh seeds 48,49,50) → FNR2, P2

Design decisions:
  - Warm start: each round continues from previous round's checkpoint
  - Augmentation pool: cumulative (P0 in Round 1, P0+P1 in Round 2)
  - Fresh attack seeds each round: ensures paraphrases not seen during retraining
  - Same hyperparameters as Phase 1: T=100, eta=0.3, |Phi|=20

Paraphrase extraction:
  Phase 1 does not save paraphrase text to disk (only best_strategy_idx).
  Phase 2 re-generates the paraphrase pool with Phase 1 seeds (42,43,44)
  and extracts the top-K paraphrases by MW weight — deterministic given
  the same seed and oracle.

Usage:
  python phase2_adaptive_defender.py \\
      --phase1_results_dir results/phase1/run_YYYYMMDD_HHMMSS \\
      --base_ckpt DataSentinel_Models/detector_large/checkpoint-5000 \\
      --paraphrase_model meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --detection_model mistralai/Mistral-7B-v0.1 \\
      --backend_model meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --repo_dir . --data_num 100 \\
      --save_dir results/phase2 \\
      --resume
"""

import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd
import torch


# ── Import building blocks ────────────────────────────────────────────────────
from gt_semantic_attack import (
    DATASETS, DATASET_TO_TASK_NAME, INJECT_DATASET,
    DATASENTINEL_SEPARATORS,
    set_seed,
    setup_repo, load_all_tasks, load_paraphrase_model,
    generate_paraphrase_pool, build_strategy_pool, apply_strategy,
    multiplicative_weights_learn, exploit_learned_strategy,
    load_detector,
)
from train_datasentinel import (
    load_detection_llm, load_backend_llm, attach_lora,
    load_gigaword_samples, train_datasentinel,
    build_X_c, update_llm, opt_ctd,
    preprocess_data, format_detection_prompt,
    SECRET_KEY, INJECTED_INSTR,
    N_IN, N_OUT, B_IN, B_OUT, LR_OUT, BETA, R, SEP_LEN,
)

TARGET_DATASETS = [d for d in DATASETS if d != INJECT_DATASET]

# Seeds: Phase 1 used 42,43,44. Phase 2 uses fresh seeds per attack round.
PHASE1_SEEDS   = [42, 43, 44]
ROUND1_SEEDS   = [45, 46, 47]
ROUND2_SEEDS   = [48, 49, 50]

# How many top paraphrases to extract per (task, seed) for augmentation
TOP_K_PARAPHRASES = 5


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint manager
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, output_dir: Path, name: str):
        self.ckpt_dir = Path(output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.ckpt_dir / f"{name}.csv"
        self._header_written = self.csv_path.exists()

    def get_completed_tasks(self) -> Set[str]:
        if not self.csv_path.exists():
            return set()
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return set()
            task_col = "true_target_task" if "true_target_task" in df.columns else "target_task"
            if "seed" in df.columns:
                return {f"{r[task_col]}__seed{int(r['seed'])}"
                        for _, r in df[[task_col, "seed"]].iterrows()}
            return set(df[task_col].unique().tolist())
        except Exception as e:
            print(f"  [Checkpoint] Read error: {e}")
            return set()

    def save_row(self, row: Dict):
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode="a", header=not self._header_written, index=False)
        self._header_written = True

    def load_all(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.csv_path)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Extract top paraphrases from Phase 1 (regenerate with same seeds)
# ─────────────────────────────────────────────────────────────────────────────

def extract_phase1_paraphrases(
    para_model, para_tok, device: str,
    target_tasks: Dict, inject_instruction: str, inject_data: str,
    detector,
    num_paraphrases: int, mw_rounds: int, mw_eta: float, data_num: int,
    top_k: int = TOP_K_PARAPHRASES,
    save_path: Path = None,
) -> List[str]:
    """
    Re-run Phase 1 attack with seeds 42,43,44 to recover the paraphrase pool
    and MW weights. Extract top-k paraphrases by final weight per (task, seed).

    This is deterministic given the same seeds — produces identical paraphrase
    pools and MW trajectories as the original Phase 1 run.

    Returns: deduplicated list of top paraphrases for augmentation (P0).
    """
    if save_path and save_path.exists():
        print(f"  [P0] Loading cached paraphrases from {save_path}")
        with open(save_path) as f:
            data = json.load(f)
        return data["paraphrases"]

    print(f"\n[P0] Extracting Phase 1 paraphrases (re-running with seeds {PHASE1_SEEDS})...")
    all_top_paraphrases = []

    for seed in PHASE1_SEEDS:
        set_seed(seed)
        for dataset in TARGET_DATASETS:
            task_name = DATASET_TO_TASK_NAME[dataset]
            print(f"  [P0] seed={seed} task={task_name}")

            paraphrases = generate_paraphrase_pool(
                para_model, para_tok, device,
                inject_instruction, inject_data,
                num_paraphrases=num_paraphrases,
            )
            strategies = build_strategy_pool(DATASENTINEL_SEPARATORS, paraphrases)

            mw_state = multiplicative_weights_learn(
                detector, strategies, target_tasks[dataset], data_num,
                T=mw_rounds, eta=mw_eta, verbose=False,
            )

            # Extract top-k paraphrase indices by MW weight
            top_indices = mw_state.top_k_strategies(k=top_k)
            for idx in top_indices:
                para_idx = strategies[idx].para_idx
                if para_idx < len(paraphrases):
                    all_top_paraphrases.append(paraphrases[para_idx])

    # Deduplicate while preserving order
    seen = set()
    unique_paraphrases = []
    for p in all_top_paraphrases:
        if p not in seen:
            seen.add(p)
            unique_paraphrases.append(p)

    print(f"\n[P0] Extracted {len(unique_paraphrases)} unique paraphrases for augmentation")

    if save_path:
        with open(save_path, "w") as f:
            json.dump({"paraphrases": unique_paraphrases,
                       "n_total": len(all_top_paraphrases),
                       "n_unique": len(unique_paraphrases)}, f, indent=2)
        print(f"  [P0] Saved to {save_path}")

    return unique_paraphrases


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Augmented retraining
# ─────────────────────────────────────────────────────────────────────────────

def build_X_c_from_paraphrases(
    D_t: List[Dict],
    D_e_sample: Dict,
    paraphrases: List[str],
    separators: List[str] = DATASENTINEL_SEPARATORS,
    max_samples: int = 500,
) -> List[Dict]:
    """
    Build contaminated training samples from semantic paraphrases.

    For each (paraphrase, separator) pair, create contaminated inputs
    of the form: x_t || separator || paraphrase
    These are added to X_c alongside the GCG-optimized separator samples.

    max_samples: cap to avoid overwhelming the training pool with paraphrases
    """
    X_c_sem = []
    y_e = D_e_sample['y']

    # Use subset of D_t to avoid memory issues
    sample_D_t = D_t[:min(len(D_t), max_samples // max(len(paraphrases) * len(separators), 1))]
    sample_D_t = sample_D_t if sample_D_t else D_t[:10]

    for para in paraphrases:
        for sep in separators:
            if sep == "__NEURALEXEC__":
                continue  # skip NeuralExec for simplicity in training
            for sample in sample_D_t:
                x_t = preprocess_data(sample['x'])
                if sep:
                    x_c = f"{x_t}{sep}{para}"
                else:
                    x_c = f"{x_t}{para}"
                X_c_sem.append({'x_c': x_c, 'target': y_e})

                if len(X_c_sem) >= max_samples:
                    return X_c_sem

    return X_c_sem


def retrain_with_augmentation(
    base_ckpt_path: str,
    D: List[Dict],
    augmentation_paraphrases: List[str],
    save_dir: str,
    detection_model_id: str,
    backend_model_id: str,
    r: int = R,
    device: str = 'cuda',
    retrain_lr: int = 0.0000025, 
    retrain_beta: int = 0.5,
) -> str:
    """
    Retrain DataSentinel from base_ckpt with semantic paraphrase augmentation.

    Modification of Algorithm 1: after building X_c from GCG separator (Step 2),
    append X_c_sem (built from augmentation_paraphrases) before running UpdateLLM.

    Returns path to final checkpoint.
    """
    from peft import PeftModel

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[Retrain] Loading detection model: {detection_model_id}")
    det_model, det_tok = load_detection_llm(detection_model_id)

    # Load base checkpoint LoRA weights (warm start).
    # Use PeftModel.from_pretrained so it reads adapter_config.json from the
    # checkpoint and loads with the EXACT rank the checkpoint was trained with
    # (r=32 for the published DataSentinel checkpoint), avoiding shape mismatches
    # that occur when attach_lora() uses our default LORA_R=64.
    print(f"[Retrain] Loading base checkpoint: {base_ckpt_path}")
    has_adapter = (
        os.path.exists(os.path.join(base_ckpt_path, "adapter_model.safetensors")) or
        os.path.exists(os.path.join(base_ckpt_path, "adapter_model.bin")) or
        os.path.exists(os.path.join(base_ckpt_path, "adapter_config.json"))
    )
    if has_adapter:
        from peft import PeftModel
        print(f"[Retrain] Using PeftModel.from_pretrained (reads rank from adapter_config.json)")
        det_model = PeftModel.from_pretrained(
            det_model, base_ckpt_path,
            is_trainable=True,   # keep adapter weights trainable for continued fine-tuning
        )
        print(f"[Retrain] LoRA loaded. Rank from checkpoint: "
              f"{det_model.peft_config['default'].r}")
    else:
        print(f"[Retrain] No LoRA adapter found at {base_ckpt_path} — attaching fresh LoRA")
        det_model = attach_lora(det_model, det_tok)

    print(f"\n[Retrain] Loading backend model: {backend_model_id}")
    back_model, back_tok = load_backend_llm(backend_model_id)

    # Initialize or load separator
    vocab_size = det_model.get_input_embeddings().weight.shape[0]
    z_ids = torch.randint(0, vocab_size, (SEP_LEN,), device=device)
    print(f"[Retrain] Initial separator: '{det_tok.decode(z_ids, skip_special_tokens=True)}'")

    print(f"[Retrain] Augmentation pool: {len(augmentation_paraphrases)} paraphrases")
    print(f"[Retrain] Running {r} training rounds...")

    for round_i in range(1, r + 1):
        print(f"\n{'='*60}")
        print(f"RETRAINING ROUND {round_i}/{r}")
        print(f"{'='*60}")

        # Step 1: Split D
        d_e_idx = random.randint(0, len(D) - 1)
        D_e = [D[d_e_idx]]
        D_t = [D[i] for i in range(len(D)) if i != d_e_idx]

        # Step 2: Inner max — GCG separator optimization
        print(f"\n[Round {round_i}] Step 2: Inner Max (OptCTD)")
        z_ids = opt_ctd(
            D_t, D_e, z_ids, det_model, det_tok, back_model, back_tok,
            n_in=N_IN, b_in=B_IN, device=device,
        )
        z_str = det_tok.decode(z_ids, skip_special_tokens=True)
        print(f"  Optimized separator: '{z_str}'")

        # Build X_c from GCG separator (standard)
        X_c_gcg = build_X_c(D_t, D_e[0], z_ids, det_tok, back_model, back_tok, device)

        # Build X_c_sem from semantic paraphrases (augmentation)
        print(f"\n[Round {round_i}] Building semantic augmentation X_c_sem...")
        X_c_sem = build_X_c_from_paraphrases(D_t, D_e[0], augmentation_paraphrases)
        print(f"  X_c_gcg: {len(X_c_gcg)} samples")
        print(f"  X_c_sem: {len(X_c_sem)} samples (paraphrase augmentation)")

        # Combined training pool
        X_c = X_c_gcg + X_c_sem
        print(f"  X_c total: {len(X_c)} samples")

        # Step 3: Outer min — UpdateLLM with combined pool
        print(f"\n[Round {round_i}] Step 3: Outer Min (UpdateLLM, augmented)")
        det_model = update_llm(
            det_model, det_tok, X_c, D_t,
            n_out=N_OUT, b_out=B_OUT,
            lr=retrain_lr, beta=retrain_beta,
            device=device,
        )

        # Save checkpoint after each round
        ckpt_dir = os.path.join(save_dir, f"checkpoint_round{round_i}")
        det_model.save_pretrained(ckpt_dir)
        det_tok.save_pretrained(ckpt_dir)
        print(f"\n[Round {round_i}] Checkpoint saved → {ckpt_dir}")

        # Save separator
        with open(os.path.join(save_dir, f"separator_round{round_i}.json"), "w") as f:
            json.dump({"round": round_i, "separator": z_str,
                       "token_ids": z_ids.tolist()}, f, indent=2)

    # Final save
    final_dir = os.path.join(save_dir, "final")
    det_model.save_pretrained(final_dir)
    det_tok.save_pretrained(final_dir)
    print(f"\n[Retrain] Final checkpoint → {final_dir}")
    return final_dir


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: Attack evaluation on retrained detector
# ─────────────────────────────────────────────────────────────────────────────

def attack_retrained_detector(
    retrained_ckpt_path: str,
    para_model, para_tok, device: str,
    target_tasks: Dict,
    inject_instruction: str, inject_data: str,
    seeds: List[int],
    num_paraphrases: int, mw_rounds: int, mw_eta: float, data_num: int,
    ckpt_manager: CheckpointManager,
    resume: bool = True,
) -> Dict[str, float]:
    """
    Run C4 attack on retrained detector with fresh seeds.
    Returns dict of {task_name: mean_fnr}.
    Also returns the discovered paraphrases for next-round augmentation.
    """
    print(f"\n[Attack] Loading retrained detector from {retrained_ckpt_path}")
    detector = load_detector(retrained_ckpt_path, device)

    done = ckpt_manager.get_completed_tasks() if resume else set()
    results = {}
    discovered_paraphrases = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n  === Attack seed={seed} ===")

        for dataset in TARGET_DATASETS:
            task_name = DATASET_TO_TASK_NAME[dataset]
            done_key = f"{task_name}__seed{seed}"

            if done_key in done:
                print(f"  o {task_name} seed={seed} (checkpoint)")
                continue

            t0 = time.time()
            print(f"\n  -> {task_name} (seed={seed})")

            paraphrases = generate_paraphrase_pool(
                para_model, para_tok, device,
                inject_instruction, inject_data,
                num_paraphrases=num_paraphrases,
            )
            strategies = build_strategy_pool(DATASENTINEL_SEPARATORS, paraphrases)

            mw_state = multiplicative_weights_learn(
                detector, strategies, target_tasks[dataset], data_num,
                T=mw_rounds, eta=mw_eta, verbose=False,
            )

            fnr, details = exploit_learned_strategy(
                detector, strategies, mw_state, target_tasks[dataset],
                data_num, mode="best",
            )

            # Collect top paraphrases from this run for next-round augmentation
            top_indices = mw_state.top_k_strategies(k=TOP_K_PARAPHRASES)
            for idx in top_indices:
                para_idx = strategies[idx].para_idx
                if para_idx < len(paraphrases):
                    discovered_paraphrases.append(paraphrases[para_idx])

            elapsed = time.time() - t0
            K = len(strategies)
            regret_bound = math.sqrt(math.log(K) / max(mw_rounds, 1))

            print(f"    FNR={fnr:.4f}  "
                  f"(K={K}, regret<={regret_bound:.4f}, {elapsed:.0f}s)")

            row = {
                "seed": seed,
                "true_target_task": task_name,
                "target_task": done_key,
                "fnr": fnr,
                "K": K,
                "T": mw_rounds,
                "eta": mw_eta,
                "regret_bound": regret_bound,
            }
            ckpt_manager.save_row(row)

            if task_name not in results:
                results[task_name] = []
            results[task_name].append(fnr)

    # Compute per-task means
    mean_results = {task: float(np.mean(fnrs)) for task, fnrs in results.items()}

    # Deduplicate discovered paraphrases
    seen = set()
    unique_discovered = []
    for p in discovered_paraphrases:
        if p not in seen:
            seen.add(p)
            unique_discovered.append(p)

    return mean_results, unique_discovered


# ─────────────────────────────────────────────────────────────────────────────
#  Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_phase2_summary(save_dir: Path):
    """Load all round results and print the adaptive dynamics table."""
    print(f"\n{'='*70}")
    print("PHASE 2: ADAPTIVE DEFENDER RESULTS")
    print("='*70")

    # Reference Round 0 (from Phase 1)
    ref_fnr = 0.9533
    ref_std = 0.090

    print(f"\n{'Round':<10} {'Mean FNR':>12} {'Std':>10} {'Notes'}")
    print("-" * 60)
    print(f"{'Round 0':<10} {ref_fnr:>12.4f} {'+-'+f'{ref_std:.3f}':>10}  "
          f"Published checkpoint (Phase 1)")

    for round_num in [1, 2]:
        ckpt_name = f"round{round_num}_attack"
        csv_path = save_dir / "checkpoints" / f"{ckpt_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                task_col = "true_target_task" if "true_target_task" in df.columns \
                    else "target_task"
                per_task = df.groupby(task_col)["fnr"].mean()
                overall_mean = float(per_task.mean())
                overall_std = float(df["fnr"].std()) if len(df) > 1 else 0.0
                seeds_used = ROUND1_SEEDS if round_num == 1 else ROUND2_SEEDS
                print(f"{'Round '+str(round_num):<10} {overall_mean:>12.4f} "
                      f"{'+-'+f'{overall_std:.3f}':>10}  "
                      f"Retrained+attacked (seeds {seeds_used})")
        else:
            print(f"{'Round '+str(round_num):<10} {'—':>12}  (not yet complete)")

    print("-" * 60)

    # LaTeX table
    print(f"\nLATEX TABLE (for §6.7 Adaptive Defender Evaluation):")
    print("-" * 60)
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Adaptive defender evaluation. After each attack round, "
          "DataSentinel is retrained with the discovered paraphrases added to "
          "its training pool. The attack then uses fresh paraphrases (new seeds) "
          "against the retrained detector.}")
    print("\\label{tab:adaptive_defender}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Round} & \\textbf{Detector} & \\textbf{FNR} \\\\")
    print("\\midrule")
    print(f"Round 0 & Published checkpoint & ${ref_fnr:.4f} \\\\pm {ref_std:.3f}$ \\\\")

    for round_num in [1, 2]:
        csv_path = save_dir / "checkpoints" / f"round{round_num}_attack.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                overall_mean = float(df["fnr"].mean())
                overall_std = float(df["fnr"].std()) if len(df) > 1 else 0.0
                desc = f"Retrained (Round {round_num-1} paraphrases)"
                print(f"Round {round_num} & {desc} & "
                      f"${overall_mean:.4f} \\\\pm {overall_std:.3f}$ \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Adaptive Defender Experiment")
    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="Path to published DataSentinel checkpoint-5000")
    parser.add_argument("--detection_model", type=str,
                        default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--backend_model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--paraphrase_model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--retrain_lr", type=float, default=0.0000025,
                        help="Learning rate for warm-start retraining")
    parser.add_argument("--retrain_beta", type=float, default=0.5,
                        help="Beta for clean loss in UpdateLLM")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm for UpdateLLM")
    parser.add_argument("--repo_dir", type=str, default=".")
    parser.add_argument("--data_num", type=int, default=100)
    parser.add_argument("--num_paraphrases", type=int, default=20)
    parser.add_argument("--mw_rounds", type=int, default=100)
    parser.add_argument("--mw_eta", type=float, default=0.3)
    parser.add_argument("--training_rounds", type=int, default=R,
                        help="DataSentinel retraining rounds (default: 3, paper value)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/phase2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_round", type=int, default=0,
                        help="Skip to this round (1 or 2) if retraining already done")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2: ADAPTIVE DEFENDER EXPERIMENT")
    print("=" * 70)
    print(f"  Base checkpoint   : {args.base_ckpt}")
    print(f"  Detection model   : {args.detection_model}")
    print(f"  Backend model     : {args.backend_model}")
    print(f"  Paraphrase model  : {args.paraphrase_model}")
    print(f"  Training rounds   : {args.training_rounds}")
    print(f"  MW: T={args.mw_rounds}, eta={args.mw_eta}, |Phi|={args.num_paraphrases}")
    print(f"  Attack seeds R1   : {ROUND1_SEEDS}")
    print(f"  Attack seeds R2   : {ROUND2_SEEDS}")
    print(f"  Save dir          : {save_dir}")
    print(f"  Resume            : {args.resume}")
    print("=" * 70)

    # ── Load tasks and models ─────────────────────────────────────────────────
    setup_repo(args.repo_dir)
    print("\n[Setup] Loading tasks...")
    target_tasks, inject_tasks = load_all_tasks(args.repo_dir, args.data_num)

    inject_task = inject_tasks[INJECT_DATASET]
    inject_instr = inject_task.get_instruction()
    inject_data_0, _ = inject_task[0]

    print("\n[Setup] Loading paraphrase model...")
    para_model, para_tok = load_paraphrase_model(args.paraphrase_model, args.device)

    # ── Step P0: Extract Phase 1 paraphrases ──────────────────────────────────
    p0_path = save_dir / "p0_paraphrases.json"

    if args.skip_round < 1:
        print("\n" + "=" * 70)
        print("STEP P0: Extracting Phase 1 paraphrases (re-running with seeds 42,43,44)")
        print("=" * 70)

        # Need detector for MW re-run
        print("[P0] Loading published detector for paraphrase extraction...")
        detector_r0 = load_detector(args.base_ckpt, args.device)

        P0 = extract_phase1_paraphrases(
            para_model, para_tok, args.device,
            target_tasks, inject_instr, inject_data_0,
            detector_r0,
            num_paraphrases=args.num_paraphrases,
            mw_rounds=args.mw_rounds,
            mw_eta=args.mw_eta,
            data_num=args.data_num,
            top_k=TOP_K_PARAPHRASES,
            save_path=p0_path,
        )
        del detector_r0
        torch.cuda.empty_cache()
    else:
        print(f"\n[P0] Loading cached P0 from {p0_path}")
        with open(p0_path) as f:
            P0 = json.load(f)["paraphrases"]
        print(f"  Loaded {len(P0)} paraphrases")

    # ── ROUND 1: Retrain with P0 → Attack with seeds 45,46,47 ────────────────
    if args.skip_round < 1:
        print("\n" + "=" * 70)
        print("ROUND 1: Retraining with P0 augmentation")
        print("=" * 70)

        round1_train_dir = str(save_dir / "round1_retrained")
        D = load_gigaword_samples(args.repo_dir)

        # Check if retraining already done
        round1_final = os.path.join(round1_train_dir, "final")
        if args.resume and os.path.exists(round1_final):
            print(f"  [Round 1] Retraining already complete — loading {round1_final}")
        else:
            random.seed(42)
            retrain_with_augmentation(
                base_ckpt_path=args.base_ckpt,
                D=D,
                augmentation_paraphrases=P0,
                save_dir=round1_train_dir,
                detection_model_id=args.detection_model,
                backend_model_id=args.backend_model,
                r=args.training_rounds,
                device=args.device,
                retrain_lr=args.retrain_lr,
                retrain_beta=args.retrain_beta,
            )

        torch.cuda.empty_cache()

    # Attack Round 1 detector with fresh seeds
    if args.skip_round < 2:
        print("\n" + "=" * 70)
        print("ROUND 1: Attacking retrained detector (seeds 45,46,47)")
        print("=" * 70)

        round1_final = str(save_dir / "round1_retrained" / "final")
        r1_ckpt = CheckpointManager(save_dir, "round1_attack")

        r1_results, P1 = attack_retrained_detector(
            retrained_ckpt_path=round1_final,
            para_model=para_model, para_tok=para_tok, device=args.device,
            target_tasks=target_tasks,
            inject_instruction=inject_instr, inject_data=inject_data_0,
            seeds=ROUND1_SEEDS,
            num_paraphrases=args.num_paraphrases,
            mw_rounds=args.mw_rounds, mw_eta=args.mw_eta,
            data_num=args.data_num,
            ckpt_manager=r1_ckpt,
            resume=args.resume,
        )

        r1_mean = float(np.mean(list(r1_results.values()))) if r1_results else float("nan")
        print(f"\n  Round 1 mean FNR: {r1_mean:.4f}")
        print(f"  Round 1 discovered {len(P1)} new paraphrases for P1")

        # Save P1
        p1_path = save_dir / "p1_paraphrases.json"
        with open(p1_path, "w") as f:
            json.dump({"paraphrases": P1, "n": len(P1)}, f, indent=2)

        torch.cuda.empty_cache()

    # ── ROUND 2: Retrain with P0+P1 → Attack with seeds 48,49,50 ─────────────
    print("\n" + "=" * 70)
    print("ROUND 2: Retraining with P0+P1 augmentation")
    print("=" * 70)

    # Load P1 if coming from skip
    p1_path = save_dir / "p1_paraphrases.json"
    if p1_path.exists():
        with open(p1_path) as f:
            P1 = json.load(f)["paraphrases"]
    else:
        print("  WARNING: P1 not found — using P0 only for Round 2 retraining")
        P1 = []

    P0_plus_P1 = list(dict.fromkeys(P0 + P1))  # deduplicated, order preserved
    print(f"  Augmentation pool: P0({len(P0)}) + P1({len(P1)}) = {len(P0_plus_P1)} unique")

    round2_train_dir = str(save_dir / "round2_retrained")
    round2_final = os.path.join(round2_train_dir, "final")
    round1_final_ckpt = str(save_dir / "round1_retrained" / "final")

    if args.resume and os.path.exists(round2_final):
        print(f"  [Round 2] Retraining already complete — loading {round2_final}")
    else:
        D = load_gigaword_samples(args.repo_dir)
        random.seed(43)
        retrain_with_augmentation(
            base_ckpt_path=round1_final_ckpt,  # warm start from Round 1
            D=D,
            augmentation_paraphrases=P0_plus_P1,
            save_dir=round2_train_dir,
            detection_model_id=args.detection_model,
            backend_model_id=args.backend_model,
            r=args.training_rounds,
            device=args.device,
            retrain_lr=args.retrain_lr,
            retrain_beta=args.retrain_beta,
        )

    torch.cuda.empty_cache()

    # Attack Round 2 detector with fresh seeds
    print("\n" + "=" * 70)
    print("ROUND 2: Attacking retrained detector (seeds 48,49,50)")
    print("=" * 70)

    r2_ckpt = CheckpointManager(save_dir, "round2_attack")

    r2_results, P2 = attack_retrained_detector(
        retrained_ckpt_path=round2_final,
        para_model=para_model, para_tok=para_tok, device=args.device,
        target_tasks=target_tasks,
        inject_instruction=inject_instr, inject_data=inject_data_0,
        seeds=ROUND2_SEEDS,
        num_paraphrases=args.num_paraphrases,
        mw_rounds=args.mw_rounds, mw_eta=args.mw_eta,
        data_num=args.data_num,
        ckpt_manager=r2_ckpt,
        resume=args.resume,
    )

    r2_mean = float(np.mean(list(r2_results.values()))) if r2_results else float("nan")
    print(f"\n  Round 2 mean FNR: {r2_mean:.4f}")

    # Save P2
    p2_path = save_dir / "p2_paraphrases.json"
    with open(p2_path, "w") as f:
        json.dump({"paraphrases": P2, "n": len(P2)}, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_phase2_summary(save_dir)
    print(f"\nAll results saved to: {save_dir}")
    print("Done!")


if __name__ == "__main__":
    main()