"""
phase1_cross_detector.py — Phase 1: Cross-Detector Evaluation
==============================================================

Runs the GT-Semantic C4 attack (MW + Φ_LLM) against every detector from
table3_ds.py, measuring:

  • FPR  — fraction of CLEAN inputs flagged as contaminated
  • FNR  — fraction of CONTAMINATED inputs the detector MISSES (evasion rate)

One detector per invocation. The SLURM array maps task indices → detector names.

Supported detectors (--detector):
  kad            Base Mistral-7B + known-answer prompt
  nllmd          Base Mistral-7B + yes/no injection question
  promptguard    Meta's PromptGuard-86M (DeBERTa classifier)
  evd            CLIP + 2-layer NN (trains or loads from checkpoint)
  ssftd          SSFTD LoRA checkpoint (requires --ssftd_path)
  ssftd_g        SSFTD-G LoRA checkpoint (requires --ssftd_g_path)

DataSentinel is NOT re-run here — use published results (FNR=0.9533±0.090).

Attack config (fixed, matching main results):
  T=100, η=0.3, K=10 paraphrases × 7 separators, seeds 42,43,44
  Inject task: SST2, Target tasks: all 6 others

Usage:
  python phase1_cross_detector.py \\
      --detector kad \\
      --paraphrase_model meta-llama/Meta-Llama-3.1-8B-Instruct \\
      --repo_dir . --data_num 100 \\
      --num_paraphrases 10 --mw_rounds 100 --mw_eta 0.3 \\
      --seeds 42,43,44 \\
      --save_dir results/phase1 \\
      --resume

  # With SSFTD checkpoint:
  python phase1_cross_detector.py --detector ssftd \\
      --ssftd_path results/ssftd_checkpoint/checkpoint-XXXX \\
      ...
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch


# ── Import building blocks ────────────────────────────────────────────────────
# gt_semantic_attack.py must be in the same directory
from gt_semantic_attack import (
    DATASETS, DATASET_TO_TASK_NAME, INJECT_DATASET,
    DATASENTINEL_SEPARATORS,
    Strategy, MWState,
    set_seed,
    setup_repo, load_all_tasks, load_paraphrase_model,
    generate_paraphrase_pool, build_strategy_pool, apply_strategy,
    multiplicative_weights_learn, exploit_learned_strategy,
)

# table3_ds.py must also be in the same directory
from table3_ds import (
    KADDetector, NLLMDDetector, PromptGuardDetector,
    EVDDetector, SSFTDDetector, DataSentinelDetectorWrapper,
    load_base_mistral,
    DATASET_TO_TASK,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Detectors that share the base Mistral-7B model
MISTRAL_DETECTORS = {"kad", "nllmd", "ssftd", "ssftd_g"}

# Target tasks (all except the inject task SST2)
TARGET_DATASETS = [d for d in DATASETS if d != INJECT_DATASET]

DETECTOR_DISPLAY = {
    "kad":         "KAD",
    "nllmd":       "NLLMD",
    "promptguard": "PromptGuard",
    "evd":         "EVD",
    "ssftd":       "SSFTD",
    "ssftd_g":     "SSFTD-G",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint manager (mirrors the pattern from gt_semantic_ablation.py)
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, output_dir: Path, detector: str, metric: str):
        self.ckpt_dir = Path(output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.ckpt_dir / f"{detector}_{metric}.csv"
        self._header_written = self.csv_path.exists()

    def get_completed_tasks(self) -> Set[str]:
        if not self.csv_path.exists():
            return set()
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return set()
            if "true_target_task" in df.columns and "seed" in df.columns:
                return {f"{r['true_target_task']}__seed{int(r['seed'])}"
                        for _, r in df[["true_target_task", "seed"]].iterrows()}
            if "target_task" in df.columns and "seed" in df.columns:
                keys = set()
                for _, r in df[["target_task", "seed"]].iterrows():
                    tt = str(r["target_task"])
                    keys.add(tt if "__seed" in tt else f"{tt}__seed{int(r['seed'])}")
                return keys
            return set(df.get("target_task", pd.Series()).unique().tolist())
        except Exception as e:
            print(f"  [Checkpoint] Read error {self.csv_path}: {e}")
            return set()

    def save_row(self, row: Dict):
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode="a", header=not self._header_written, index=False)
        self._header_written = True


# ─────────────────────────────────────────────────────────────────────────────
#  Detector factory
# ─────────────────────────────────────────────────────────────────────────────

def build_detector(detector_name: str, args, mistral_config,
                   target_tasks: Dict, inject_tasks: Dict, device: str):
    """
    Instantiate the requested detector. Returns a detector object with
    a .detect(text) → int method.
    """
    if detector_name in MISTRAL_DETECTORS:
        model_name = mistral_config["model_info"]["name"]
        base_model, tokenizer = load_base_mistral(model_name, device)
    else:
        base_model = tokenizer = None

    if detector_name == "kad":
        det = KADDetector(base_model, tokenizer)

    elif detector_name == "nllmd":
        det = NLLMDDetector(base_model, tokenizer)

    elif detector_name == "promptguard":
        det = PromptGuardDetector()

    elif detector_name == "evd":
        evd = EVDDetector(args.save_dir)
        if os.path.exists(evd.ckpt):
            evd.load()
        else:
            print("  [EVD] No checkpoint found — training now...")
            evd.train(target_tasks["gigaword"], inject_tasks["gigaword"])
        det = evd

    elif detector_name == "ssftd":
        if not args.ssftd_path:
            raise ValueError("--ssftd_path required for --detector ssftd")
        det = SSFTDDetector(args.ssftd_path, mistral_config)

    elif detector_name == "ssftd_g":
        if not args.ssftd_g_path:
            raise ValueError("--ssftd_g_path required for --detector ssftd_g")
        # SSFTDDetector handles both variants — same class, different checkpoint
        det = SSFTDDetector(args.ssftd_g_path, mistral_config)

    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    print(f"  [{DETECTOR_DISPLAY[detector_name]}] Ready.")
    return det


# ─────────────────────────────────────────────────────────────────────────────
#  FPR measurement (clean inputs)
# ─────────────────────────────────────────────────────────────────────────────

def measure_fpr(detector, target_tasks: Dict, data_num: int) -> Dict[str, float]:
    """
    Measure FPR on CLEAN inputs across all 7 tasks (including SST2).
    FPR = fraction of clean samples flagged as contaminated.
    """
    fpr_results = {}
    for dataset in DATASETS:
        task = target_tasks[dataset]
        task_name = DATASET_TO_TASK_NAME[dataset]
        n = min(len(task), data_num)
        flags = 0
        for i in range(n):
            data_prompt, _ = task[i]
            pred = detector.detect(data_prompt)
            if pred == 1:
                flags += 1
        fpr = flags / max(n, 1)
        fpr_results[task_name] = fpr
        print(f"    FPR {task_name:<35}: {fpr:.4f} ({flags}/{n})")
    return fpr_results


# ─────────────────────────────────────────────────────────────────────────────
#  GT-Semantic C4 attack (FNR measurement)
# ─────────────────────────────────────────────────────────────────────────────

def run_c4_attack_on_detector(
    detector,
    para_model, para_tok, device: str,
    target_task, target_dataset: str,
    inject_instruction: str, inject_data: str,
    num_paraphrases: int, mw_rounds: int, mw_eta: float,
    data_num: int,
) -> Dict:
    """
    Run the full C4 attack (MW + Φ_LLM) against a given detector on one task.
    Returns a result dict with FNR and supporting statistics.
    """
    task_name = DATASET_TO_TASK_NAME[target_dataset]
    print(f"    [{task_name}] Generating paraphrase pool...")

    paraphrases = generate_paraphrase_pool(
        para_model, para_tok, device,
        inject_instruction, inject_data,
        num_paraphrases=num_paraphrases,
    )
    strategies = build_strategy_pool(DATASENTINEL_SEPARATORS, paraphrases)
    K = len(strategies)

    print(f"    [{task_name}] MW learning (T={mw_rounds}, η={mw_eta}, K={K})...")
    mw_state = multiplicative_weights_learn(
        detector, strategies, target_task, data_num,
        T=mw_rounds, eta=mw_eta, verbose=False,
    )

    fnr, details = exploit_learned_strategy(
        detector, strategies, mw_state, target_task, data_num, mode="best",
    )

    regret_bound = math.sqrt(math.log(K) / max(mw_rounds, 1))

    return {
        "fnr": fnr,
        "K": K,
        "T": mw_rounds,
        "eta": mw_eta,
        "regret_bound": regret_bound,
        "best_strategy_idx": details["best_strategy_idx"],
        "queries_mw": mw_rounds,
        "queries_eval": details["total"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Cross-Detector GT-Semantic Attack")
    parser.add_argument("--detector", type=str, required=True,
                        choices=["kad", "nllmd", "promptguard", "evd", "ssftd", "ssftd_g"],
                        help="Which detector to evaluate")
    parser.add_argument("--paraphrase_model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--repo_dir", type=str, default=".")
    parser.add_argument("--data_num", type=int, default=100)
    parser.add_argument("--num_paraphrases", type=int, default=10,
                        help="|Φ_LLM| — set to 10 to match K≈70 (7 seps × 10 paras)")
    parser.add_argument("--mw_rounds", type=int, default=100)
    parser.add_argument("--mw_eta", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/phase1")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--resume", action="store_true")
    # SSFTD/SSFTD-G checkpoint paths (optional)
    parser.add_argument("--ssftd_path", type=str, default=None)
    parser.add_argument("--ssftd_g_path", type=str, default=None)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    det_name = args.detector
    det_label = DETECTOR_DISPLAY[det_name]

    print("=" * 70)
    print(f"PHASE 1: CROSS-DETECTOR EVALUATION — {det_label}")
    print("=" * 70)
    print(f"  Detector      : {det_label}")
    print(f"  Attack        : GT-Semantic C4 (MW + Φ_LLM)")
    print(f"  T={args.mw_rounds}, η={args.mw_eta}, |Φ_LLM|={args.num_paraphrases}")
    print(f"  Seeds         : {seeds}")
    print(f"  data_num      : {args.data_num}")
    print(f"  Save dir      : {save_dir}")
    print("=" * 70)

    # ── Repo and tasks ────────────────────────────────────────────────────────
    setup_repo(args.repo_dir)
    import OpenPromptInjection as PI
    from OpenPromptInjection.utils import open_config

    print("\n[1/4] Loading tasks...")
    target_tasks, inject_tasks = load_all_tasks(args.repo_dir, args.data_num)

    # ── Mistral config (needed for some detectors) ────────────────────────────
    mistral_config = open_config("./configs/model_configs/mistral_config.json")
    mistral_config["params"]["device"] = args.device

    # ── Load detector ─────────────────────────────────────────────────────────
    print(f"\n[2/4] Loading detector: {det_label}...")
    detector = build_detector(det_name, args, mistral_config,
                               target_tasks, inject_tasks, args.device)

    # ── FPR on clean inputs (run once, deterministic) ─────────────────────────
    print(f"\n[3/4] Measuring FPR on clean inputs...")
    fpr_ckpt = CheckpointManager(save_dir, det_name, "fpr")
    fpr_done = fpr_ckpt.get_completed_tasks() if args.resume else set()

    if f"all__seed0" not in fpr_done:
        set_seed(seeds[0])  # FPR is deterministic; one seed suffices
        fpr_results = measure_fpr(detector, target_tasks, args.data_num)
        fpr_avg = float(np.mean(list(fpr_results.values())))
        print(f"  Average FPR: {fpr_avg:.4f}")

        fpr_row = {
            "detector": det_name,
            "seed": 0,
            "true_target_task": "all",
            "target_task": "all__seed0",
            "fpr_avg": fpr_avg,
            **{f"fpr_{k.replace(' ', '_').replace('.', '')}": v
               for k, v in fpr_results.items()},
        }
        fpr_ckpt.save_row(fpr_row)

        # Save human-readable FPR summary
        with open(save_dir / f"{det_name}_fpr.json", "w") as f:
            json.dump({"detector": det_label, "fpr": fpr_results, "avg": fpr_avg}, f, indent=2)
    else:
        print(f"  FPR already measured (resuming) — loading from checkpoint.")
        df_fpr = pd.read_csv(fpr_ckpt.csv_path)
        fpr_avg = float(df_fpr["fpr_avg"].iloc[0])
        print(f"  Average FPR (cached): {fpr_avg:.4f}")

    # ── Load paraphrase model ─────────────────────────────────────────────────
    print(f"\n[4/4] Loading paraphrase model & running GT-Semantic attack...")
    para_model, para_tok = load_paraphrase_model(args.paraphrase_model, args.device)

    inject_task = inject_tasks[INJECT_DATASET]
    inject_instr = inject_task.get_instruction()
    inject_data_0, _ = inject_task[0]

    # Checkpoint for FNR results
    fnr_ckpt = CheckpointManager(save_dir, det_name, "fnr")
    fnr_done = fnr_ckpt.get_completed_tasks() if args.resume else set()

    all_fnrs = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n  === Seed {seed} ===")

        for dataset in TARGET_DATASETS:
            task_name = DATASET_TO_TASK_NAME[dataset]
            done_key = f"{task_name}__seed{seed}"

            if done_key in fnr_done:
                print(f"  ⊙ {task_name} seed={seed} (skipped — checkpoint)")
                continue

            t0 = time.time()
            print(f"\n  → {task_name} (seed={seed})")

            try:
                result = run_c4_attack_on_detector(
                    detector=detector,
                    para_model=para_model, para_tok=para_tok, device=args.device,
                    target_task=target_tasks[dataset],
                    target_dataset=dataset,
                    inject_instruction=inject_instr,
                    inject_data=inject_data_0,
                    num_paraphrases=args.num_paraphrases,
                    mw_rounds=args.mw_rounds,
                    mw_eta=args.mw_eta,
                    data_num=args.data_num,
                )

                elapsed = time.time() - t0
                fnr = result["fnr"]
                all_fnrs.append(fnr)

                print(f"    ✓ FNR={fnr:.4f}  "
                      f"(K={result['K']}, regret≤{result['regret_bound']:.4f}, {elapsed:.0f}s)")

                row = {
                    "detector": det_name,
                    "seed": seed,
                    "true_target_task": task_name,
                    "target_task": done_key,
                    **result,
                }
                fnr_ckpt.save_row(row)

            except Exception as e:
                print(f"    ✗ FAILED: {e}")
                import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PHASE 1 SUMMARY — {det_label}")
    print(f"{'='*70}")

    # Load all FNR results from checkpoint for final summary
    df_fnr = pd.read_csv(fnr_ckpt.csv_path) if fnr_ckpt.csv_path.exists() else pd.DataFrame()

    if not df_fnr.empty and "true_target_task" in df_fnr.columns:
        print(f"\n  Per-task FNR (mean ± std across {len(seeds)} seeds):")
        print(f"  {'Task':<35} {'Mean':>8} {'Std':>8} {'#Seeds':>8}")
        print("  " + "-" * 62)
        per_task_means = []
        task_col = "true_target_task"
        for task_name, grp in df_fnr.groupby(task_col):
            m = float(grp["fnr"].mean())
            s = float(grp["fnr"].std()) if len(grp) > 1 else 0.0
            per_task_means.append(m)
            std_str = f"±{s:.4f}" if s > 0 else "—"
            print(f"  {task_name:<35} {m:>8.4f} {std_str:>8} {len(grp):>8}")
        overall_mean = float(np.mean(per_task_means))
        overall_std = float(df_fnr["fnr"].std()) if len(df_fnr) > 1 else 0.0
        print("  " + "-" * 62)
        print(f"  {'AVERAGE':<35} {overall_mean:>8.4f} ±{overall_std:.4f}")
        print(f"\n  FPR (clean inputs): {fpr_avg:.4f}")
        print(f"\n  Comparison to DataSentinel (reference):")
        print(f"    DataSentinel FNR: 0.9533 ± 0.090")
        print(f"    {det_label:<15} FNR: {overall_mean:.4f} ± {overall_std:.4f}")
        print(f"    DataSentinel FPR: 0.0000")
        print(f"    {det_label:<15} FPR: {fpr_avg:.4f}")

        # Save final summary
        summary = {
            "detector": det_label,
            "config": {
                "num_paraphrases": args.num_paraphrases,
                "mw_rounds": args.mw_rounds,
                "mw_eta": args.mw_eta,
                "seeds": seeds,
                "data_num": args.data_num,
            },
            "fnr_mean": overall_mean,
            "fnr_std": overall_std,
            "fpr_avg": fpr_avg,
        }
        with open(save_dir / f"{det_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved → {save_dir}/{det_name}_summary.json")

    print(f"{'='*70}")
    print("Done!")


if __name__ == "__main__":
    main()