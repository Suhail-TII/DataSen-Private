#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_phase2.sh — Submit Phase 2 Adaptive Defender jobs
#
# USAGE
#   bash submit_phase2.sh                        # full run (P0 + 2 rounds)
#   bash submit_phase2.sh --skip-round 1         # skip to Round 2 (retraining done)
#   bash submit_phase2.sh --dry-run              # preview
#
# Phase 2 is a single sequential job (not an array) since rounds depend on
# each other. Two chained segments provide 48h total coverage.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_CKPT="/dpc/kuex0037/Open-Prompt-Injection/DataSentinel_Models/detector_large/checkpoint-5000"
DETECTION_MODEL="mistralai/Mistral-7B-v0.1"
BACKEND_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PARAPHRASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
REPO_DIR="/dpc/kuex0037/Open-Prompt-Injection"
DATA_NUM=100
NUM_PARAPHRASES=20
MW_ROUNDS=100
MW_ETA=0.3
TRAINING_ROUNDS=3
SKIP_ROUND=0
RETRAIN_LR=0.0000025
RETRAIN_BETA=0.5
GRAD_CLIP=1.0
DRY_RUN=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-ckpt)        BASE_CKPT="$2";        shift 2 ;;
        --detection-model)  DETECTION_MODEL="$2";  shift 2 ;;
        --backend-model)    BACKEND_MODEL="$2";    shift 2 ;;
        --paraphrase-model) PARAPHRASE_MODEL="$2"; shift 2 ;;
        --repo-dir)         REPO_DIR="$2";         shift 2 ;;
        --data-num)         DATA_NUM="$2";         shift 2 ;;
        --num-paraphrases)  NUM_PARAPHRASES="$2";  shift 2 ;;
        --mw-rounds)        MW_ROUNDS="$2";        shift 2 ;;
        --mw-eta)           MW_ETA="$2";           shift 2 ;;
        --training-rounds)  TRAINING_ROUNDS="$2";  shift 2 ;;
        --retrain-lr)       RETRAIN_LR="$2";       shift 2 ;;
        --retrain-beta)     RETRAIN_BETA="$2";     shift 2 ;;
        --grad-clip)        GRAD_CLIP="$2";        shift 2 ;;
        --skip-round)       SKIP_ROUND="$2";       shift 2 ;;
        --dry-run)          DRY_RUN=1;             shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Output directory ──────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/dpc/kuex0037/Open-Prompt-Injection/results/phase2/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR" logs

echo "======================================================================"
echo "Phase 2: Adaptive Defender — Job Submission"
echo "======================================================================"
echo "  Base checkpoint   : $BASE_CKPT"
echo "  Detection model   : $DETECTION_MODEL"
echo "  Backend model     : $BACKEND_MODEL"
echo "  Paraphrase model  : $PARAPHRASE_MODEL"
echo "  Repo dir          : $REPO_DIR"
echo "  data_num          : $DATA_NUM"
echo "  num_paraphrases   : $NUM_PARAPHRASES"
echo "  mw_rounds         : $MW_ROUNDS"
echo "  mw_eta            : $MW_ETA"
echo "  training_rounds   : $TRAINING_ROUNDS"
echo "  skip_round        : $SKIP_ROUND"
echo "  Output dir        : $RUN_DIR"
echo "  Dry run           : $DRY_RUN"
echo "======================================================================"

export RUN_DIR
export BASE_CKPT
export DETECTION_MODEL
export BACKEND_MODEL
export PARAPHRASE_MODEL
export REPO_DIR
export DATA_NUM
export NUM_PARAPHRASES
export MW_ROUNDS
export MW_ETA
export TRAINING_ROUNDS
export SKIP_ROUND
export RETRAIN_LR
export RETRAIN_BETA
export GRAD_CLIP

SBATCH_CMD="sbatch --export=ALL slurm/slurm_phase2.sh"

echo ""
echo "Submitting Segment 1/2 (24h walltime)..."
echo "  $SBATCH_CMD"

if [[ $DRY_RUN -eq 0 ]]; then
    OUT1=$($SBATCH_CMD)
    JID1=$(echo "$OUT1" | grep -oP '\d+' | head -1)
    echo "  -> Segment 1 Job ID: $JID1"

    OUT2=$(sbatch --export=ALL \
                  --dependency=afterany:$JID1 slurm/slurm_phase2.sh)
    JID2=$(echo "$OUT2" | grep -oP '\d+' | head -1)
    echo "  -> Segment 2 Job ID: $JID2 (after $JID1)"
else
    echo "  -> [DRY RUN] Would submit 2 chained segments"
fi

echo ""
echo "======================================================================"
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/phase2_*.out"
echo ""
echo "Results directory:"
echo "  $RUN_DIR"
echo ""
echo "View results when done:"
echo "  python phase2_adaptive_defender.py --save_dir $RUN_DIR --resume"
echo "  (prints summary table without re-running)"
echo ""
echo "Timeline:"
echo "  P0 extraction   ~3-4h"
echo "  Round 1 retrain ~3-4h"
echo "  Round 1 attack  ~2-3h"
echo "  Round 2 retrain ~3-4h"
echo "  Round 2 attack  ~2-3h"
echo "  Total           ~13-18h  (well within 24h Segment 1)"
echo "======================================================================"