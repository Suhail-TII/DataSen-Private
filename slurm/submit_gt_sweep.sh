#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_gt_sweep.sh — Submits GT-Semantic Hyperparameter Sweep jobs
#
# USAGE
#   bash submit_gt_sweep.sh                      # all 3 sweeps (T, η, K)
#   bash submit_gt_sweep.sh --sweeps T           # only T sweep
#   bash submit_gt_sweep.sh --sweeps T,eta       # T and η sweeps
#   bash submit_gt_sweep.sh --dry-run            # preview only
#
# Each sweep runs as a separate array task on its own GPU node, in parallel.
# Jobs use --resume so they can be killed and restarted without losing progress.
#
# Sweep T   (5 values × 6 tasks = 30 runs):  ~3-4 hours
# Sweep η   (5 values × 6 tasks = 30 runs):  ~3-4 hours
# Sweep K   (5 values × 6 tasks = 30 runs):  ~4-5 hours (larger pools at K=50)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SWEEPS="T,eta,K"
CKPT_DETECTOR="/dpc/kuex0037/Open-Prompt-Injection/DataSentinel_Models/detector_large/checkpoint-5000"
PARAPHRASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
REPO_DIR="/dpc/kuex0037/Open-Prompt-Injection"
DATA_NUM=100
T_VALUES="25,50,100,200,400"
ETA_VALUES="0.1,0.2,0.3,0.5,1.0"
K_VALUES="5,10,20,30,50"
T_FIXED=100
ETA_FIXED=0.3
NUM_PARA_FIXED=20
SEEDS="42,43,44"
DRY_RUN=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --sweeps)            SWEEPS="$2";            shift 2 ;;
        --ckpt-detector)     CKPT_DETECTOR="$2";     shift 2 ;;
        --paraphrase-model)  PARAPHRASE_MODEL="$2";  shift 2 ;;
        --repo-dir)          REPO_DIR="$2";          shift 2 ;;
        --data-num)          DATA_NUM="$2";          shift 2 ;;
        --T-values)          T_VALUES="$2";          shift 2 ;;
        --eta-values)        ETA_VALUES="$2";        shift 2 ;;
        --K-values)          K_VALUES="$2";          shift 2 ;;
        --T-fixed)           T_FIXED="$2";           shift 2 ;;
        --eta-fixed)         ETA_FIXED="$2";         shift 2 ;;
        --num-para-fixed)    NUM_PARA_FIXED="$2";    shift 2 ;;
        --seeds)             SEEDS="$2";             shift 2 ;;
        --dry-run)           DRY_RUN=1;              shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Setup output directory ────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/dpc/kuex0037/Open-Prompt-Injection/results/gt_sweep/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR" logs

echo "======================================================================"
echo "GT-Semantic Sweep — Job Submission"
echo "======================================================================"
echo "  Sweeps            : $SWEEPS"
echo "  T values          : $T_VALUES"
echo "  η values          : $ETA_VALUES"
echo "  K values          : $K_VALUES"
echo "  Fixed T / η / |Φ| : $T_FIXED / $ETA_FIXED / $NUM_PARA_FIXED"
echo "  Seeds             : $SEEDS"
echo "  Checkpoint        : $CKPT_DETECTOR"
echo "  Paraphrase model  : $PARAPHRASE_MODEL"
echo "  Output dir        : $RUN_DIR"
echo "  Dry run           : $DRY_RUN"
echo "======================================================================"

# ── Convert sweeps list to array task indices ────────────────────────────────
# T → 0, eta → 1, K → 2
ARRAY_INDICES=""
IFS=',' read -ra SWEEP_LIST <<< "$SWEEPS"
for S in "${SWEEP_LIST[@]}"; do
    S=$(echo $S | tr -d ' ')
    case $S in
        T)   IDX=0 ;;
        eta) IDX=1 ;;
        K)   IDX=2 ;;
        *)   echo "Unknown sweep: $S"; exit 1 ;;
    esac
    if [[ -z "$ARRAY_INDICES" ]]; then
        ARRAY_INDICES="$IDX"
    else
        ARRAY_INDICES="$ARRAY_INDICES,$IDX"
    fi
done

echo "  Array indices     : $ARRAY_INDICES"

# ── Build export string ──────────────────────────────────────────────────────
EXPORT="RUN_DIR=$RUN_DIR"
EXPORT="$EXPORT,CKPT_DETECTOR=$CKPT_DETECTOR"
EXPORT="$EXPORT,PARAPHRASE_MODEL=$PARAPHRASE_MODEL"
EXPORT="$EXPORT,REPO_DIR=$REPO_DIR"
EXPORT="$EXPORT,DATA_NUM=$DATA_NUM"
EXPORT="$EXPORT,T_VALUES=$T_VALUES"
EXPORT="$EXPORT,ETA_VALUES=$ETA_VALUES"
EXPORT="$EXPORT,K_VALUES=$K_VALUES"
EXPORT="$EXPORT,T_FIXED=$T_FIXED"
EXPORT="$EXPORT,ETA_FIXED=$ETA_FIXED"
EXPORT="$EXPORT,NUM_PARA_FIXED=$NUM_PARA_FIXED"
EXPORT="$EXPORT,SEEDS=$SEEDS"

# ── Submit chained segments ──────────────────────────────────────────────────
CMD_SEG1="sbatch --array=$ARRAY_INDICES --export=$EXPORT slurm/slurm_gt_sweep.sh"

echo ""
echo "Submitting Segment 1/3 (72h walltime each)..."
echo "  $CMD_SEG1"

if [[ $DRY_RUN -eq 0 ]]; then
    OUT1=$($CMD_SEG1)
    JID1=$(echo "$OUT1" | grep -oP '\d+' | head -1)
    echo "  → Segment 1 Job ID: $JID1"
    
    OUT2=$(sbatch --array=$ARRAY_INDICES --export=$EXPORT \
                  --dependency=afterany:$JID1 slurm/slurm_gt_sweep.sh)
    JID2=$(echo "$OUT2" | grep -oP '\d+' | head -1)
    echo "  → Segment 2 Job ID: $JID2 (starts after $JID1)"
    
    OUT3=$(sbatch --array=$ARRAY_INDICES --export=$EXPORT \
                  --dependency=afterany:$JID2 slurm/slurm_gt_sweep.sh)
    JID3=$(echo "$OUT3" | grep -oP '\d+' | head -1)
    echo "  → Segment 3 Job ID: $JID3 (starts after $JID2)"
else
    echo "  → [DRY RUN] Would submit 3 chained segments"
fi

echo ""
echo "======================================================================"
echo "Submission complete — monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/gt_sweep_*_*.out"
echo ""
echo "Output directory:"
echo "  $RUN_DIR"
echo ""
echo "When jobs finish, view summary:"
echo "  cat $RUN_DIR/sweep_summary.json"
echo "  python gt_semantic_sweep.py --analyze_only --save_dir $RUN_DIR \\"
echo "      --ckpt_detector $CKPT_DETECTOR"
echo "======================================================================"
