#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_gt_ablation.sh — Submits GT-Semantic Ablation jobs
#
# USAGE
#   bash submit_gt_ablation.sh                          # all 3 groups
#   bash submit_gt_ablation.sh --groups 1               # only Group 1
#   bash submit_gt_ablation.sh --groups 1,2             # Groups 1 and 2
#   bash submit_gt_ablation.sh --dry-run                # preview without submitting
#
# Each Group runs as a separate array task on its own GPU node, in parallel.
# Jobs use --resume so they can be killed and restarted without losing progress.
#
# Group 1 (Component isolation):  ~1-2 hours
# Group 2 (Algorithm comparison): ~3-4 hours (greedy enumeration is the slow part)
# Group 3 (Cross-task transfer):  ~2-3 hours
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
GROUPS="1,2,3"
CKPT_DETECTOR="/dpc/kuex0037/DataSentinel/DataSentinel_Models/detector_large/checkpoint-5000"
PARAPHRASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
REPO_DIR="/dpc/kuex0037/DataSentinel/Open-Prompt-Injection"
DATA_NUM=100
NUM_PARAPHRASES=20
MW_ROUNDS=100
MW_ETA=0.3
DRY_RUN=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --groups)            GROUPS="$2";            shift 2 ;;
        --ckpt-detector)     CKPT_DETECTOR="$2";     shift 2 ;;
        --paraphrase-model)  PARAPHRASE_MODEL="$2";  shift 2 ;;
        --repo-dir)          REPO_DIR="$2";          shift 2 ;;
        --data-num)          DATA_NUM="$2";          shift 2 ;;
        --num-paraphrases)   NUM_PARAPHRASES="$2";   shift 2 ;;
        --mw-rounds)         MW_ROUNDS="$2";         shift 2 ;;
        --mw-eta)            MW_ETA="$2";            shift 2 ;;
        --dry-run)           DRY_RUN=1;              shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Setup output directory ────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/dpc/kuex0037/DataSentinel/results/gt_ablation/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR" logs

echo "======================================================================"
echo "GT-Semantic Ablation — Job Submission"
echo "======================================================================"
echo "  Groups            : $GROUPS"
echo "  Checkpoint        : $CKPT_DETECTOR"
echo "  Paraphrase model  : $PARAPHRASE_MODEL"
echo "  Repo dir          : $REPO_DIR"
echo "  data_num          : $DATA_NUM"
echo "  num_paraphrases   : $NUM_PARAPHRASES"
echo "  mw_rounds         : $MW_ROUNDS"
echo "  mw_eta            : $MW_ETA"
echo "  Output dir        : $RUN_DIR"
echo "  Dry run           : $DRY_RUN"
echo "======================================================================"

# ── Convert groups list to array task indices ────────────────────────────────
# G1 → task 0, G2 → task 1, G3 → task 2
ARRAY_INDICES=""
IFS=',' read -ra GROUP_LIST <<< "$GROUPS"
for G in "${GROUP_LIST[@]}"; do
    G=$(echo $G | tr -d ' ')
    case $G in
        1) IDX=0 ;;
        2) IDX=1 ;;
        3) IDX=2 ;;
        *) echo "Unknown group: $G"; exit 1 ;;
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
EXPORT="$EXPORT,NUM_PARAPHRASES=$NUM_PARAPHRASES"
EXPORT="$EXPORT,MW_ROUNDS=$MW_ROUNDS"
EXPORT="$EXPORT,MW_ETA=$MW_ETA"

# ── Submit primary segment ────────────────────────────────────────────────────
CMD_SEG1="sbatch --array=$ARRAY_INDICES --export=$EXPORT slurm/slurm_gt_ablation.sh"

echo ""
echo "Submitting Segment 1/3 (72h walltime each)..."
echo "  $CMD_SEG1"

if [[ $DRY_RUN -eq 0 ]]; then
    OUT1=$($CMD_SEG1)
    JID1=$(echo "$OUT1" | grep -oP '\d+' | head -1)
    echo "  → Segment 1 Job ID: $JID1"
    
    # Chained segments — each waits for the previous one. Resume kicks in
    # automatically thanks to checkpointing. Total = 9 days walltime coverage.
    OUT2=$(sbatch --array=$ARRAY_INDICES --export=$EXPORT \
                  --dependency=afterany:$JID1 slurm/slurm_gt_ablation.sh)
    JID2=$(echo "$OUT2" | grep -oP '\d+' | head -1)
    echo "  → Segment 2 Job ID: $JID2 (starts after $JID1)"
    
    OUT3=$(sbatch --array=$ARRAY_INDICES --export=$EXPORT \
                  --dependency=afterany:$JID2 slurm/slurm_gt_ablation.sh)
    JID3=$(echo "$OUT3" | grep -oP '\d+' | head -1)
    echo "  → Segment 3 Job ID: $JID3 (starts after $JID2)"
else
    echo "  → [DRY RUN] Would submit 3 chained segments"
fi

echo ""
echo "======================================================================"
echo "Submission complete — monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/gt_ablation_*_*.out"
echo ""
echo "Output directory:"
echo "  $RUN_DIR"
echo ""
echo "When jobs finish, view summary:"
echo "  cat $RUN_DIR/ablation_summary.json"
echo "======================================================================"