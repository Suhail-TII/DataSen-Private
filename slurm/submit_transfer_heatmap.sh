#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_transfer_heatmap.sh — Run Group 3 for all 5 remaining source tasks
#
# MRPC is already done. This submits one array task per remaining source:
#   0 = jfleg
#   1 = hsol
#   2 = rte
#   3 = sms_spam
#   4 = gigaword
#
# Results go into the SAME directory as your existing MRPC G3 run so that
# generate_transfer_heatmap.py can read all 6 sources from one place.
#
# USAGE
#   bash submit_transfer_heatmap.sh --existing-run-dir results/gt_ablation/run_YYYYMMDD
#   bash submit_transfer_heatmap.sh --existing-run-dir results/gt_ablation/run_YYYYMMDD --dry-run
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
EXISTING_RUN_DIR=""
CKPT_DETECTOR="/dpc/kuex0037/Open-Prompt-Injection/DataSentinel_Models/detector_large/checkpoint-5000"
PARAPHRASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
REPO_DIR="/dpc/kuex0037/Open-Prompt-Injection"
DATA_NUM=100
NUM_PARAPHRASES=20
MW_ROUNDS=100
MW_ETA=0.3
SEEDS="42,43,44"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --existing-run-dir)  EXISTING_RUN_DIR="$2"; shift 2 ;;
        --ckpt-detector)     CKPT_DETECTOR="$2";    shift 2 ;;
        --paraphrase-model)  PARAPHRASE_MODEL="$2"; shift 2 ;;
        --repo-dir)          REPO_DIR="$2";         shift 2 ;;
        --data-num)          DATA_NUM="$2";         shift 2 ;;
        --num-paraphrases)   NUM_PARAPHRASES="$2";  shift 2 ;;
        --mw-rounds)         MW_ROUNDS="$2";        shift 2 ;;
        --mw-eta)            MW_ETA="$2";           shift 2 ;;
        --seeds)             SEEDS="$2";            shift 2 ;;
        --dry-run)           DRY_RUN=1;             shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$EXISTING_RUN_DIR" ]]; then
    echo "ERROR: --existing-run-dir is required (path to your MRPC G3 run)"
    echo "Usage: bash submit_transfer_heatmap.sh --existing-run-dir results/gt_ablation/run_YYYYMMDD"
    exit 1
fi

if [[ ! -d "$EXISTING_RUN_DIR" ]]; then
    echo "ERROR: Directory not found: $EXISTING_RUN_DIR"
    exit 1
fi

# Results go into the existing run dir so all sources are co-located
RUN_DIR="$EXISTING_RUN_DIR"
mkdir -p logs

# Source tasks remaining (MRPC=mrpc already done)
SOURCE_TASKS=("jfleg" "hsol" "rte" "sms_spam" "gigaword")

echo "======================================================================"
echo "Transfer Heatmap — Additional Source Task Jobs"
echo "======================================================================"
echo "  Existing run dir  : $RUN_DIR"
echo "  Sources to run    : ${SOURCE_TASKS[*]}"
echo "  Seeds             : $SEEDS"
echo "  Detector ckpt     : $CKPT_DETECTOR"
echo "  Paraphrase model  : $PARAPHRASE_MODEL"
echo "  data_num          : $DATA_NUM"
echo "  num_paraphrases   : $NUM_PARAPHRASES"
echo "  mw_rounds         : $MW_ROUNDS / mw_eta : $MW_ETA"
echo "  Dry run           : $DRY_RUN"
echo "======================================================================"

export RUN_DIR
export CKPT_DETECTOR
export PARAPHRASE_MODEL
export REPO_DIR
export DATA_NUM
export NUM_PARAPHRASES
export MW_ROUNDS
export MW_ETA
export SEEDS
export SOURCE_TASKS_STR="${SOURCE_TASKS[*]}"

# Submit one array job covering all 5 sources (indices 0-4)
SBATCH_CMD="sbatch --array=0-4 --export=ALL slurm/slurm_transfer_heatmap.sh"

echo ""
echo "Submitting array job (5 source tasks in parallel)..."
echo "  $SBATCH_CMD"

if [[ $DRY_RUN -eq 0 ]]; then
    OUT1=$($SBATCH_CMD)
    JID1=$(echo "$OUT1" | grep -oP '\d+' | head -1)
    echo "  -> Job ID: $JID1"

    # Safety segment
    OUT2=$(sbatch --array=0-4 --export=ALL \
                  --dependency=afterany:$JID1 slurm/slurm_transfer_heatmap.sh)
    JID2=$(echo "$OUT2" | grep -oP '\d+' | head -1)
    echo "  -> Safety segment: $JID2 (after $JID1)"
else
    echo "  -> [DRY RUN] Would submit array=0-4 + safety segment"
fi

echo ""
echo "======================================================================"
echo "When complete, generate heatmap:"
echo "  python generate_transfer_heatmap.py --run_dir $RUN_DIR"
echo "======================================================================"