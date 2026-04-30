#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_t02_oracle.sh — T0.2: Cross-Model Oracle Validation
#
# Re-runs Phase 1 C4 attack against DataSentinel using Qwen2.5-7B-Instruct
# as the paraphrase oracle instead of LLaMA-3.1-8B-Instruct.
#
# Purpose: show that FNR ~0.95 is not a LLaMA-3 artifact but holds
# regardless of the paraphrase oracle used.
#
# Results go into a dedicated directory. Collect with:
#   python collect_phase1_results.py --results_dir results/t02_oracle/run_TIMESTAMP
#
# USAGE
#   bash submit_t02_oracle.sh
#   bash submit_t02_oracle.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CKPT_DETECTOR="/dpc/kuex0037/Open-Prompt-Injection/DataSentinel_Models/detector_large/checkpoint-5000"
PARAPHRASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
REPO_DIR="/dpc/kuex0037/Open-Prompt-Injection"
DATA_NUM=100
NUM_PARAPHRASES=20
MW_ROUNDS=100
MW_ETA=0.3
SEEDS="42,43,44"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/dpc/kuex0037/Open-Prompt-Injection/results/t02_oracle/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR" logs

echo "======================================================================"
echo "T0.2: Cross-Model Oracle Validation"
echo "======================================================================"
echo "  Oracle (new)      : $PARAPHRASE_MODEL"
echo "  Oracle (original) : meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "  Detector ckpt     : $CKPT_DETECTOR"
echo "  Seeds             : $SEEDS"
echo "  num_paraphrases   : $NUM_PARAPHRASES"
echo "  mw_rounds         : $MW_ROUNDS / mw_eta : $MW_ETA"
echo "  Output dir        : $RUN_DIR"
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

# Run DataSentinel detector only (not the alternative detectors)
# Reuse phase1 infrastructure but target only the main detector
SBATCH_CMD="sbatch --export=ALL slurm/slurm_t02_oracle.sh"

echo ""
echo "Submitting T0.2 oracle job..."
echo "  $SBATCH_CMD"

if [[ $DRY_RUN -eq 0 ]]; then
    OUT1=$($SBATCH_CMD)
    JID1=$(echo "$OUT1" | grep -oP '\d+' | head -1)
    echo "  -> Segment 1 Job ID: $JID1"

    OUT2=$(sbatch --export=ALL \
                  --dependency=afterany:$JID1 slurm/slurm_t02_oracle.sh)
    JID2=$(echo "$OUT2" | grep -oP '\d+' | head -1)
    echo "  -> Segment 2 Job ID: $JID2 (after $JID1)"
else
    echo "  -> [DRY RUN] Would submit 2 chained segments"
fi

echo ""
echo "======================================================================"
echo "Estimated runtime: ~4-6h (same as Phase 1 DataSentinel job)"
echo ""
echo "When done, collect results:"
echo "  python collect_t02_results.py --results_dir $RUN_DIR"
echo "======================================================================"