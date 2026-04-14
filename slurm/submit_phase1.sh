#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# submit_phase1.sh — Submit Phase 1 Cross-Detector Evaluation jobs
#
# USAGE
#   bash submit_phase1.sh                            # all 4 detectors (no SSFTD)
#   bash submit_phase1.sh --detectors kad,promptguard
#   bash submit_phase1.sh --ssftd-path /path/ckpt   # include SSFTD
#   bash submit_phase1.sh --dry-run                  # preview without submitting
#
# Each detector gets its own array task on its own GPU node, all in parallel.
# Walltime is 24h per segment; three chained segments = 72h total coverage.
# Checkpointing means any failed job can be restarted from where it stopped.
#
# Estimated runtimes per detector (H200, data_num=100, 3 seeds):
#   KAD         ~4-6h   (Mistral-7B inference per query, no LLM oracle reload)
#   NLLMD       ~4-6h   (same)
#   PromptGuard ~1-2h   (DeBERTa, fast inference)
#   EVD         ~1-2h   (CLIP, fast inference; +30min first-run training)
#   SSFTD       ~4-6h   (same as KAD)
#   SSFTD-G     ~4-6h   (same)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_DETECTORS="kad,nllmd,promptguard,evd"   # SSFTD excluded by default
PARAPHRASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
REPO_DIR="/dpc/kuex0037/Open-Prompt-Injection"
DATA_NUM=100
NUM_PARAPHRASES=20
MW_ROUNDS=100
MW_ETA=0.3
SEEDS="42,43,44"
SSFTD_PATH=""
SSFTD_G_PATH=""
DRY_RUN=0

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --detectors)        RUN_DETECTORS="$2";     shift 2 ;;
        --paraphrase-model) PARAPHRASE_MODEL="$2";  shift 2 ;;
        --repo-dir)         REPO_DIR="$2";          shift 2 ;;
        --data-num)         DATA_NUM="$2";          shift 2 ;;
        --num-paraphrases)  NUM_PARAPHRASES="$2";   shift 2 ;;
        --mw-rounds)        MW_ROUNDS="$2";         shift 2 ;;
        --mw-eta)           MW_ETA="$2";            shift 2 ;;
        --seeds)            SEEDS="$2";             shift 2 ;;
        --ssftd-path)       SSFTD_PATH="$2";        shift 2 ;;
        --ssftd-g-path)     SSFTD_G_PATH="$2";      shift 2 ;;
        --dry-run)          DRY_RUN=1;              shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Detector name → array index map ──────────────────────────────────────────
# Must match DETECTORS array in slurm_phase1.sh
declare -A DET_IDX=(
    [kad]=0
    [nllmd]=1
    [promptguard]=2
    [evd]=3
    [ssftd]=4
    [ssftd_g]=5
)

# ── Build array indices for requested detectors ───────────────────────────────
ARRAY_INDICES=""
IFS=',' read -ra DET_LIST <<< "$RUN_DETECTORS"
for DET in "${DET_LIST[@]}"; do
    DET=$(echo "$DET" | tr -d ' ')
    if [[ ! -v DET_IDX[$DET] ]]; then
        echo "Unknown detector: $DET (choices: kad nllmd promptguard evd ssftd ssftd_g)"
        exit 1
    fi
    IDX=${DET_IDX[$DET]}
    # Warn if SSFTD requested without checkpoint
    if [ "$DET" = "ssftd" ] && [ -z "$SSFTD_PATH" ]; then
        echo "WARNING: ssftd requested but --ssftd-path not provided. Job will skip at runtime."
    fi
    if [ "$DET" = "ssftd_g" ] && [ -z "$SSFTD_G_PATH" ]; then
        echo "WARNING: ssftd_g requested but --ssftd-g-path not provided. Job will skip at runtime."
    fi
    ARRAY_INDICES="${ARRAY_INDICES:+$ARRAY_INDICES,}$IDX"
done

# ── Output directory ──────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="/dpc/kuex0037/Open-Prompt-Injection/results/phase1/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR" logs

echo "======================================================================"
echo "Phase 1: Cross-Detector Evaluation — Job Submission"
echo "======================================================================"
echo "  Detectors         : $RUN_DETECTORS"
echo "  Array indices     : $ARRAY_INDICES"
echo "  Seeds             : $SEEDS"
echo "  Paraphrase model  : $PARAPHRASE_MODEL"
echo "  Repo dir          : $REPO_DIR"
echo "  data_num          : $DATA_NUM"
echo "  num_paraphrases   : $NUM_PARAPHRASES"
echo "  mw_rounds         : $MW_ROUNDS"
echo "  mw_eta            : $MW_ETA"
echo "  SSFTD path        : ${SSFTD_PATH:-<not set>}"
echo "  SSFTD-G path      : ${SSFTD_G_PATH:-<not set>}"
echo "  Output dir        : $RUN_DIR"
echo "  Dry run           : $DRY_RUN"
echo "======================================================================"

# ── Export variables (avoids sbatch comma-parsing issues with SEEDS) ──────────
export RUN_DIR
export PARAPHRASE_MODEL
export REPO_DIR
export DATA_NUM
export NUM_PARAPHRASES
export MW_ROUNDS
export MW_ETA
export SEEDS
export SSFTD_PATH
export SSFTD_G_PATH

# ── Submit three chained segments (72h total coverage) ────────────────────────
SBATCH_CMD="sbatch --array=$ARRAY_INDICES --export=ALL slurm/slurm_phase1.sh"

echo ""
echo "Submitting Segment 1/3 (24h walltime)..."
echo "  $SBATCH_CMD"

if [[ $DRY_RUN -eq 0 ]]; then
    OUT1=$($SBATCH_CMD)
    JID1=$(echo "$OUT1" | grep -oP '\d+' | head -1)
    echo "  → Segment 1 Job ID: $JID1"

    OUT2=$(sbatch --array=$ARRAY_INDICES --export=ALL \
                  --dependency=afterany:$JID1 slurm/slurm_phase1.sh)
    JID2=$(echo "$OUT2" | grep -oP '\d+' | head -1)
    echo "  → Segment 2 Job ID: $JID2 (after $JID1)"

    OUT3=$(sbatch --array=$ARRAY_INDICES --export=ALL \
                  --dependency=afterany:$JID2 slurm/slurm_phase1.sh)
    JID3=$(echo "$OUT3" | grep -oP '\d+' | head -1)
    echo "  → Segment 3 Job ID: $JID3 (after $JID2)"
else
    echo "  → [DRY RUN] Would submit:"
    echo "     Segment 1: $SBATCH_CMD"
    echo "     Segment 2: (after Segment 1)"
    echo "     Segment 3: (after Segment 2)"
fi

echo ""
echo "======================================================================"
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/phase1_*_*.out"
echo ""
echo "Results directory:"
echo "  $RUN_DIR"
echo ""
echo "Per-detector summaries (when done):"
for DET in "${DET_LIST[@]}"; do
    echo "  $RUN_DIR/${DET}_summary.json"
done
echo ""
echo "Collect all results when done:"
echo "  python collect_phase1_results.py --results_dir $RUN_DIR"
echo "======================================================================"