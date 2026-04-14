#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_phase1.sh — Cross-Detector Phase 1 SLURM job
#
# One array task per detector:
#   0 = KAD
#   1 = NLLMD
#   2 = PromptGuard
#   3 = EVD
#   4 = SSFTD      (skipped if --ssftd_path not set)
#   5 = SSFTD-G    (skipped if --ssftd_g_path not set)
#
# Environment variables (set via --export=ALL from submit_phase1.sh):
#   RUN_DIR            output directory
#   PARAPHRASE_MODEL   HF model name for LLM oracle
#   REPO_DIR           path to Open-Prompt-Injection repo
#   DATA_NUM           samples per task (default 100)
#   NUM_PARAPHRASES    |Φ_LLM| (default 20)
#   MW_ROUNDS          T for MW (default 100)
#   MW_ETA             MW learning rate (default 0.3)
#   SEEDS              comma-separated seeds (default 42,43,44)
#   SSFTD_PATH         optional — path to SSFTD LoRA checkpoint
#   SSFTD_G_PATH       optional — path to SSFTD-G LoRA checkpoint
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuex0037
#SBATCH --qos=normal
#SBATCH --job-name=phase1_xdet
#SBATCH --output=logs/phase1_%A_%a.out
#SBATCH --error=logs/phase1_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=100045490@ku.ac.ae

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel_h200_gpu
module load cuda/12.4
module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DataSen

echo "Python    : $(which python)"
echo "Conda env : $CONDA_DEFAULT_ENV"

# ── Storage ───────────────────────────────────────────────────────────────────
PROJECT_DIR=/dpc/kuex0037
export HF_HOME=$PROJECT_DIR/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$PROJECT_DIR/hf_datasets
mkdir -p $HF_HOME $HF_DATASETS_CACHE

# Load API keys / HF token from .env if present
if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
    set -a
    source "$SLURM_SUBMIT_DIR/.env"
    set +a
fi

if [ -n "$HUGGINGFACE_TOKEN" ]; then
    hf auth login --token "$HUGGINGFACE_TOKEN" 2>/dev/null || \
        huggingface-cli login --token "$HUGGINGFACE_TOKEN" 2>/dev/null || \
        echo "Note: HF login command not found, relying on cached token"
fi

unset PYTORCH_CUDA_ALLOC_CONF
cd $SLURM_SUBMIT_DIR

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_NUM=${DATA_NUM:-100}
NUM_PARAPHRASES=${NUM_PARAPHRASES:-20}
MW_ROUNDS=${MW_ROUNDS:-100}
MW_ETA=${MW_ETA:-0.3}
SEEDS=${SEEDS:-42,43,44}
PARAPHRASE_MODEL=${PARAPHRASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
REPO_DIR=${REPO_DIR:-.}

# ── Detector map: array index → detector name ─────────────────────────────────
DETECTORS=("kad" "nllmd" "promptguard" "evd" "ssftd" "ssftd_g")
DETECTOR=${DETECTORS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "Phase 1: Cross-Detector Evaluation"
echo "Job        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Detector   : $DETECTOR"
echo "Seeds      : $SEEDS"
echo "Output dir : $RUN_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python phase1_cross_detector.py"
CMD="$CMD --detector $DETECTOR"
CMD="$CMD --paraphrase_model $PARAPHRASE_MODEL"
CMD="$CMD --repo_dir $REPO_DIR"
CMD="$CMD --data_num $DATA_NUM"
CMD="$CMD --num_paraphrases $NUM_PARAPHRASES"
CMD="$CMD --mw_rounds $MW_ROUNDS"
CMD="$CMD --mw_eta $MW_ETA"
CMD="$CMD --seeds $SEEDS"
CMD="$CMD --save_dir $RUN_DIR"
CMD="$CMD --resume"

# Optional SSFTD checkpoint paths
if [ -n "${SSFTD_PATH:-}" ] && [ "$DETECTOR" = "ssftd" ]; then
    CMD="$CMD --ssftd_path $SSFTD_PATH"
fi
if [ -n "${SSFTD_G_PATH:-}" ] && [ "$DETECTOR" = "ssftd_g" ]; then
    CMD="$CMD --ssftd_g_path $SSFTD_G_PATH"
fi

# Skip SSFTD/SSFTD-G if no checkpoint provided
if [ "$DETECTOR" = "ssftd" ] && [ -z "${SSFTD_PATH:-}" ]; then
    echo "SKIPPING: --ssftd_path not set. Run with SSFTD_PATH=/path/to/ckpt."
    exit 0
fi
if [ "$DETECTOR" = "ssftd_g" ] && [ -z "${SSFTD_G_PATH:-}" ]; then
    echo "SKIPPING: --ssftd_g_path not set. Run with SSFTD_G_PATH=/path/to/ckpt."
    exit 0
fi

echo "Command: $CMD"
echo "----------------------------------------------------------------------"
$CMD

STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS