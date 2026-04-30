#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_transfer_source.sh — Run Group 3 transfer for one source task
#
# Designed to be submitted as an array job:
#   Array index → source task:
#     0 = jfleg
#     1 = hsol
#     2 = rte
#     3 = sms_spam
#     4 = gigaword
#
# Each source writes to its own isolated save_dir to avoid checkpoint
# contamination across sources. Results are merged after all jobs complete.
#
# Submit with:
#   sbatch --array=0-4 slurm/slurm_transfer_source.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuex0037
#SBATCH --qos=normal
#SBATCH --job-name=transfer_src
#SBATCH --output=logs/transfer_src_%A_%a.out
#SBATCH --error=logs/transfer_src_%A_%a.err
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

if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
    set -a; source "$SLURM_SUBMIT_DIR/.env"; set +a
fi
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" 2>/dev/null || true
fi

unset PYTORCH_CUDA_ALLOC_CONF

# Force transformers to use cached models only — avoids 401 auth errors
# on gated repos (LLaMA-3.1) since model is already in hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd $SLURM_SUBMIT_DIR

# ── Source task map ───────────────────────────────────────────────────────────
SOURCE_TASKS=("jfleg" "hsol" "rte" "sms_spam" "gigaword")
SOURCE=${SOURCE_TASKS[$SLURM_ARRAY_TASK_ID]}
SAVE_DIR="results/gt_ablation/transfer_${SOURCE}"
mkdir -p "$SAVE_DIR" logs

echo "======================================================================"
echo "Transfer Heatmap — Source Task: $SOURCE"
echo "Job        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Save dir   : $SAVE_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

python gt_semantic_ablation.py \
    --ckpt_detector DataSentinel_Models/detector_large/checkpoint-5000 \
    --paraphrase_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --repo_dir /dpc/kuex0037/Open-Prompt-Injection \
    --data_num 100 \
    --num_paraphrases 20 \
    --mw_rounds 100 \
    --mw_eta 0.3 \
    --seeds 42,43,44 \
    --groups 3 \
    --transfer_source $SOURCE \
    --save_dir $SAVE_DIR \
    --resume

STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS