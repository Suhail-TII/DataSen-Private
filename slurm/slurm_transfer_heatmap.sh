#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_transfer_heatmap.sh — Group 3 cross-task transfer for one source task
#
# Array index → source task:
#   0 = jfleg
#   1 = hsol
#   2 = rte
#   3 = sms_spam
#   4 = gigaword
#
# Writes checkpoints into the existing run dir alongside the MRPC results,
# so generate_transfer_heatmap.py can read all 6 sources from one place.
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
#SBATCH --job-name=transfer_heatmap
#SBATCH --output=logs/transfer_heatmap_%A_%a.out
#SBATCH --error=logs/transfer_heatmap_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=100045490@ku.ac.ae

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load intel_h200_gpu
module load cuda/12.4
module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DataSen

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
cd $SLURM_SUBMIT_DIR

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_NUM=${DATA_NUM:-100}
NUM_PARAPHRASES=${NUM_PARAPHRASES:-20}
MW_ROUNDS=${MW_ROUNDS:-100}
MW_ETA=${MW_ETA:-0.3}
SEEDS=${SEEDS:-42,43,44}
PARAPHRASE_MODEL=${PARAPHRASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
REPO_DIR=${REPO_DIR:-.}

# ── Source task map ───────────────────────────────────────────────────────────
SOURCE_TASKS=("jfleg" "hsol" "rte" "sms_spam" "gigaword")
SOURCE=${SOURCE_TASKS[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "Transfer Heatmap — Source Task: $SOURCE"
echo "Job        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Output dir : $RUN_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

CMD="python gt_semantic_ablation.py"
CMD="$CMD --ckpt_detector $CKPT_DETECTOR"
CMD="$CMD --paraphrase_model $PARAPHRASE_MODEL"
CMD="$CMD --repo_dir $REPO_DIR"
CMD="$CMD --data_num $DATA_NUM"
CMD="$CMD --num_paraphrases $NUM_PARAPHRASES"
CMD="$CMD --mw_rounds $MW_ROUNDS"
CMD="$CMD --mw_eta $MW_ETA"
CMD="$CMD --seeds $SEEDS"
CMD="$CMD --groups 3"
CMD="$CMD --transfer_source $SOURCE"
CMD="$CMD --save_dir $RUN_DIR"
CMD="$CMD --resume"

echo "Command: $CMD"
echo "----------------------------------------------------------------------"
$CMD

STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS