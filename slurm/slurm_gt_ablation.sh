#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_gt_ablation.sh — GT-Semantic Ablation Study job script
#
# Submitted as a Slurm array job by submit_gt_ablation.sh.
# Each array task handles one Group: 0=G1, 1=G2, 2=G3.
#
# Required environment variables (set by submit_gt_ablation.sh via --export):
#   RUN_DIR          : output directory for this run
#   CKPT_DETECTOR    : path to DataSentinel checkpoint
#   PARAPHRASE_MODEL : HF model name for paraphrase oracle
#   REPO_DIR         : path to Open-Prompt-Injection repo
#   DATA_NUM         : samples per task (default 100)
#   NUM_PARAPHRASES  : |Φ_LLM| (default 20)
#   MW_ROUNDS        : T for MW (default 100)
#   MW_ETA           : MW learning rate (default 0.3)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuex0037
#SBATCH --qos=normal
#SBATCH --job-name=gt_ablation
#SBATCH --output=logs/gt_ablation_%A_%a.out
#SBATCH --error=logs/gt_ablation_%A_%a.err
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

# H200s have 141GB VRAM — no CPU offloading needed
unset PYTORCH_CUDA_ALLOC_CONF

cd $SLURM_SUBMIT_DIR

# ── Map array task index → group ─────────────────────────────────────────────
# 0 = G1 (component isolation)
# 1 = G2 (algorithm comparison)
# 2 = G3 (cross-task transfer)
GROUP_LIST=(1 2 3)
GROUP=${GROUP_LIST[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "Job        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Group      : G$GROUP"
echo "Output dir : $RUN_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# ── Set defaults if not provided ──────────────────────────────────────────────
DATA_NUM=${DATA_NUM:-100}
NUM_PARAPHRASES=${NUM_PARAPHRASES:-20}
MW_ROUNDS=${MW_ROUNDS:-100}
MW_ETA=${MW_ETA:-0.3}
PARAPHRASE_MODEL=${PARAPHRASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
REPO_DIR=${REPO_DIR:-.}

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python gt_semantic_ablation.py"
CMD="$CMD --ckpt_detector $CKPT_DETECTOR"
CMD="$CMD --paraphrase_model $PARAPHRASE_MODEL"
CMD="$CMD --repo_dir $REPO_DIR"
CMD="$CMD --data_num $DATA_NUM"
CMD="$CMD --num_paraphrases $NUM_PARAPHRASES"
CMD="$CMD --mw_rounds $MW_ROUNDS"
CMD="$CMD --mw_eta $MW_ETA"
CMD="$CMD --groups $GROUP"
CMD="$CMD --save_dir $RUN_DIR"
CMD="$CMD --resume"     # always resume from checkpoint if one exists

echo "Command: $CMD"
echo "----------------------------------------------------------------------"

$CMD

STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS