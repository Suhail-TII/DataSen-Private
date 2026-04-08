#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_gt_sweep.sh — GT-Semantic Hyperparameter Sweep job script
#
# Submitted as a Slurm array job by submit_gt_sweep.sh.
# Each array task handles one sweep: 0=T, 1=eta, 2=K.
#
# Required environment variables (set via --export):
#   RUN_DIR          : output directory for this run
#   CKPT_DETECTOR    : path to DataSentinel checkpoint
#   PARAPHRASE_MODEL : HF model name for paraphrase oracle
#   REPO_DIR         : path to Open-Prompt-Injection repo
#   DATA_NUM         : samples per task (default 100)
#   T_VALUES         : comma-separated values for T sweep
#   ETA_VALUES       : comma-separated values for eta sweep
#   K_VALUES         : comma-separated values for K sweep
#   T_FIXED          : fixed T when not the swept variable
#   ETA_FIXED        : fixed eta when not swept
#   NUM_PARA_FIXED   : fixed num_paraphrases when not swept
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
#SBATCH --job-name=gt_sweep
#SBATCH --output=logs/gt_sweep_%A_%a.out
#SBATCH --error=logs/gt_sweep_%A_%a.err
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

# ── Map array index → sweep ──────────────────────────────────────────────────
SWEEP_LIST=("T" "eta" "K")
SWEEP=${SWEEP_LIST[$SLURM_ARRAY_TASK_ID]}

echo "======================================================================"
echo "Job        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Sweep      : $SWEEP"
echo "Output dir : $RUN_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# ── Defaults ─────────────────────────────────────────────────────────────────
DATA_NUM=${DATA_NUM:-100}
PARAPHRASE_MODEL=${PARAPHRASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
REPO_DIR=${REPO_DIR:-.}
T_VALUES=${T_VALUES:-25,50,100,200,400}
ETA_VALUES=${ETA_VALUES:-0.1,0.2,0.3,0.5,1.0}
K_VALUES=${K_VALUES:-5,10,20,30,50}
T_FIXED=${T_FIXED:-100}
ETA_FIXED=${ETA_FIXED:-0.3}
NUM_PARA_FIXED=${NUM_PARA_FIXED:-20}

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python gt_semantic_sweep.py"
CMD="$CMD --ckpt_detector $CKPT_DETECTOR"
CMD="$CMD --paraphrase_model $PARAPHRASE_MODEL"
CMD="$CMD --repo_dir $REPO_DIR"
CMD="$CMD --data_num $DATA_NUM"
CMD="$CMD --sweeps $SWEEP"
CMD="$CMD --T_values $T_VALUES"
CMD="$CMD --eta_values $ETA_VALUES"
CMD="$CMD --K_values $K_VALUES"
CMD="$CMD --T_fixed $T_FIXED"
CMD="$CMD --eta_fixed $ETA_FIXED"
CMD="$CMD --num_paraphrases_fixed $NUM_PARA_FIXED"
CMD="$CMD --save_dir $RUN_DIR"
CMD="$CMD --resume"

echo "Command: $CMD"
echo "----------------------------------------------------------------------"

$CMD

STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS