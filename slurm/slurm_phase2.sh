#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_phase2.sh — Adaptive Defender Experiment (Phase 2)
#
# Single job (not an array) — rounds must run sequentially:
#   P0 extraction → Round 1 retrain → Round 1 attack
#               → Round 2 retrain → Round 2 attack
#
# Estimated walltime:
#   P0 extraction       ~3-4h  (re-runs MW on 6 tasks × 3 seeds)
#   Round 1 retrain     ~3-4h  (DataSentinel Algorithm 1, 3 rounds)
#   Round 1 attack      ~2-3h  (C4 on 6 tasks × 3 seeds)
#   Round 2 retrain     ~3-4h
#   Round 2 attack      ~2-3h
#   Total               ~13-18h
#
# We request 24h and chain a second segment for safety.
# The --resume flag means any interruption loses at most one (task, seed).
#
# Environment variables (set via --export=ALL from submit_phase2.sh):
#   RUN_DIR            output directory
#   BASE_CKPT          path to published DataSentinel checkpoint-5000
#   DETECTION_MODEL    HF model id for detection LLM
#   BACKEND_MODEL      HF model id for backend LLM
#   PARAPHRASE_MODEL   HF model id for paraphrase oracle
#   REPO_DIR           path to Open-Prompt-Injection repo
#   DATA_NUM           samples per task (default 100)
#   NUM_PARAPHRASES    |Phi_LLM| (default 20)
#   MW_ROUNDS          T for MW (default 100)
#   MW_ETA             MW learning rate (default 0.3)
#   TRAINING_ROUNDS    DataSentinel retraining rounds (default 3)
#   SKIP_ROUND         skip to round N if retraining already done (default 0)
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
#SBATCH --job-name=phase2_adaptive
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err
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
    hf auth login --token "$HUGGINGFACE_TOKEN" 2>/dev/null || \
        huggingface-cli login --token "$HUGGINGFACE_TOKEN" 2>/dev/null || \
        echo "Note: relying on cached HF token"
fi

unset PYTORCH_CUDA_ALLOC_CONF
cd $SLURM_SUBMIT_DIR

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_NUM=${DATA_NUM:-100}
NUM_PARAPHRASES=${NUM_PARAPHRASES:-20}
MW_ROUNDS=${MW_ROUNDS:-100}
MW_ETA=${MW_ETA:-0.3}
TRAINING_ROUNDS=${TRAINING_ROUNDS:-3}
SKIP_ROUND=${SKIP_ROUND:-0}
RETRAIN_LR=${RETRAIN_LR:-0.0000025}
RETRAIN_BETA=${RETRAIN_BETA:-0.5}
GRAD_CLIP=${GRAD_CLIP:-1.0}
PARAPHRASE_MODEL=${PARAPHRASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
DETECTION_MODEL=${DETECTION_MODEL:-mistralai/Mistral-7B-v0.1}
BACKEND_MODEL=${BACKEND_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
REPO_DIR=${REPO_DIR:-.}

echo "======================================================================"
echo "Phase 2: Adaptive Defender Experiment"
echo "Job        : $SLURM_JOB_ID"
echo "Base ckpt  : $BASE_CKPT"
echo "Output dir : $RUN_DIR"
echo "Skip round : $SKIP_ROUND"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

CMD="python phase2_adaptive_defender.py"
CMD="$CMD --base_ckpt $BASE_CKPT"
CMD="$CMD --detection_model $DETECTION_MODEL"
CMD="$CMD --backend_model $BACKEND_MODEL"
CMD="$CMD --paraphrase_model $PARAPHRASE_MODEL"
CMD="$CMD --repo_dir $REPO_DIR"
CMD="$CMD --data_num $DATA_NUM"
CMD="$CMD --num_paraphrases $NUM_PARAPHRASES"
CMD="$CMD --mw_rounds $MW_ROUNDS"
CMD="$CMD --mw_eta $MW_ETA"
CMD="$CMD --training_rounds $TRAINING_ROUNDS"
CMD="$CMD --retrain_lr $RETRAIN_LR"
CMD="$CMD --retrain_beta $RETRAIN_BETA"
CMD="$CMD --grad_clip $GRAD_CLIP"
CMD="$CMD --save_dir $RUN_DIR"
CMD="$CMD --resume"

if [ "$SKIP_ROUND" -gt 0 ]; then
    CMD="$CMD --skip_round $SKIP_ROUND"
fi

echo "Command: $CMD"
echo "----------------------------------------------------------------------"
$CMD

STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS