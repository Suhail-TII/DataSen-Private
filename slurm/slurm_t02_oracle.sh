#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm_t02_oracle.sh — T0.2: Cross-Model Oracle Validation
#
# Runs the full C4 attack (MW + Phi_LLM) against DataSentinel using
# Qwen2.5-7B-Instruct as the paraphrase oracle.
# Same hyperparameters as main results: T=100, eta=0.3, |Phi|=20, seeds 42,43,44
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
#SBATCH --job-name=t02_oracle
#SBATCH --output=logs/t02_oracle_%j.out
#SBATCH --error=logs/t02_oracle_%j.err
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

DATA_NUM=${DATA_NUM:-100}
NUM_PARAPHRASES=${NUM_PARAPHRASES:-20}
MW_ROUNDS=${MW_ROUNDS:-100}
MW_ETA=${MW_ETA:-0.3}
SEEDS=${SEEDS:-42,43,44}
PARAPHRASE_MODEL=${PARAPHRASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
REPO_DIR=${REPO_DIR:-.}

echo "======================================================================"
echo "T0.2: Cross-Model Oracle Validation"
echo "Job        : $SLURM_JOB_ID"
echo "Oracle     : $PARAPHRASE_MODEL"
echo "Output dir : $RUN_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

CMD="python gt_semantic_attack.py"
CMD="$CMD --ckpt_detector $CKPT_DETECTOR"
CMD="$CMD --paraphrase_model $PARAPHRASE_MODEL"
CMD="$CMD --repo_dir $REPO_DIR"
CMD="$CMD --data_num $DATA_NUM"
CMD="$CMD --num_paraphrases $NUM_PARAPHRASES"
CMD="$CMD --mw_rounds $MW_ROUNDS"
CMD="$CMD --mw_eta $MW_ETA"
CMD="$CMD --save_dir $RUN_DIR"
CMD="$CMD --skip_baseline"
CMD="$CMD --seed 42"

echo "Command: $CMD"
echo "Note: gt_semantic_attack.py runs one seed at a time."
echo "Running seeds 42, 43, 44 sequentially..."
echo "----------------------------------------------------------------------"

for SEED in 42 43 44; do
    echo ""
    echo "=== Seed $SEED ==="
    python gt_semantic_attack.py \
        --ckpt_detector $CKPT_DETECTOR \
        --paraphrase_model $PARAPHRASE_MODEL \
        --repo_dir $REPO_DIR \
        --data_num $DATA_NUM \
        --num_paraphrases $NUM_PARAPHRASES \
        --mw_rounds $MW_ROUNDS \
        --mw_eta $MW_ETA \
        --save_dir ${RUN_DIR}/seed_${SEED} \
        --skip_baseline \
        --seed $SEED
done

STATUS=$?
echo "----------------------------------------------------------------------"
echo "All seeds complete. Exit status: $STATUS"
exit $STATUS