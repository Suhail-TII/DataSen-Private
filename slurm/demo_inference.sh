#!/bin/bash
#SBATCH --job-name=demo_inference
#SBATCH --account=kuex0037
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=demo_inference_%j.out
#SBATCH --error=demo_inference_%j.err

echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"

# ── Use existing cache dirs in project space ──────────────────────────────────
export HF_HOME=/dpc/kuex0037/hf_cache
export TRANSFORMERS_CACHE=/dpc/kuex0037/hf_cache
export HF_DATASETS_CACHE=/dpc/kuex0037/hf_datasets

# ── Conda init ────────────────────────────────────────────────────────────────
__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi
unset __conda_setup

conda activate DataSen
echo "Python: $(which python)"
echo "HF cache: $HF_HOME"

cd /dpc/kuex0037/Open-Prompt-Injection

python run_demo_inference.py

echo "Done: $(date)"

