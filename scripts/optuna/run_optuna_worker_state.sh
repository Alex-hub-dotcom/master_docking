#!/bin/bash
#SBATCH --job-name=teko_v5_state
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=96:00:00
#SBATCH --output=/home/schux00/logs/optuna_v5_state_%A_%a.out
#SBATCH --error=/home/schux00/logs/optuna_v5_state_%A_%a.err

set -euo pipefail

SEED=$((42 + ${SLURM_ARRAY_TASK_ID:-0}))

echo "=============================================="
echo "TEKO Optuna v5 - STATE-BASED (Debug)"
echo "=============================================="
echo "Job: $SLURM_JOB_ID | Array: ${SLURM_ARRAY_TASK_ID:-0} | Node: $SLURMD_NODENAME | Seed: $SEED"
echo "Started: $(date)"
echo "=============================================="

mkdir -p /home/schux00/logs
mkdir -p /home/schux00/optuna

cd /home/schux00/teko

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --mount /home/schux00/optuna:/home/schux00/optuna \
    --mount /home/schux00/logs:/workspace/logs \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/optuna/train_optuna_v5_state.py \
        --seed $SEED \
        --headless \
        --enable_cameras

echo "Finished: $(date)"
