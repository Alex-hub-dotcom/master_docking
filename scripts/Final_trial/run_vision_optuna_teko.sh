#!/bin/bash
#SBATCH --job-name=teko_v4
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=55G
#SBATCH --time=30-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/schux00/logs/optuna_v4_%A_%a.out
#SBATCH --error=/home/schux00/logs/optuna_v4_%A_%a.err

set -euo pipefail

echo "=============================================="
echo "TEKO Vision Optuna v5 - 50 Stages FINAL"
echo "=============================================="
echo "Job Array: ${SLURM_ARRAY_JOB_ID:-NA}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Started: $(date)"
echo ""
echo "Database: teko_vision_v5.db"
echo "Stages: 50 (S0-S41 precision + S42-S49 search)"
echo "Tolerance: fixed 2cm "
echo "=============================================="

mkdir -p /home/schux00/logs
mkdir -p /home/schux00/checkpoints
mkdir -p /home/schux00/tensorboard
mkdir -p /home/schux00/optuna

cd /home/schux00/teko

# Use task ID as seed offset for diversity
SEED=$((42 + ${SLURM_ARRAY_TASK_ID:-0}))

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/checkpoints:/home/schux00/checkpoints \
  --mount /home/schux00/tensorboard:/home/schux00/tensorboard \
  --mount /home/schux00/logs:/home/schux00/logs \
  --mount /home/schux00/optuna:/home/schux00/optuna \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/Final_trial/train_vision_optuna_teko.py \
    --seed $SEED \
    --headless \
    --enable_cameras

echo "Finished: $(date)"