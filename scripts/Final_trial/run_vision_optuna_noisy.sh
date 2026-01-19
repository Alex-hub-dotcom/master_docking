#!/bin/bash
#SBATCH --job-name=vis_noisy
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=55G
#SBATCH --time=10-00:00:00
#SBATCH --output=/home/schux00/logs/vision_optuna_noisy_%A_%a.out
#SBATCH --error=/home/schux00/logs/vision_optuna_noisy_%A_%a.err
#SBATCH --array=0-2%3

set -euo pipefail

echo "=============================================="
echo "TEKO Vision Optuna v3 NOISY - NSGA-II + SQLite"
echo "=============================================="
echo "Job: ${SLURM_JOB_ID:-NA} | Array: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Started: $(date)"
echo "=============================================="

mkdir -p /home/schux00/logs
mkdir -p /home/schux00/checkpoints
mkdir -p /home/schux00/tensorboard
mkdir -p /home/schux00/optuna

cd /home/schux00/teko

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
    /workspace/teko/scripts/Final_trial/train_vision_optuna_noisy.py \
    --seed $SEED \
    --headless \
    --enable_cameras

echo "Finished: $(date)"