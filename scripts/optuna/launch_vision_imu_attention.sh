#!/bin/bash
#SBATCH --job-name=teko_attn
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=55G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/schux00/logs/teko_vision_imu_attention_%j.out
#SBATCH --error=/home/schux00/logs/teko_vision_imu_attention_%j.err
## Optional (uncomment to run multiple seeds as an array):
## #SBATCH --array=0-3%4

set -euo pipefail
IFS=$'\n\t'

SEED=$((42 + ${SLURM_ARRAY_TASK_ID:-0}))

echo "=============================================="
echo "TEKO Vision + IMU + Spatial Attention (v10)"
echo "=============================================="
echo "Job: ${SLURM_JOB_ID:-NA} | Seed: ${SEED}"
echo "Node: ${SLURMD_NODENAME:-NA}"
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
  --env PYTHONNOUSERSITE=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/optuna/train_optuna_vision_imu_attention.py \
    --seed "${SEED}" \
    --headless \
    --enable_cameras

echo "Finished: $(date)"
