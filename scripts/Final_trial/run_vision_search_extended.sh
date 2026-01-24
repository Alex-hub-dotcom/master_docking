#!/bin/bash
#SBATCH --job-name=search_ext
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=55G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/schux00/logs/search_extended_%j.out
#SBATCH --error=/home/schux00/logs/search_extended_%j.err

set -euo pipefail

echo "=============================================="
echo "TEKO Vision SEARCH EXTENDED - Maximize SSR"
echo "=============================================="
echo "Job: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Started: $(date)"
echo "Target: 99% SSR"
echo "Max time: 72 hours"
echo "=============================================="

mkdir -p /home/schux00/logs
mkdir -p /home/schux00/checkpoints
mkdir -p /home/schux00/tensorboard

cd /home/schux00/teko

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/checkpoints:/home/schux00/checkpoints \
  --mount /home/schux00/tensorboard:/home/schux00/tensorboard \
  --mount /home/schux00/logs:/home/schux00/logs \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/Final_trial/train_vision_search_extended.py \
    --headless \
    --enable_cameras

echo "Finished: $(date)"