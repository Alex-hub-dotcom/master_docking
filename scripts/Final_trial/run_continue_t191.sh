#!/bin/bash
#SBATCH --job-name=T191_cont
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=55G
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/schux00/logs/continue_T191_%j.out
#SBATCH --error=/home/schux00/logs/continue_T191_%j.err

set -euo pipefail

echo "=============================================="
echo "CONTINUE T191 FROM CHECKPOINT"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Started: $(date)"
echo ""
echo "Checkpoint: BEST_T191_S49_76pct.pt"
echo "Target: 95% SSR at 1cm"
echo "Max time: 120h"
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
    /workspace/teko/scripts/Final_trial/continue_t191.py \
    --seed 42 \
    --headless \
    --enable_cameras

echo "Finished: $(date)"