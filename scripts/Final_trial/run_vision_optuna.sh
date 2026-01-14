#!/bin/bash
#SBATCH --job-name=vis_optuna
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=55G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/schux00/logs/vision_optuna_%A_%a.out
#SBATCH --error=/home/schux00/logs/vision_optuna_%A_%a.err
#SBATCH --array=0-2%3

set -euo pipefail

WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "=============================================="
echo "TEKO Vision Optuna - Worker ${WORKER_ID}"
echo "=============================================="
echo "Job: ${SLURM_ARRAY_JOB_ID:-NA} | Task: ${WORKER_ID}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Started: $(date)"
echo "=============================================="

mkdir -p /home/schux00/logs
mkdir -p /home/schux00/optuna
mkdir -p /home/schux00/tensorboard

cd /home/schux00/teko

# Create study on first run (only worker 0)
if [ "${WORKER_ID}" -eq 0 ]; then
    enroot start --rw \
      --mount /home/schux00/teko:/workspace/teko \
      --mount /home/schux00/optuna:/home/schux00/optuna \
      --env PYTHONPATH=/workspace/teko/source/teko \
      --env PYTHONUNBUFFERED=1 \
      /home/schux00/alex_optuna_isaac.sqsh \
      /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/Final_trial/train_vision_optuna.py \
        --create-study || true
    sleep 5
fi

# Small delay to avoid race conditions
sleep $((WORKER_ID * 3))

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/optuna:/home/schux00/optuna \
  --mount /home/schux00/tensorboard:/home/schux00/tensorboard \
  --mount /home/schux00/logs:/home/schux00/logs \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/Final_trial/train_vision_optuna.py \
    --headless \
    --enable_cameras

echo "Finished: $(date)"