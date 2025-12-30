#!/bin/bash
#SBATCH --job-name=teko_yawcheck
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --output=/home/schux00/logs/yawcheck_%j.out
#SBATCH --error=/home/schux00/logs/yawcheck_%j.err

set -euo pipefail

mkdir -p /home/schux00/logs

echo "Node:    $SLURMD_NODENAME"
echo "Start:   $(date)"
echo "=============================================="

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/logs:/workspace/logs \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/sanity_scripts/check_yaw_error.py \
    --headless --enable_cameras

echo "=============================================="
echo "End:     $(date)"
