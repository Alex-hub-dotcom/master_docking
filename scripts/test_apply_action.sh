#!/bin/bash
#SBATCH --job-name=test_action
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/schux00/logs/test_action_%j.out
#SBATCH --error=/home/schux00/logs/test_action_%j.err

set -euo pipefail

echo "=== TEST APPLY ACTION ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at $(date)"

cd /home/schux00/teko

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/test_apply_action.py \
        --headless

echo "=== DONE at $(date) ==="