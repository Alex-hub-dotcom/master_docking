#!/bin/bash
#SBATCH --job-name=inv_rev
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/schux00/logs/inv_rev_%j.out
#SBATCH --error=/home/schux00/logs/inv_rev_%j.err

set -euo pipefail

echo "=== INVESTIGATE REVERSE ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at $(date)"

cd /home/schux00/teko

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/investigate_reverse.py \
        --headless

echo "=== DONE at $(date) ==="