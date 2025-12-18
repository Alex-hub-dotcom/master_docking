#!/bin/bash
#SBATCH --job-name=sanity_wheels
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/schux00/logs/sanity_wheels_%j.out
#SBATCH --error=/home/schux00/logs/sanity_wheels_%j.err

set -euo pipefail

echo "=== SANITY CHECK: WHEELS & DYNAMICS ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at $(date)"

cd /home/schux00/teko
mkdir -p /home/schux00/teko/scripts/sanity_checks/out/wheels

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/sanity_checks/debug_wheels_and_dynamics.py \
    --headless \
    --num_envs 1 \
    --phase_duration 3.0 \
    --record_video 1 \
    --output_dir /workspace/teko/scripts/sanity_checks/out/wheels

echo "=== DONE at $(date) ==="