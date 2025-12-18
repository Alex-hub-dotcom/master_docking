#!/bin/bash
#SBATCH --job-name=sanity_camera
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/schux00/logs/sanity_camera_%j.out
#SBATCH --error=/home/schux00/logs/sanity_camera_%j.err

set -euo pipefail

echo "=== SANITY CHECK: CAMERA ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at $(date)"

cd /home/schux00/teko
mkdir -p /home/schux00/teko/scripts/sanity_checks/out/camera

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/sanity_checks/debug_spawn_and_snap_camera.py \
    --headless \
    --num_envs 1 \
    --warmup_steps 20 \
    --capture_steps 15 \
    --move_linear 0.8 \
    --output_dir /workspace/teko/scripts/sanity_checks/out/camera

echo "=== DONE at $(date) ==="