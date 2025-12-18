#!/bin/bash
#SBATCH --job-name=list_prims
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/schux00/logs/list_prims_%j.out
#SBATCH --error=/home/schux00/logs/list_prims_%j.err

cd /home/schux00/teko

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --env PYTHONPATH=/workspace/teko/source/teko \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/sanity_checks/list_prims.py \
    --headless \
    --enable_cameras
