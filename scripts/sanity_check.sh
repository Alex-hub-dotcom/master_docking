#!/bin/bash
#SBATCH --job-name=teko_sanity
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/schux00/logs/sanity_%j.out
#SBATCH --error=/home/schux00/logs/sanity_%j.err

cd /home/schux00/teko

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/teko_sanity_check.py \
        --test all \
        --num_envs 4 \
        --headless \
        --save_images