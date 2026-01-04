#!/bin/bash
#SBATCH --job-name=teko_state
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/schux00/logs/teko_state_%j.out
#SBATCH --error=/home/schux00/logs/teko_state_%j.err

cd /home/schux00/teko

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --mount /home/schux00/optuna:/home/schux00/optuna \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/optuna/train_optuna_state.py \
        --headless \
        --seed 42
