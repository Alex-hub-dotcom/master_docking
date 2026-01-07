#!/bin/bash
#SBATCH --job-name=teko_imu
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/schux00/logs/teko_state_imu_%j.out
#SBATCH --error=/home/schux00/logs/teko_state_imu_%j.err

cd /home/schux00/teko

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --mount /home/schux00/optuna:/home/schux00/optuna \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/optuna/train_optuna_state_imu.py \
        --headless \
        --seed $RANDOM
