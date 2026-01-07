#!/bin/bash
#SBATCH --job-name=teko_vimu
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/schux00/logs/teko_vision_imu_%A_%a.out
#SBATCH --error=/home/schux00/logs/teko_vision_imu_%A_%a.err

cd /home/schux00/teko

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --mount /home/schux00/optuna:/home/schux00/optuna \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/optuna/train_optuna_vision_imu.py \
        --headless \
        --enable_cameras \
        --seed $((42 + SLURM_ARRAY_TASK_ID))
