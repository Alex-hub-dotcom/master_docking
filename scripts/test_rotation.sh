#!/bin/bash
#SBATCH --job-name=rot_test
#SBATCH --output=/home/schux00/logs/rotation_test.out
#SBATCH --error=/home/schux00/logs/rotation_test.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

cd /home/schux00
enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/teko/scripts/test_rotation.py --headless
