#!/bin/bash
#SBATCH --job-name=urdf_conv
#SBATCH --output=/home/schux00/logs/urdf_convert.out
#SBATCH --error=/home/schux00/logs/urdf_convert.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/teko/scripts/convert_urdf.py --headless
