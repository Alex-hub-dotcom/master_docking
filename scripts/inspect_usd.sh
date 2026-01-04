#!/bin/bash
#SBATCH --job-name=usd_insp
#SBATCH --output=/home/schux00/logs/usd_inspect.out
#SBATCH --error=/home/schux00/logs/usd_inspect.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh /workspace/teko/scripts/inspect_usd.py
