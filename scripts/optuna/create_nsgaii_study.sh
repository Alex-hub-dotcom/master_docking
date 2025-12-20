#!/bin/bash
#SBATCH --job-name=create_study
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/schux00/logs/create_study_%j.out
#SBATCH --error=/home/schux00/logs/create_study_%j.err

cd /home/schux00/teko

enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --mount /home/schux00/optuna:/home/schux00/optuna \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh \
    /workspace/teko/scripts/optuna/train_optuna_nsgaii.py --create-study

echo "Done!"
