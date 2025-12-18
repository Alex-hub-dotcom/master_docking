#!/bin/bash
#SBATCH --job-name=optuna_init
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/home/schux00/logs/optuna_init_%j.out
#SBATCH --error=/home/schux00/logs/optuna_init_%j.err

set -euo pipefail

echo "=============================================="
echo "Creating Optuna Study"
echo "=============================================="

mkdir -p /home/schux00/logs
mkdir -p /home/schux00/optuna

cd /home/schux00/teko

# Create study (just initializes the database, doesn't need GPU really)
enroot start --rw \
    --mount /home/schux00/teko:/workspace/teko \
    --mount /home/schux00/optuna:/workspace/optuna \
    --mount /home/schux00/optuna:/home/schux00/optuna \
    --env PYTHONPATH=/workspace/teko/source/teko \
    /home/schux00/alex_optuna_isaac.sqsh \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        /workspace/teko/scripts/optuna/train_optuna_ppo.py \
        --create-study \
        --headless


echo "=============================================="
echo "Study created! Now launch workers with:"
echo "  sbatch run_optuna_worker.sh"
echo "=============================================="