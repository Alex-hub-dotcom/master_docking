#!/bin/bash
#SBATCH --job-name=state_full
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=45G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/schux00/logs/state_full_%j.out
#SBATCH --error=/home/schux00/logs/state_full_%j.err

set -euo pipefail

echo "=============================================="
echo "TEKO State FULL - Oracle Baseline"
echo "=============================================="
echo "Job: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Started: $(date)"
echo ""
echo "This policy receives FULL state information."
echo "It's the oracle baseline to prove S41 is solvable."
echo "=============================================="

mkdir -p /home/schux00/logs
mkdir -p /home/schux00/checkpoints
mkdir -p /home/schux00/tensorboard

cd /home/schux00/teko

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/checkpoints:/home/schux00/checkpoints \
  --mount /home/schux00/tensorboard:/home/schux00/tensorboard \
  --mount /home/schux00/logs:/home/schux00/logs \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/Final_trial/train_state_full.py \
    --headless

echo "Finished: $(date)"