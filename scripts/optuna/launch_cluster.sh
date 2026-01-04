#!/bin/bash
# =============================================================================
# TEKO Training Cluster Launch Script
# =============================================================================
# Launches 8 workers: 6 vision-based + 2 state-based
# 
# Array indices:
#   0-5: Vision workers (train_optuna_vision.py)
#   6-7: State-based workers (train_optuna_state.py)
#
# Usage:
#   ./launch_cluster.sh           # Launch all 8 workers
#   ./launch_cluster.sh vision    # Launch only vision workers (0-5)
#   ./launch_cluster.sh state     # Launch only state workers (6-7)
#   ./launch_cluster.sh create    # Create Optuna studies only
#
# Author: Alexandre Schleier Neves da Silva
# =============================================================================

#SBATCH --job-name=teko_train
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/schux00/logs/teko_train_%A_%a.out
#SBATCH --error=/home/schux00/logs/teko_train_%A_%a.err

# =============================================================================
# Configuration
# =============================================================================

TEKO_ROOT="/home/schux00/teko"
CONTAINER="/home/schux00/alex_optuna_isaac.sqsh"
OPTUNA_DIR="/home/schux00/optuna"
LOG_DIR="/home/schux00/logs"

# Create directories if needed
mkdir -p "$OPTUNA_DIR"
mkdir -p "$LOG_DIR"

# =============================================================================
# Determine worker type based on array index
# =============================================================================

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TASK_ID" -le 5 ]; then
    WORKER_TYPE="vision"
    SCRIPT_NAME="train_optuna_vision.py"
    ENABLE_CAMERAS="--enable_cameras"
else
    WORKER_TYPE="state"
    SCRIPT_NAME="train_optuna_state.py"
    ENABLE_CAMERAS=""  # No cameras for state-based
fi

echo "=============================================="
echo "TEKO Training Worker"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${TASK_ID}"
echo "Worker Type: ${WORKER_TYPE}"
echo "Script: ${SCRIPT_NAME}"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo "=============================================="

# =============================================================================
# Run training
# =============================================================================

cd "$TEKO_ROOT"

enroot start --rw \
    --mount "${TEKO_ROOT}:/workspace/teko" \
    --mount "${OPTUNA_DIR}:/home/schux00/optuna" \
    --env PYTHONPATH=/workspace/teko/source/teko \
    --env PYTHONUNBUFFERED=1 \
    --env CUDA_VISIBLE_DEVICES=0 \
    "$CONTAINER" \
    /workspace/isaaclab/_isaac_sim/python.sh -u \
        "/workspace/teko/scripts/optuna/${SCRIPT_NAME}" \
        --headless \
        ${ENABLE_CAMERAS} \
        --seed $((42 + TASK_ID))

echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="