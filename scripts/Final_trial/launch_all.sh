#!/bin/bash
# =============================================================================
# TEKO Final Experiments - Launch All Jobs
# =============================================================================
# This script launches all experiments for the thesis:
# 1. Vision Optimal (1 job) - Best hyperparameters from Trial 80
# 2. Vision Optuna (3 jobs) - Hyperparameter search comparison
# 3. State Full (1 job) - Oracle baseline with full information
# =============================================================================

set -euo pipefail

SCRIPT_DIR="/home/schux00/teko/scripts/Final_trial"

echo "=============================================="
echo "TEKO Final Experiments Launcher"
echo "=============================================="
echo "Time: $(date)"
echo ""

# Check if scripts exist
for script in run_vision_optimal.sh run_vision_optuna.sh run_state_full.sh; do
    if [ ! -f "${SCRIPT_DIR}/${script}" ]; then
        echo "ERROR: ${script} not found in ${SCRIPT_DIR}"
        exit 1
    fi
done

echo "Launching experiments..."
echo ""

# 1. Vision Optimal
echo "[1/3] Launching Vision Optimal..."
JOB_OPTIMAL=$(sbatch "${SCRIPT_DIR}/run_vision_optimal.sh" | awk '{print $4}')
echo "      Job ID: ${JOB_OPTIMAL}"

# 2. Vision Optuna (3 workers)
echo "[2/3] Launching Vision Optuna (3 workers)..."
JOB_OPTUNA=$(sbatch "${SCRIPT_DIR}/run_vision_optuna.sh" | awk '{print $4}')
echo "      Job ID: ${JOB_OPTUNA} (array 0-2)"

# 3. State Full
echo "[3/3] Launching State Full..."
JOB_STATE=$(sbatch "${SCRIPT_DIR}/run_state_full.sh" | awk '{print $4}')
echo "      Job ID: ${JOB_STATE}"

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Summary:"
echo "  - Vision Optimal: ${JOB_OPTIMAL}"
echo "  - Vision Optuna:  ${JOB_OPTUNA}_[0-2]"
echo "  - State Full:     ${JOB_STATE}"
echo ""
echo "Total: 5 jobs (1 + 3 + 1)"
echo ""
echo "Monitor with:"
echo "  squeue -u schux00"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir=/home/schux00/tensorboard"
echo ""
echo "Logs:"
echo "  tail -f /home/schux00/logs/*.out"
echo ""