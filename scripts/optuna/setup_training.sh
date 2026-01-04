#!/bin/bash
# =============================================================================
# TEKO Training Setup Script
# =============================================================================
# Run this ONCE before launching workers to:
# 1. Create Optuna study databases
# 2. Copy training scripts to teko/scripts/
# 3. Copy state environment to proper location
#
# Usage:
#   ./setup_training.sh
#
# Author: Alexandre Schleier Neves da Silva
# =============================================================================

set -e

TEKO_ROOT="/home/schux00/teko"
OPTUNA_DIR="/home/schux00/optuna"
CONTAINER="/home/schux00/alex_optuna_isaac.sqsh"

echo "=============================================="
echo "TEKO Training Setup"
echo "=============================================="

# Create directories
echo "[1/4] Creating directories..."
mkdir -p "$OPTUNA_DIR"
mkdir -p "$TEKO_ROOT/scripts"
mkdir -p "/home/schux00/logs"

# Copy environment file
echo "[2/4] Copying state-based environment..."
cp /home/claude/teko_env_state.py \
   "$TEKO_ROOT/source/teko/teko/tasks/direct/teko/teko_env_state.py"

# Copy training scripts
echo "[3/4] Copying training scripts..."
cp /home/claude/train_optuna_vision.py "$TEKO_ROOT/scripts/"
cp /home/claude/train_optuna_state.py "$TEKO_ROOT/scripts/"
cp /home/claude/launch_cluster.sh "$TEKO_ROOT/scripts/"
chmod +x "$TEKO_ROOT/scripts/launch_cluster.sh"

# Create Optuna studies
echo "[4/4] Creating Optuna studies..."

# Vision study
enroot start --rw \
    --mount "${TEKO_ROOT}:/workspace/teko" \
    --mount "${OPTUNA_DIR}:/home/schux00/optuna" \
    --env PYTHONPATH=/workspace/teko/source/teko \
    "$CONTAINER" \
    /workspace/isaaclab/_isaac_sim/python.sh -c "
import sys
sys.path.insert(0, '/workspace/teko/scripts')
from train_optuna_vision import make_storage, create_study, OPTUNA_CONFIG
storage = make_storage(OPTUNA_CONFIG['storage_path'])
create_study(OPTUNA_CONFIG['study_name'], storage)
print('[OK] Vision study created')
"

# State study
enroot start --rw \
    --mount "${TEKO_ROOT}:/workspace/teko" \
    --mount "${OPTUNA_DIR}:/home/schux00/optuna" \
    --env PYTHONPATH=/workspace/teko/source/teko \
    "$CONTAINER" \
    /workspace/isaaclab/_isaac_sim/python.sh -c "
import sys
sys.path.insert(0, '/workspace/teko/scripts')
from train_optuna_state import make_storage, create_study, OPTUNA_CONFIG
storage = make_storage(OPTUNA_CONFIG['storage_path'])
create_study(OPTUNA_CONFIG['study_name'], storage)
print('[OK] State study created')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Files created:"
echo "  - $TEKO_ROOT/source/teko/teko/tasks/direct/teko/teko_env_state.py"
echo "  - $TEKO_ROOT/scripts/train_optuna_vision.py"
echo "  - $TEKO_ROOT/scripts/train_optuna_state.py"
echo "  - $TEKO_ROOT/scripts/launch_cluster.sh"
echo "  - $OPTUNA_DIR/teko_vision_v6.db"
echo "  - $OPTUNA_DIR/teko_state_v6.db"
echo ""
echo "To launch training:"
echo "  cd $TEKO_ROOT/scripts"
echo "  sbatch launch_cluster.sh"
echo ""
echo "To monitor:"
echo "  watch -n 30 'squeue -u schux00'"
echo "  tail -f /home/schux00/logs/teko_train_*.out"
echo ""