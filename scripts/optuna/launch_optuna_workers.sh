#!/bin/bash
# =============================================================================
# LAUNCH MULTIPLE OPTUNA WORKERS
# =============================================================================
# Usage:
#   ./launch_optuna_workers.sh 7    # Launch 7 workers
#   ./launch_optuna_workers.sh 4    # Launch 4 workers
#   ./launch_optuna_workers.sh      # Launch 4 workers (default)
# =============================================================================

set -euo pipefail

NUM_WORKERS=${1:-4}

echo "=============================================="
echo "Launching $NUM_WORKERS Optuna Workers"
echo "=============================================="

# Check if study exists
if [ ! -f "/home/schux00/optuna/teko_study.db" ]; then
    echo "⚠️  Study database not found!"
    echo "   Creating study first..."
    sbatch /home/schux00/teko/scripts/optuna/create_optuna_study.sh
    echo ""
    echo "   Wait for the init job to complete, then run this script again."
    exit 1
fi

# Launch workers
for i in $(seq 1 $NUM_WORKERS); do
    JOB_ID=$(sbatch --parsable /home/schux00/teko/scripts/optuna/run_optuna_worker.sh)
    echo "  Worker $i: Job $JOB_ID submitted"
done

echo ""
echo "=============================================="
echo "All $NUM_WORKERS workers submitted!"
echo ""
echo "Monitor with:"
echo "  squeue -u schux00"
echo "  tail -f /home/schux00/logs/optuna_*.out"
echo ""
echo "Check Optuna progress:"
echo "  python -c \"import optuna; s=optuna.load_study('teko_ppo_optimization_v1', 'sqlite:////home/schux00/optuna/teko_study.db'); print(f'Trials: {len(s.trials)}')\""
echo "=============================================="