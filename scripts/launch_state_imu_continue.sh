#!/bin/bash
#SBATCH --job-name=state_cont
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/schux00/logs/state_imu_continue_%j.out
#SBATCH --error=/home/schux00/logs/state_imu_continue_%j.err

echo "=============================================="
echo "TEKO State+IMU CONTINUE - S28 -> S32"
echo "=============================================="
echo "Job: ${SLURM_JOB_ID:-NA}"
echo "Node: ${SLURMD_NODENAME:-NA}"
echo "Started: $(date)"
echo "=============================================="

mkdir -p /home/schux00/logs
mkdir -p /home/schux00/checkpoints

cd /home/schux00/teko

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/logs:/workspace/logs \
  --mount /home/schux00/checkpoints:/home/schux00/checkpoints \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/train_state_imu_continue.py \
    --headless

echo "Finished: $(date)"
