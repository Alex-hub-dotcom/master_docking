#!/bin/bash
#SBATCH --job-name=yawaux_dbg
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=55G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/schux00/logs/vision_yawaux_debug_%j.out
#SBATCH --error=/home/schux00/logs/vision_yawaux_debug_%j.err

echo "TEKO Vision+YawAux Debug"
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURMD_NODENAME}"
date

mkdir -p /home/schux00/logs /home/schux00/checkpoints
cd /home/schux00/teko

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/logs:/workspace/logs \
  --mount /home/schux00/checkpoints:/home/schux00/checkpoints \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/train_vision_yawaux_debug.py \
    --headless --enable_cameras

echo "Finished: $(date)"
