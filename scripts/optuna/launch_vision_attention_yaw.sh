#!/bin/bash
#SBATCH --job-name=attn_yaw_opt
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/schux00/logs/teko_attn_yaw_optuna_%j.out
#SBATCH --error=/home/schux00/logs/teko_attn_yaw_optuna_%j.err

echo "=============================================="
echo "TEKO Vision + Attention + YawAux - Optuna"
echo "=============================================="

cd /home/schux00/teko

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/logs:/workspace/logs \
  --mount /home/schux00/checkpoints:/home/schux00/checkpoints \
  --mount /home/schux00/optuna:/home/schux00/optuna \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/optuna/train_vision_attention_yaw_optuna.py \
    --headless

echo "Finished: $(date)"
