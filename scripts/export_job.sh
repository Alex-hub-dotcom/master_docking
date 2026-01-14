#!/bin/bash
#SBATCH --job-name=export
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:10:00
#SBATCH --output=/home/schux00/logs/export_%j.out

cd /home/schux00/teko

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --mount /home/schux00/checkpoints:/home/schux00/checkpoints \
  --mount /home/schux00:/home/schux00 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh /workspace/teko/scripts/export_policy.py \
    --checkpoint /home/schux00/checkpoints/vision_attn_yaw_S27_52008k.pt \
    --output /home/schux00/teko_policy_S27_attn_yaw.pt

echo "Done!"
ls -la /home/schux00/teko_policy_S27_attn_yaw.pt
