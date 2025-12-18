#!/bin/bash
#SBATCH --job-name=sanity_goal
#SBATCH --partition=compute
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/schux00/logs/sanity_goal_%j.out
#SBATCH --error=/home/schux00/logs/sanity_goal_%j.err

set -euo pipefail

echo "=== SANITY CHECK: SUCCESS POSE (TRAIN CRITERION) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at $(date)"

cd /home/schux00/teko
mkdir -p /home/schux00/teko/scripts/sanity_checks/out/goal

enroot start --rw \
  --mount /home/schux00/teko:/workspace/teko \
  --env PYTHONPATH=/workspace/teko/source/teko \
  --env PYTHONUNBUFFERED=1 \
  /home/schux00/alex_optuna_isaac.sqsh \
  /workspace/isaaclab/_isaac_sim/python.sh -u \
    /workspace/teko/scripts/sanity_checks/debug_spawn_goal_overlap.py \
    --headless \
    --num_envs 1 \
    --warmup_steps 10 \
    --settle_steps 10 \
    --active_robot_root_path "/World/envs/env_0/Robot" \
    --female_sphere_prim "/World/envs/env_0/Robot/teko_urdf/TEKO_Body/TEKO_ConnectorRear/SphereRear" \
    --male_sphere_prim "/World/envs/env_0/RobotGoal/teko_urdf/TEKO_Body/TEKO_ConnectorMale/TEKO_ConnectorPin/SpherePin" \
    --teleport 1 \
    --output_dir /workspace/teko/scripts/sanity_checks/out/goal \
    --cam_back 0.0 \
    --cam_side -2.5 \
    --cam_up 1.2 \
    --cam_yaw_deg 0 \
    --lookat_z 0.35

echo "=== DONE at $(date) ==="
