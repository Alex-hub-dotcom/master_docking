#!/usr/bin/env python3
import sys
import math
import torch

from isaaclab.app import AppLauncher

# --- same yaw extraction as your UPDATED reward_functions.py (w, x, y, z) ---
def extract_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)

def angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def compute_yaw_error(robot_yaw: torch.Tensor, robot_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    rear_yaw = robot_yaw + torch.pi
    return angle_wrap(rear_yaw - goal_yaw)

def main():
    # AppLauncher args
    import argparse
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # headless, but cameras ON (your env uses TiledCamera)
    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args)
    sim = app.app

    try:
        # Make TEKO importable
        sys.path.insert(0, "/workspace/teko/source/teko")

        from teko.tasks.direct.teko.teko_env import TekoEnv
        from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

        cfg = TekoEnvCfg()
        cfg.scene.num_envs = 32  # small batch for fast debug
        cfg.enable_curriculum = True
        cfg.asymmetric_critic = True

        env = TekoEnv(cfg=cfg)

        # Stage 0 reset
        env.set_curriculum_level(0)
        obs, _ = env.reset()

        robot_quat = env.robot.data.root_quat_w
        robot_pos  = env.robot.data.root_pos_w
        goal_pos   = env.goal_positions

        robot_yaw = extract_yaw_wxyz(robot_quat)
        yaw_error = compute_yaw_error(robot_yaw, robot_pos, goal_pos)

        yaw_abs = yaw_error.abs()
        yaw_abs_deg = yaw_abs * 180.0 / math.pi

        print("\n=== YAW ERROR SANITY CHECK (Stage 0 reset) ===")
        print("yaw_abs mean (deg):", yaw_abs_deg.mean().item())
        print("yaw_abs min  (deg):", yaw_abs_deg.min().item())
        print("yaw_abs max  (deg):", yaw_abs_deg.max().item())
        print("robot_yaw mean (deg):", (robot_yaw.mean().item() * 180.0 / math.pi))
        print("============================================\n")

        env.close()

    finally:
        sim.close()

if __name__ == "__main__":
    main()
