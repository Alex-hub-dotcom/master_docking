#!/usr/bin/env python3
import sys
import math
import torch
from isaaclab.app import AppLauncher

def extract_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)

def angle_wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

def compute_yaw_error(robot_yaw: torch.Tensor, robot_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
    vec_to_goal = goal_pos - robot_pos
    goal_yaw = torch.atan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    rear_yaw = robot_yaw + torch.pi
    return angle_wrap(rear_yaw - goal_yaw)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=13)
    parser.add_argument("--steps", type=int, default=400)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    args.enable_cameras = True

    app = AppLauncher(args)
    sim = app.app

    try:
        sys.path.insert(0, "/workspace/teko/source/teko")
        from teko.tasks.direct.teko.teko_env import TekoEnv
        from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

        cfg = TekoEnvCfg()
        cfg.scene.num_envs = 64
        cfg.enable_curriculum = True
        cfg.asymmetric_critic = True

        env = TekoEnv(cfg=cfg)
        env.set_curriculum_level(int(args.stage))
        obs, _ = env.reset()

        device = env.device
        N = cfg.scene.num_envs
        K = int(args.steps)

        improved = 0
        total = 0
        turning_correct_cnt = 0

        for _ in range(K):
            robot_quat0 = env.robot.data.root_quat_w
            robot_pos0  = env.robot.data.root_pos_w
            goal_pos    = env.goal_positions

            yaw0 = extract_yaw_wxyz(robot_quat0)
            yaw_err0 = compute_yaw_error(yaw0, robot_pos0, goal_pos)

            # random pure rotation
            w_cmd = (2.0 * torch.rand(N, device=device) - 1.0)
            v_cmd = torch.zeros(N, device=device)
            action = torch.stack([v_cmd, w_cmd], dim=-1)

            obs, reward, term, trunc, _ = env.step(action)

            robot_quat1 = env.robot.data.root_quat_w
            robot_pos1  = env.robot.data.root_pos_w

            yaw1 = extract_yaw_wxyz(robot_quat1)
            yaw_err1 = compute_yaw_error(yaw1, robot_pos1, goal_pos)

            # Did |yaw_error| improve?
            imp = (yaw_err1.abs() < yaw_err0.abs())
            improved += imp.sum().item()
            total += N

            # turning_bonus logic in your reward: (yaw_error * yaw_rate) < 0
            yaw_rate_after = env.robot.data.root_ang_vel_w[:, 2]
            turning_correct = (yaw_err0 * yaw_rate_after) < 0
            turning_correct_cnt += turning_correct.sum().item()

        print("\n=== TURNING LOGIC CHECK ===")
        print(f"Stage: {args.stage} | steps: {K} | envs: {N}")
        print("Frac improved |yaw_error| :", improved / total)
        print("Frac turning_correct (reward rule):", turning_correct_cnt / total)
        print("=================================\n")

        env.close()
    finally:
        sim.close()

if __name__ == "__main__":
    main()
