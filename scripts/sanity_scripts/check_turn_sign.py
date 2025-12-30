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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=13)
    parser.add_argument("--steps", type=int, default=200)
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

        w_list = []
        omega_list = []
        dyaw_list = []

        # Rodar com v=0 pra isolar rotação
        for _ in range(K):
            yaw0 = extract_yaw_wxyz(env.robot.data.root_quat_w)
            omega = env.robot.data.root_ang_vel_w[:, 2]

            w_cmd = (2.0 * torch.rand(N, device=device) - 1.0)  # [-1,1]
            v_cmd = torch.zeros(N, device=device)
            action = torch.stack([v_cmd, w_cmd], dim=-1)

            obs, reward, term, trunc, _ = env.step(action)

            yaw1 = extract_yaw_wxyz(env.robot.data.root_quat_w)
            dyaw = angle_wrap(yaw1 - yaw0)

            w_list.append(w_cmd.detach())
            omega_list.append(omega.detach())
            dyaw_list.append(dyaw.detach())

        w = torch.cat(w_list)
        omega = torch.cat(omega_list)
        dyaw = torch.cat(dyaw_list)

        def mean_item(x): return x.mean().item()

        print("\n=== TURN SIGN SANITY CHECK ===")
        print(f"Stage: {args.stage} | steps: {K} | envs: {N}")
        print("mean(w_cmd * omega):", mean_item(w * omega))
        print("mean(omega * delta_yaw):", mean_item(omega * dyaw))
        print("mean(w_cmd * delta_yaw):", mean_item(w * dyaw))
        print("================================\n")

        env.close()

    finally:
        sim.close()

if __name__ == "__main__":
    main()
