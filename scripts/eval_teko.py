# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import argparse, time, os, sys

# 1) Launch Kit before importing omni/* users
from isaaclab.app import AppLauncher


def main(args):
    app = AppLauncher(headless=args.headless).app

    # 2) Make sure Python can find your package tree
    repo_root = os.path.dirname(os.path.dirname(__file__))           # /workspace/teko
    sys.path.append(os.path.join(repo_root, "source", "teko"))       # import teko.…
    sys.path.append("/workspace/isaaclab/source")

    import torch
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # Build cfg
    cfg = TekoEnvCfg()
    cfg.independent_wheels = args.independent
    cfg.action_space = 4 if args.independent else 2
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = args.env_spacing
    cfg.episode_length_s = args.episode_len
    cfg.action_scale = args.scale
    cfg.spawn_height = args.spawn_height

    # Create env
    env = TekoEnv(cfg)
    env.reset()

    device = env.device
    N = cfg.scene.num_envs

    # Constant actions
    if args.independent:
        a = torch.tensor([args.fl, args.fr, args.rl, args.rr], dtype=torch.float32, device=device)
    else:
        a = torch.tensor([args.left, args.right], dtype=torch.float32, device=device)
    actions = a.unsqueeze(0).repeat(N, 1).contiguous()

    # Run
    steps = int(args.seconds * (120 / cfg.decimation))
    t0 = time.time()
    for _ in range(steps):
        env.step(actions)
    print(f"done in {time.time() - t0:.2f}s")

    app.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--independent", action="store_true", help="4-wheel independent mode")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--env-spacing", type=float, default=4.0)
    p.add_argument("--episode-len", type=float, default=5.0)
    p.add_argument("--scale", type=float, default=10.0)      # safer default
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--spawn-height", type=float, default=0.05)
    # tracks (default)
    p.add_argument("--left", type=float, default=0.6)
    p.add_argument("--right", type=float, default=0.6)
    # independent
    p.add_argument("--fl", type=float, default=0.6)
    p.add_argument("--fr", type=float, default=0.6)
    p.add_argument("--rl", type=float, default=0.6)
    p.add_argument("--rr", type=float, default=0.6)
    main(p.parse_args())
