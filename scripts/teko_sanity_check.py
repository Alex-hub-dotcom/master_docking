# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Sanity Check - TiledCamera version
========================================
Adapted from teko_sanity_check.py to use TekoEnvTiledIMU (training env).
Saves camera frames to verify what the agent sees.
"""

import argparse
import sys
import os
import time

parser = argparse.ArgumentParser(description="TEKO Sanity Check (Tiled)")
parser.add_argument("--test", type=str, default="camera",
                    choices=["all", "geometry", "camera"],
                    help="Which test to run")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()

from isaaclab.app import AppLauncher
launcher_args = argparse.Namespace(headless=args.headless, enable_cameras=True)
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app

import torch
import numpy as np
from PIL import Image

from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env_tiled_imu import TekoEnvTiledIMU


def test_camera(env):
    """Test TiledCamera and save frames."""
    out_dir = f"/home/schux00/logs/sanity_tiled/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  TILED CAMERA CHECK")
    print("=" * 70)

    env.reset()

    # Warmup
    for _ in range(10):
        env.step(torch.zeros((env.num_envs, 2), device=env.device))

    # Capture frames
    for i in range(5):
        obs, _, _, _, _ = env.step(torch.zeros((env.num_envs, 2), device=env.device))

        # Grayscale frame stack (what agent sees)
        rgb = obs["rgb"][0]  # [4, 128, 128]
        frame = rgb[-1].clamp(0, 1)
        img = (frame * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(img, mode="L").save(f"{out_dir}/gray_frame_{i}.png")
        print(f"  Frame {i}: min={frame.min():.3f} max={frame.max():.3f} mean={frame.mean():.3f} std={frame.std():.3f}")

    # Save raw RGB
    env.tiled_camera.update(dt=0.0)
    raw = env.tiled_camera.data.output["rgb"][0]
    if raw.shape[-1] == 4:
        raw = raw[..., :3]
    Image.fromarray(raw.cpu().numpy().astype("uint8")).save(f"{out_dir}/raw_rgb.png")
    print(f"  Raw RGB shape: {raw.shape}")

    # Robot/goal info
    robot_pos = env.robot.data.root_pos_w[0].cpu().numpy()
    robot_quat = env.robot.data.root_quat_w[0].cpu().numpy()
    goal_pos = env.goal_positions[0].cpu().numpy()

    x, y, z, w = robot_quat
    yaw_rad = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_deg = np.rad2deg(yaw_rad)

    print(f"\n  Robot pos:  ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
    print(f"  Robot yaw:  {yaw_deg:.1f}°")
    print(f"  Goal pos:   ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f})")
    print(f"\n  Files saved to: {out_dir}")
    print("=" * 70)


def main():
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = args.num_envs

    env = TekoEnvTiledIMU(cfg=cfg)

    try:
        test_camera(env)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()