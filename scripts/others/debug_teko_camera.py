# SPDX-License-Identifier: BSD-3-Clause
"""
TEKO Camera Diagnostics
=======================
Goal: Verify that the environment is returning RGB observations with:
- Correct shape: [num_envs, 3, H, W]
- Correct range: [0.0, 1.0] (float32)
- Reasonable image content (not all zeros or NaNs)

Usage:
    /workspace/isaaclab/_isaac_sim/python.sh scripts/debug_teko_camera.py
"""

from isaaclab.app import AppLauncher
import torch
import numpy as np
import time
import os

# Optional: save a sample frame as PNG
SAVE_IMAGE = True
OUTPUT_DIR = "camera_debug"


def main(headless=False):
    # ---------------------------------------------------------------
    # 1. Launch Isaac Lab with cameras enabled
    # ---------------------------------------------------------------
    app_launcher = AppLauncher(
        headless=headless,
        enable_cameras=True,
    )
    app = app_launcher.app

    # Import AFTER app init
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # ---------------------------------------------------------------
    # 2. Create env (single env for simplicity)
    # ---------------------------------------------------------------
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.render_interval = 1

    env = TekoEnv(cfg)
    env.reset()
    print("[INFO] TEKO environment ready (camera diagnostics).")

    # Give the sim a few frames to fully spawn sensors
    for _ in range(5):
        app.update()
        time.sleep(0.05)

    # ---------------------------------------------------------------
    # 3. Grab observations directly from env
    # ---------------------------------------------------------------
    # Option A: directly call _get_observations (for diagnostics only)
    obs = env._get_observations()
    rgb = obs["rgb"]  # Tensor [B, 3, H, W] on env.device

    print("\n[INFO] Raw RGB observation:")
    print(f"  type       : {type(rgb)}")
    print(f"  dtype      : {rgb.dtype}")
    print(f"  device     : {rgb.device}")
    print(f"  shape      : {tuple(rgb.shape)}")  # (B, 3, H, W)

    # Move to CPU for NumPy stats
    rgb_cpu = rgb.detach().cpu().numpy()

    # Global stats
    rgb_min = float(rgb_cpu.min())
    rgb_max = float(rgb_cpu.max())
    rgb_mean = float(rgb_cpu.mean())
    print("\n[STATS] Global:")
    print(f"  min        : {rgb_min:.4f}")
    print(f"  max        : {rgb_max:.4f}")
    print(f"  mean       : {rgb_mean:.4f}")

    # Per-channel stats
    # shape: [B, 3, H, W] -> [3, H, W] for B=1
    rgb_ch = rgb_cpu[0]
    ch_names = ["R", "G", "B"]
    print("\n[STATS] Per-channel (B=0):")
    for i, name in enumerate(ch_names):
        ch = rgb_ch[i]
        print(
            f"  {name}: min={ch.min():.4f}, max={ch.max():.4f}, "
            f"mean={ch.mean():.4f}, std={ch.std():.4f}"
        )

    # A few random pixels (just to see variety)
    H, W = rgb_ch.shape[1], rgb_ch.shape[2]
    print("\n[CHECK] Random pixels (R,G,B) at some coords:")
    rng = np.random.default_rng(42)
    for _ in range(5):
        y = int(rng.integers(0, H))
        x = int(rng.integers(0, W))
        r = rgb_ch[0, y, x]
        g = rgb_ch[1, y, x]
        b = rgb_ch[2, y, x]
        print(f"  (x={x:3d}, y={y:3d}) -> ({r:.3f}, {g:.3f}, {b:.3f})")

    # ---------------------------------------------------------------
    # 4. Optional: Save an image to disk
    # ---------------------------------------------------------------
    if SAVE_IMAGE:
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # Convert [3, H, W] in [0,1] -> [H, W, 3] in [0,255] uint8
            img = np.clip(rgb_ch.transpose(1, 2, 0) * 255.0, 0, 255).astype(
                np.uint8
            )

            out_path = os.path.join(OUTPUT_DIR, "teko_camera_sample.png")
            from PIL import Image

            Image.fromarray(img).save(out_path)
            print(f"\n[INFO] Saved sample frame to: {out_path}")
        except Exception as e:
            print(f"[WARN] Failed to save image: {e}")

    # ---------------------------------------------------------------
    # 5. Short stepping loop (optional)
    # ---------------------------------------------------------------
    print("\n[INFO] Stepping a few frames with zero actions...")
    num_steps = 20
    actions = torch.zeros(
        (env.scene.cfg.num_envs, 2), device=env.device
    )  # [v=0, w=0]

    for i in range(num_steps):
        # Some IsaacLab versions return (obs, rew, terminated, truncated, info)
        # but we don't need them here; we focus on camera data
        env.step(actions)
        app.update()
        time.sleep(0.02)

    print("[INFO] Camera diagnostics complete. Closing app.")
    app.close()


if __name__ == "__main__":
    main(headless=False)
