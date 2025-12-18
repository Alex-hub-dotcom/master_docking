#!/usr/bin/env python3
"""
Debug script: visualize arena boundaries and robot body boxes.

- Runs TEKO env with N parallel envs
- Shows:
    * red arena boundaries (|x| = arena_half_x, |y| = arena_half_y)
    * green rectangular footprint around ACTIVE robot
    * red rectangular footprint around STATIC robot
- No camera image capture or saving.

Run with:

/workspace/isaaclab/_isaac_sim/python.sh \
    /workspace/teko/scripts/debug_show_boundaries.py
"""

import time
import torch

# 1) Launch Isaac app FIRST (with cameras enabled, since TekoEnv always creates a Camera)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(
    headless=False,
    enable_cameras=True,   # required because TekoEnv always creates a Camera sensor
)
app = app_launcher.app

# 2) Only now import TEKO / IsaacLab stuff
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env import TekoEnv


def main():
    # --- Configure env for debugging ---
    cfg = TekoEnvCfg()

    # Number of parallel envs you want to see (change if you like)
    cfg.scene.num_envs = 4

    # No curriculum needed for this visualization
    cfg.enable_curriculum = False

    # Make sure visual debug helpers are ON
    cfg.debug_boundaries = True      # red arena lines
    cfg.debug_robot_boxes = True     # green (active) + red (static) body rectangles

    # Create env with GUI rendering
    env = TekoEnv(cfg=cfg, render_mode="human")
    obs, _ = env.reset()

    num_envs = cfg.scene.num_envs
    action_dim = cfg.action_space[0]

    # Zero actions (robot stays mostly still, just for visualization)
    zero_action = torch.zeros((num_envs, action_dim), device=env.device)

    print("[DEBUG] Environment created. You should see:")
    print("       - Red rectangle marking arena boundaries")
    print("       - Green box around ACTIVE robot body")
    print("       - Red box around STATIC robot body (collision penalty zone)")
    print("       Close the Isaac window to stop.")

    # Main loop: keep stepping while the app window is open
    while app.is_running():
        obs, rew, terminated, truncated, info = env.step(zero_action)
        # small sleep just to avoid pegging CPU/GPU unnecessarily
        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()
    app.close()
