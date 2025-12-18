#!/usr/bin/env python3
"""
Sanity check: wall + static-body collision logic.

What it does
------------
- Creates a single TEKO env with:
    * red arena boundaries
    * green active chassis box
    * red static chassis box
- Applies a constant action (default: drive towards the static robot)
- Every step prints:
    * step index
    * robot local (x, y) in arena
    * inside_static_box flag
    * speed
    * out_of_bounds flag
    * terminated flag

Run with:

/workspace/isaaclab/_isaac_sim/python.sh \
    /workspace/teko/scripts/debug_collision_sanity.py
"""

import time
import torch

from isaaclab.app import AppLauncher

# Launch Isaac with GUI and cameras (TEKO always spawns a camera)
app_launcher = AppLauncher(
    headless=False,
    enable_cameras=True,
)
app = app_launcher.app

from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env import TekoEnv


def main():
    # --- Configure env for debugging ---
    cfg = TekoEnvCfg()
    cfg.scene.num_envs = 1
    cfg.enable_curriculum = False

    # Visual debug helpers
    cfg.debug_boundaries = True       # red arena lines
    # if you added this field in cfg:
    cfg.debug_robot_boxes = True      # green (active) + red (static) body boxes

    # Create env
    env = TekoEnv(cfg=cfg, render_mode="human")
    obs, _ = env.reset()

    num_envs = cfg.scene.num_envs
    action_dim = cfg.action_space[0]

    # -------- ACTION TO TEST --------
    # v<0 tends to move the robot "forward" towards +X in world
    # (because the robot is spawned rotated 180 deg).
    forward_to_static = torch.zeros((num_envs, action_dim), device=env.device)
    forward_to_static[:, 0] = -0.8  # v_cmd
    forward_to_static[:, 1] = 0.0   # w_cmd

    # You can later try different actions, e.g. towards a wall, by changing this tensor.
    action = forward_to_static
    # --------------------------------

    print("\n[DEBUG] Starting sanity check. Drive into the static red box.")
    print("       Watch the green (active) and red (static) boxes + console logs.\n")

    step = 0
    max_steps = 600

    while app.is_running() and step < max_steps:
        obs, rew, terminated, truncated, info = env.step(action)
        step += 1

        # --- Recompute the same logic as in _get_dones (for one env) ---

        # Robot global position
        robot_pos_global = env.robot.data.root_pos_w  # [N, 3]
        env_origins = env.scene.env_origins           # [N, 3]
        robot_pos_local = robot_pos_global - env_origins

        # Out-of-bounds check
        hx = float(env._arena_half_x)
        hy = float(env._arena_half_y)
        out_of_bounds = (
            (torch.abs(robot_pos_local[:, 0]) > hx) |
            (torch.abs(robot_pos_local[:, 1]) > hy)
        )

        # Static robot root (global)
        static_root_pos = env.goal_positions          # [N, 3]
        diff = robot_pos_global - static_root_pos
        dx = diff[:, 0]
        dy = diff[:, 1]

        half_len_static = 0.5 * env._static_body_length
        half_wid_static = 0.5 * env._static_body_width

        inside_static_box = (
            (dx.abs() <= half_len_static) &
            (dy.abs() <= half_wid_static)
        )

        # Speed (for collision gating)
        lin_vel = env.robot.data.root_lin_vel_w
        speed = torch.norm(lin_vel[:, :2], dim=-1)

        # Log for env 0
        print(
            f"step={step:03d} | "
            f"loc=({robot_pos_local[0,0]:+.2f}, {robot_pos_local[0,1]:+.2f}) | "
            f"inside_static_box={bool(inside_static_box[0])} | "
            f"speed={speed[0]:.2f} | "
            f"OOB={bool(out_of_bounds[0])} | "
            f"terminated={bool(terminated[0])}"
        )

        if terminated.any():
            print("\n[DEBUG] Episode terminated. "
                  "If this happened when the green box entered the red box with non-zero speed, "
                  "the STATIC collision logic is working.")
            break

        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()
    app.close()
