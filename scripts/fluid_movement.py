# SPDX-License-Identifier: BSD-3-Clause
"""
Circular motion test for TEKO robot.
------------------------------------
Drives the robot forward while applying a slight angular difference
between left and right wheels to create a smooth circular trajectory.
"""

from isaaclab.app import AppLauncher
import time
import torch

# Motion parameters
LINEAR_SPEED = 1.2           # Base forward velocity
ANGULAR_FACTOR = 0.7         # <1.0 turns left, >1.0 turns right
DURATION = 50.0              # Total runtime (seconds)

def main(headless=False):
    # Launch Isaac Lab
    app = AppLauncher(headless=headless).app

    # Import environment after app initialization
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # Initialize environment
    cfg = TekoEnvCfg()
    env = TekoEnv(cfg)
    env.reset()

    # Build action tensor (L, R)
    # Slight difference between left/right creates circular motion
    N = cfg.scene.num_envs
    left_speed = LINEAR_SPEED * 1.0
    right_speed = LINEAR_SPEED * ANGULAR_FACTOR
    actions = torch.tensor([[left_speed, right_speed]], device=env.device).repeat(N, 1)

    print(f"[INFO] TEKO circular motion test started — L={left_speed:.2f}, R={right_speed:.2f}")

    # Run simulation loop
    t0 = time.time()
    try:
        while app.is_running() and time.time() - t0 < DURATION:
            env.step(actions)
            app.update()
    finally:
        print("[INFO] Shutting down simulation.")
        app.close()

if __name__ == "__main__":
    main(headless=False)
