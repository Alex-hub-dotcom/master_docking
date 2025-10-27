# SPDX-License-Identifier: BSD-3-Clause
"""
Fluid movement test for TEKO robot.
------------------------------------
Drives the robot forward while applying a slight angular difference
between left and right wheels to create a smooth circular trajectory.
Now compatible with Isaac Lab 0.47.1 and TekoEnv (camera + LiDAR).
"""

from isaaclab.app import AppLauncher
import time
import torch


# Motion parameters
LINEAR_SPEED = 1.2          # Forward linear velocity (m/s equivalent)
ANGULAR_FACTOR = 0.7        # <1.0 = turn left, >1.0 = turn right
DURATION = 50.0             # Total runtime (seconds)
STEP_HZ = 30                # Simulation step frequency

def main(headless=False):
    # ------------------------------------------------------------------
    # Launch Isaac Lab
    # ------------------------------------------------------------------
    app = AppLauncher(headless=headless).app

    # Import environment after Isaac Sim initializes
    from teko.tasks.direct.teko.teko_env import TekoEnv
    from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg

    # ------------------------------------------------------------------
    # Initialize environment
    # ------------------------------------------------------------------
    cfg = TekoEnvCfg()
    env = TekoEnv(cfg)
    env.reset()
    print("[INFO] TEKO environment ready.")

    # ------------------------------------------------------------------
    # Define constant circular motion
    # ------------------------------------------------------------------
    N = cfg.scene.num_envs
    left_speed = LINEAR_SPEED * 1.0
    right_speed = LINEAR_SPEED * ANGULAR_FACTOR
    actions = torch.tensor([[left_speed, right_speed]], device=env.device).repeat(N, 1)

    print(f"[INFO] TEKO circular motion test started — L={left_speed:.2f}, R={right_speed:.2f}")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        step = 0
        while app.is_running() and time.time() - t0 < DURATION:
            env.step(actions)
            app.update()

            # Log progress every ~1 second
            if step % STEP_HZ == 0:
                obs = env._get_observations()
                rgb_ok = "rgb" in obs and obs["rgb"].sum() > 0
                lidar_ok = "lidar" in obs and obs["lidar"].sum() > 0
                print(f"Step {step:03d} | RGB={rgb_ok} | LiDAR={lidar_ok} | Time={time.time()-t0:.1f}s")

            step += 1
    except KeyboardInterrupt:
        print("[INFO] Interrupted manually.")
    finally:
        print("[INFO] Shutting down simulation.")
        app.close()


if __name__ == "__main__":
    main(headless=False)
