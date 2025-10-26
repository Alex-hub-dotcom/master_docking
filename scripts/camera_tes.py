print("\n✅ TEKO camera_tes.py foi iniciado!\n")
# SPDX-License-Identifier: BSD-3-Clause
"""
Simple camera test script for the TEKO environment.

This script instantiates the TEKO environment, runs a few steps,
and prints the shape of the RGB images returned by the tiled camera.

Usage:
    ./isaaclab.sh -p /workspace/teko/scripts/camera_test.py --enable_cameras
"""

from isaaclab.app import AppLauncher
import torch

# -----------------------------------------------------------------------------
# Launch Isaac Lab with camera support
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(headless=False)  # set to True if you don't need the GUI
app = app_launcher.app

# -----------------------------------------------------------------------------
# Import the TEKO environment and config
# -----------------------------------------------------------------------------
from teko.tasks.direct.teko.teko_env_cfg import TekoEnvCfg
from teko.tasks.direct.teko.teko_env import TekoEnv

# -----------------------------------------------------------------------------
# Initialize environment
# -----------------------------------------------------------------------------
cfg = TekoEnvCfg()
env = TekoEnv(cfg)

# Reset environment before running
env.reset()

print("[INFO] TEKO environment initialized successfully.")
print("[INFO] Starting camera test...")

# -----------------------------------------------------------------------------
# Run a few simulation steps and print RGB shapes
# -----------------------------------------------------------------------------
for step_idx in range(200):
    # Zero action (stop motors)
    actions = torch.zeros((1, 2), device=env.device)

    # Apply action and step simulation
    env.step(actions)

    # Get observations
    obs = env._get_observations()

    # Extract and print RGB shape
    rgb = obs.get("policy", None)
    if rgb is not None:
        print(f"Step {step_idx:02d} | RGB shape: {tuple(rgb.shape)} | min={rgb.min():.3f}, max={rgb.max():.3f}")
    else:
        print(f"Step {step_idx:02d} | No RGB data received.")

print("[INFO] Camera test finished successfully.")

# -----------------------------------------------------------------------------
# Close simulation app cleanly
# -----------------------------------------------------------------------------
app.close()
