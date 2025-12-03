#!/usr/bin/env python3
"""
TEKO Camera Preview - Capture single frame to verify what the robot sees.
"""

import argparse
from isaaclab.app import AppLauncher

# Must parse args before AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # CRITICAL: enable cameras

app = AppLauncher(args)
sim = app.app

# Now import after AppLauncher
import torch
from PIL import Image
from teko.tasks.direct.teko.teko_env import TekoEnv, TekoEnvCfg

# ---------------------------------------
# Create environment (1 env only)
# ---------------------------------------
cfg = TekoEnvCfg()
cfg.scene.num_envs = 1
cfg.enable_curriculum = False

env = TekoEnv(cfg=cfg)

# ---------------------------------------
# Step a few times to let camera initialize
# ---------------------------------------
obs_dict, _ = env.reset()

for _ in range(10):
    action = torch.zeros((1, 2), device=env.device)
    obs_dict, _, _, _, _ = env.step(action)

# ---------------------------------------
# Get grayscale stacked frames
# ---------------------------------------
img = obs_dict["rgb"][0]      # (4, 84, 84)
print(f"Observation shape: {img.shape}")
print(f"Value range: [{img.min():.3f}, {img.max():.3f}]")

# Save all 4 frames
for i in range(img.shape[0]):
    frame = img[i]
    frame_np = (frame * 255).cpu().numpy().astype("uint8")
    Image.fromarray(frame_np, mode='L').save(f"preview_frame_{i}.png")
    print(f"Saved: preview_frame_{i}.png")

# Also save the last frame as main preview
last_frame = img[-1]
last_frame_np = (last_frame * 255).cpu().numpy().astype("uint8")
Image.fromarray(last_frame_np, mode='L').save("preview_frame.png")
print("Saved: preview_frame.png (last frame)")

# ---------------------------------------
# Cleanup
# ---------------------------------------
env.close()
sim.close()

print("\n✅ Done! Check preview_frame.png")